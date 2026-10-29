"""
Source of truth for notebooks/train_f1_model_colab.ipynb.

Edit the cells here, then regenerate the notebook:
    python notebooks/make_notebook.py
"""
import json
import os

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")})

def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.strip("\n")})

md(r"""
# 🏎️ F1 Race Prediction — Gerçek Veriyle Model Eğitimi (Google Colab)

Bu notebook, `models/ml_predictor.py` içindeki sentetik eğitim verisinin yerine **gerçek F1 sonuçlarıyla** (FastF1) bir model eğitir.

**Kullanım**
1. `Runtime → Change runtime type → T4 GPU` (XGBoost GPU'yu kullanır; RF/GB zaten CPU'da çalışır — veri küçük olduğu için CPU runtime da yeterli)
2. `Runtime → Run all`
3. Sonunda `f1_model.joblib` ve `f1_training_data.csv` otomatik indirilir → projede `models/trained/` klasörüne koy.

⏱️ Veri indirme birkaç dakika sürer. Cache'i kalıcı yapmak için aşağıda `USE_DRIVE = True` yap; sonraki çalıştırmalar dakikalar sürer.

**Değerlendirme:** Walk-forward — her yarış sadece kendinden önceki verilerle eğitilen modelle tahmin edilir. Yapılandırma 2025'e göre seçilir, 2026 test olarak raporlanır. Rastgele ayırma geleceği sızdırır, bu yüzden kullanılmıyor.
""")

code(r"""
!pip install -q fastf1 xgboost joblib
!nvidia-smi -L || echo "GPU yok - CPU ile devam"
""")

code(r"""
import os, time, warnings, logging, subprocess
import numpy as np
import pandas as pd
import fastf1

warnings.filterwarnings('ignore')
fastf1.set_log_level('ERROR')
logging.disable(logging.WARNING)   # FastF1'in INFO/WARNING kalabalığını gizle

YEARS = list(range(2018, 2027))   # 2026'nın tamamlanmış yarışları da dahil
USE_DRIVE = False                 # True → cache Google Drive'da kalır, tekrar indirme olmaz

CACHE_DIR = '/content/fastf1_cache'
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    CACHE_DIR = '/content/drive/MyDrive/f1_cache'
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

HAS_GPU = subprocess.run(['nvidia-smi'], capture_output=True).returncode == 0
DEVICE = 'cuda' if HAS_GPU else 'cpu'
print('XGBoost device:', DEVICE)
""")

md("## 1. Veri indirme (Jolpica/Ergast: yarış + sıralama sonuçları, FastF1: hava durumu)")

code(r"""
import requests

try:
    from fastf1 import _api as f1api      # FastF1 >= 3.5
except ImportError:
    from fastf1 import api as f1api

API = 'https://api.jolpi.ca/ergast/f1'   # Ergast'ın devamı; sezonu tek seferde, sayfa sayfa verir


def jolpica(path):
    # Bir sezonun tüm sonuçlarını sayfalayarak çek → {round: {...race, 'rows': [...]}}
    races, offset = {}, 0
    while True:
        for attempt in range(8):
            r = requests.get(f'{API}/{path}.json', params={'limit': 100, 'offset': offset}, timeout=30)
            if r.status_code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            r.raise_for_status()
            break
        data = r.json()['MRData']
        for race in data['RaceTable']['Races']:
            entry = races.setdefault(int(race['round']), {**race, 'rows': []})
            entry['rows'] += race.get('Results', race.get('QualifyingResults', []))
        offset += 100
        if offset >= int(data['total']):
            return races
        time.sleep(0.5)


def lap_seconds(s):
    if not s:
        return np.nan
    m, sec = s.split(':') if ':' in s else ('0', s)
    return int(m) * 60 + float(sec)


def is_wet_race(year, rnd):
    # Hava verisi F1 canlı zamanlamadan (Ergast limitine takılmaz); yoksa kuru varsay
    try:
        wd = f1api.weather_data(fastf1.get_session(year, rnd, 'R').api_path)
        rain = pd.Series(wd['Rainfall']).map(lambda v: str(v).lower() in ('true', '1'))
        return int(rain.mean() > 0.1)
    except Exception as e:
        weather_fails.append(f'{year} R{rnd}: {type(e).__name__}')
        return 0


now = pd.Timestamp.now()
frames, weather_fails = [], []
for year in YEARS:
    results = jolpica(f'{year}/results')
    quali = jolpica(f'{year}/qualifying')
    for rnd, race in sorted(results.items()):
        if pd.Timestamp(race['date']) >= now - pd.Timedelta(days=1):
            continue
        best = {q['Driver']['driverId']: pd.Series([lap_seconds(q.get(k)) for k in ('Q1', 'Q2', 'Q3')]).min()
                for q in quali.get(rnd, {'rows': []})['rows']}
        quali_pos = {q['Driver']['driverId']: int(q['position']) for q in quali.get(rnd, {'rows': []})['rows']}
        pole = np.nanmin(list(best.values())) if best else np.nan
        rows = pd.DataFrame([{
            'year': year,
            'round': rnd,
            'event': race['raceName'],
            'circuit': race['Circuit']['circuitId'],
            'date': pd.Timestamp(race['date']),
            'driver': x['Driver']['driverId'],
            'team': x['Constructor']['constructorId'],
            'grid': int(x['grid']),
            'finish': int(x['position']),
            'points': float(x['points']),
            'status': x['status'],
            'quali_gap_pct': (best.get(x['Driver']['driverId'], np.nan) / pole - 1) * 100,
            'quali_pos': quali_pos.get(x['Driver']['driverId'], np.nan),
        } for x in race['rows']])
        rows['is_wet'] = is_wet_race(year, rnd)
        frames.append(rows)
        print(f"✅ {year} R{rnd:02d} {race['raceName']}" + ('  🌧️' if rows['is_wet'].iat[0] else ''))

raw = pd.concat(frames, ignore_index=True)
raw.to_csv('f1_raw_results.csv', index=False)
print(raw.shape)
if weather_fails:
    print(f'⚠️ {len(weather_fails)} yarışta hava verisi alınamadı (kuru varsayıldı):', weather_fails[:5])
""")

md(r"""
## 2. Özellik mühendisliği

Tüm özellikler **sadece o yarıştan önceki** verilerle hesaplanır (`shift(1)`), yani model tahmin anında bilinmeyen bir bilgiyi görmez. Aynı özellikler 2026'da canlı tahmin için de FastF1'den hesaplanabilir.
""")

code(r"""
# Takım isim değişikliklerini tek bir "soy" altında topla
TEAM_ALIASES = {
    'toro_rosso': 'rb', 'alphatauri': 'rb',
    'force_india': 'aston_martin', 'racing_point': 'aston_martin',
    'renault': 'alpine',
    'alfa': 'audi', 'sauber': 'audi',
}

df = raw.dropna(subset=['finish']).copy()
df['team'] = df['team'].replace(TEAM_ALIASES)
df['grid'] = df['grid'].fillna(20).replace(0, 20).clip(1, 22)          # pit lane start → arka sıra
df['dnf'] = (~df['status'].str.contains('Finished|Lap', case=False, na=False)).astype(int)
df['race_id'] = df['year'] * 100 + df['round']
df['gain'] = df['grid'] - df['finish']
df = df.sort_values(['race_id', 'finish']).reset_index(drop=True)

df['quali_gap_pct'] = df['quali_gap_pct'].fillna(df.groupby('race_id')['quali_gap_pct'].transform('max')).fillna(3.0)


def past_rolling(frame, by, col, window):
    return frame.groupby(by)[col].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())

# --- Pilot formu ---
df['driver_form_5'] = past_rolling(df, 'driver', 'finish', 5)
df['driver_points_5'] = past_rolling(df, 'driver', 'points', 5)
df['driver_dnf_10'] = past_rolling(df, 'driver', 'dnf', 10)
df['driver_gain_5'] = past_rolling(df, 'driver', 'gain', 5)
df['driver_experience'] = df.groupby('driver').cumcount()
df['driver_track_avg'] = df.groupby(['driver', 'circuit'])['finish'].transform(lambda s: s.shift(1).expanding().mean())

# --- Takım formu (takım-yarış seviyesinde, aynı yarıştaki takım arkadaşı sızmasın) ---
team_race = (df.groupby(['race_id', 'team'])
               .agg(t_finish=('finish', 'mean'), t_points=('points', 'sum'), t_dnf=('dnf', 'mean'))
               .reset_index().sort_values('race_id'))
team_race['team_form_5'] = past_rolling(team_race, 'team', 't_finish', 5)
team_race['team_points_5'] = past_rolling(team_race, 'team', 't_points', 5)
team_race['team_dnf_10'] = past_rolling(team_race, 'team', 't_dnf', 10)
team_race['team_season_form'] = (team_race.assign(t_year=team_race['race_id'] // 100)
                                 .groupby(['t_year', 'team'])['t_finish']
                                 .transform(lambda s: s.shift(1).expanding().mean()))
df = df.merge(team_race[['race_id', 'team', 'team_form_5', 'team_points_5', 'team_dnf_10', 'team_season_form']],
              on=['race_id', 'team'], how='left')

# --- Sezon içi şampiyona durumu (yarıştan önce) ---
df['season_points_before'] = df.groupby(['year', 'driver'])['points'].transform(lambda s: s.shift(1).cumsum()).fillna(0)
df['season_rank_before'] = df.groupby('race_id')['season_points_before'].rank(ascending=False, method='min')

# --- Pist karakteri: önceki yıllarda bu pistte ortalama pozisyon değişimi (sollama kolaylığı) ---
race_change = (df[df['dnf'] == 0].assign(abs_gain=lambda d: d['gain'].abs())
                 .groupby(['race_id', 'circuit'])['abs_gain'].mean().reset_index().sort_values('race_id'))
race_change['circuit_pos_change'] = race_change.groupby('circuit')['abs_gain'].transform(lambda s: s.shift(1).expanding().mean())
df = df.merge(race_change[['race_id', 'circuit_pos_change']], on='race_id', how='left')

# --- Eksik değerler (ilk yarışlar, yeni pilotlar) ---
df['driver_form_5'] = df['driver_form_5'].fillna(df['grid'])
df['driver_track_avg'] = df['driver_track_avg'].fillna(df['driver_form_5'])
df['team_form_5'] = df['team_form_5'].fillna(12)
for c in ['driver_points_5', 'team_points_5', 'driver_gain_5']:
    df[c] = df[c].fillna(0)
for c in ['driver_dnf_10', 'team_dnf_10']:
    df[c] = df[c].fillna(df['dnf'].mean())
df['circuit_pos_change'] = df['circuit_pos_change'].fillna(df['circuit_pos_change'].median())

# --- Yeni özellikler ---
# Sezon içi form: kural değişikliği olan sezonlarda (2022, 2026) eski formun etkisini azaltır
df['driver_season_form'] = (df.groupby(['year', 'driver'])['finish']
                              .transform(lambda s: s.shift(1).expanding().mean())
                              .fillna(df['driver_form_5']))
df['team_season_form'] = df['team_season_form'].fillna(df['team_form_5'])
# Takım arkadaşına göre: aynı araçla ne kadar hızlı (sıralama turu ve grid)
team_key = [df['race_id'], df['team']]
df['teammate_quali_delta'] = df['quali_gap_pct'] - df.groupby(team_key)['quali_gap_pct'].transform('mean')
df['grid_vs_teammate'] = df['grid'] - df.groupby(team_key)['grid'].transform('mean')
# Grid ile aracın normal temposu arasındaki fark (kötü sıralama → toparlama potansiyeli)
df['grid_vs_team_form'] = df['grid'] - df['team_form_5']
df['reg_era_2026'] = (df['year'] >= 2026).astype(int)
# Grid cezası: sıralamadaki yerinden daha geride başladıysa kaybedilen pozisyon sayısı
df['quali_pos'] = df['quali_pos'].fillna(df['grid'])
df['grid_penalty'] = (df['grid'] - df['quali_pos']).clip(lower=0)

BASE_FEATURES = [
    'grid', 'quali_gap_pct', 'is_wet', 'round',
    'driver_form_5', 'driver_points_5', 'driver_dnf_10', 'driver_gain_5',
    'driver_experience', 'driver_track_avg',
    'team_form_5', 'team_points_5', 'team_dnf_10',
    'season_points_before', 'season_rank_before', 'circuit_pos_change',
]
FEATURES = BASE_FEATURES + [
    'driver_season_form', 'team_season_form', 'teammate_quali_delta',
    'grid_vs_teammate', 'grid_vs_team_form', 'reg_era_2026',
    'grid_penalty',
]
print(df[FEATURES].describe().T[['mean', 'min', 'max']])
""")

md(r"""
## 3. Walk-forward değerlendirme

Her yarış, **sadece o yarıştan önceki** verilerle eğitilen modelle tahmin edilir (2 yarışta bir yeniden eğitim). Böylece 2026 verisi de sezon ilerledikçe eğitime girer — simülatörün gerçek kullanımıyla aynı.

Denenen yapılandırmalar:
- **A** — mevcut model (eski özellikler, bitiş sırası hedefi)
- **B** — yeni özellikler + *delta* hedef (grid'e göre pozisyon değişimi) + yakın sezonlara ağırlık
- **C** — yeni özellikler + delta hedef
- **D** — yeni özellikler + bitiş sırası hedefi + yakın sezonlara ağırlık

Seçim 2025 (doğrulama) sonuçlarına göre yapılır, 2026 sonuçları sadece raporlanır.
""")

code(r"""
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

MODEL_NAMES = ['rf', 'gb', 'xgb']
HALF_LIFE = 2   # yıl: 2 yıl önceki sezon yarı ağırlık alır


def make_models():
    return {
        'rf': RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=5, n_jobs=-1, random_state=42),
        'gb': GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42),
        'xgb': xgb.XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
                                colsample_bytree=0.8, tree_method='hist', device=DEVICE, random_state=42),
    }


def fit_models(train, features, target_mode, half_life):
    y = train['finish'] - train['grid'] if target_mode == 'delta' else train['finish']
    w = None if half_life is None else 0.5 ** ((train['year'].max() - train['year']) / half_life)
    models = make_models()
    for m in models.values():
        m.fit(train[features], y, sample_weight=w)
    return models


def predict_models(models, frame, features, target_mode):
    offset = frame['grid'].values if target_mode == 'delta' else 0
    return {name: m.predict(frame[features]) + offset for name, m in models.items()}


def walk_forward(features, target_mode, half_life, eval_years=(2025, 2026), refit_every=2):
    races = sorted(df.loc[df['year'].isin(eval_years), 'race_id'].unique())
    out = []
    for i in range(0, len(races), refit_every):
        chunk = races[i:i + refit_every]
        models = fit_models(df[df['race_id'] < chunk[0]], features, target_mode, half_life)
        test = df[df['race_id'].isin(chunk)]
        preds = test[['race_id', 'year', 'driver', 'grid', 'finish']].copy()
        for name, p in predict_models(models, test, features, target_mode).items():
            preds[name] = p
        out.append(preds)
    res = pd.concat(out)
    res['ensemble'] = res[MODEL_NAMES].mean(axis=1)
    return res


def evaluate(frame, pred):
    f = frame.assign(pred=np.asarray(pred))
    sp, win, pod = [], [], []
    for _, r in f.groupby('race_id'):
        sp.append(spearmanr(r['pred'], r['finish']).correlation)
        win.append(r.loc[r['pred'].idxmin(), 'finish'] == 1)
        pod.append(len(set(r.nsmallest(3, 'pred')['driver']) & set(r.nsmallest(3, 'finish')['driver'])) / 3)
    return {
        'MAE': round((f['pred'] - f['finish']).abs().mean(), 3),
        'Spearman': round(np.nanmean(sp), 3),
        'Kazanan': round(np.mean(win), 3),
        'Podyum': round(np.mean(pod), 3),
    }


CONFIGS = {
    'A': (BASE_FEATURES, 'abs', None),
    'B': (FEATURES, 'delta', HALF_LIFE),
    'C': (FEATURES, 'delta', None),
    'D': (FEATURES, 'abs', HALF_LIFE),
}

rows, wf = [], {}
for cname, (feats, mode, hl) in CONFIGS.items():
    t0 = time.time()
    wf[cname] = walk_forward(feats, mode, hl)
    for model in MODEL_NAMES + ['ensemble']:
        for yr in (2025, 2026):
            part = wf[cname][wf[cname]['year'] == yr]
            rows.append({'config': cname, 'model': model, 'yıl': yr, **evaluate(part, part[model])})
    print(f'{cname} bitti ({time.time() - t0:.0f} sn)')

any_wf = next(iter(wf.values()))
for yr in (2025, 2026):
    part = any_wf[any_wf['year'] == yr]
    rows.append({'config': '-', 'model': 'baseline (grid)', 'yıl': yr, **evaluate(part, part['grid'])})

report = pd.DataFrame(rows).pivot_table(index=['config', 'model'], columns='yıl',
                                         values=['Spearman', 'MAE', 'Kazanan', 'Podyum'])
report
""")

code(r"""
# 2025 (doğrulama) Spearman'a göre en iyi yapılandırma + model
val = pd.DataFrame(rows)
val = val[(val['yıl'] == 2025) & (val['config'] != '-')].sort_values('Spearman', ascending=False)
BEST_CONFIG, BEST_MODEL = val.iloc[0]['config'], val.iloc[0]['model']
best_features, TARGET_MODE, BEST_HALF_LIFE = CONFIGS[BEST_CONFIG]
WEIGHTS = ({n: 1 / len(MODEL_NAMES) for n in MODEL_NAMES} if BEST_MODEL == 'ensemble'
           else {n: float(n == BEST_MODEL) for n in MODEL_NAMES})
print(f'Seçilen: config {BEST_CONFIG}, model {BEST_MODEL}, ağırlıklar {WEIGHTS}')

test_metrics = {r['model'] if r['config'] == '-' else f"{r['config']}/{r['model']}":
                {k: r[k] for k in ('MAE', 'Spearman', 'Kazanan', 'Podyum')}
                for r in rows if r['yıl'] == 2026}
pd.DataFrame(test_metrics).T.loc[['baseline (grid)', 'A/ensemble', f'{BEST_CONFIG}/{BEST_MODEL}']]
""")

md("## 4. Tüm veriyle son eğitim ve kaydetme")

code(r"""
import joblib, sklearn
import matplotlib.pyplot as plt
from datetime import datetime

final = fit_models(df, best_features, TARGET_MODE, BEST_HALF_LIFE)

imp = pd.Series(final['rf'].feature_importances_, index=best_features).sort_values()
imp.plot.barh(figsize=(7, 7), title='Random Forest - özellik önemi')
plt.tight_layout(); plt.show()

bundle = {
    'models': {'rf': final['rf'], 'gb': final['gb']},
    # XGBoost pickle'ı sürüm/GPU bağımlı → sürümden bağımsız JSON olarak sakla
    'xgb_json': bytes(final['xgb'].get_booster().save_raw('json')).decode(),
    'weights': WEIGHTS,
    'features': best_features,
    'target_mode': TARGET_MODE,
    'half_life': BEST_HALF_LIFE,
    'team_aliases': TEAM_ALIASES,
    'test_metrics': test_metrics,
    'trained_on': f"{df['year'].min()}-{df['year'].max()} ({df['race_id'].nunique()} yarış)",
    'created': datetime.now().isoformat(timespec='seconds'),
    'versions': {'sklearn': sklearn.__version__, 'xgboost': xgb.__version__, 'fastf1': fastf1.__version__},
}
joblib.dump(bundle, 'f1_model.joblib', compress=3)
df.to_csv('f1_training_data.csv', index=False)
print('Kaydedildi. Lokal kurulumda aynı sürümleri kullan:')
print(f"  pip install scikit-learn=={sklearn.__version__} xgboost=={xgb.__version__}")

from google.colab import files
files.download('f1_model.joblib')
files.download('f1_training_data.csv')
""")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_f1_model_colab.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
