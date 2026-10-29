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
            'car_number': str(x['number']),
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
## 1b. Antrenman temposu (FastF1, 2022+)

Yarış temposunun asıl göstergesi **uzun run** hızıdır: pilotun pit yapmadan arka arkaya attığı turlar. Buradan iki özellik çıkıyor: uzun run ortalamasının seansın en hızlısına göre farkı (`practice_pace_gap`) ve tek turun farkı (`practice_best_gap`).

Sadece 2022+ alınıyor: mevcut kural dönemi ve FastF1'in saatlik istek limiti (seans başına **tek** istek).

Bu adım yalnızca Colab Secrets üzerinden F1 TV Access girişi (`F1_USERNAME`, `F1_PASSWORD`) verildiğinde çalışır; aksi halde atlanır.
""")

code(r"""
# 2022 = mevcut kural döneminin başlangıcı; ayrıca FastF1 saatlik istek limitini aşmamak için
PRACTICE_YEARS = range(2022, 2027)

# Ekim 2026'dan beri canlı timing arşivi F1 TV Access girişi istiyor; FastF1 bunu
# F1_USERNAME / F1_PASSWORD env değişkenlerinden okur. Giriş yoksa her 2023+ seansı
# SessionNotAvailableError verir ve FastF1'in saatlik istek bütçesi boşa yanar.
try:
    from google.colab import userdata
    for key in ('F1_USERNAME', 'F1_PASSWORD'):
        if not os.environ.get(key):
            os.environ[key] = userdata.get(key)
except Exception:
    pass   # secret yok / Colab dışında

HAS_F1_LOGIN = bool(os.environ.get('F1_USERNAME') and os.environ.get('F1_PASSWORD'))
if not HAS_F1_LOGIN:
    PRACTICE_YEARS = range(0)
    print('ℹ️ Antrenman tempo verisi atlanıyor: F1 TV Access girişi gerekli (Colab Secrets: F1_USERNAME / F1_PASSWORD). Model bu veri olmadan da çalışıyor.')


def practice_session(year, rnd):
    # Sprint hafta sonlarında sadece FP1 yapılır
    sprint = ('sprint', 'sprint_shootout', 'sprint_qualifying')
    return 'FP1' if fastf1.get_event(year, rnd)['EventFormat'] in sprint else 'FP2'


def practice_pace(year, rnd):
    # Seans başına tek istek: tüm tur verisi tek çağrıda gelir
    laps_data, _ = f1api.timing_data(fastf1.get_session(year, rnd, practice_session(year, rnd)).api_path)
    laps = laps_data[laps_data['LapTime'].notna()].copy()
    laps['seconds'] = laps['LapTime'].dt.total_seconds()
    # Pit giriş/çıkış turları temsil edici değil
    laps['invalid'] = laps['PitInTime'].notna() | laps['PitOutTime'].notna()

    drivers = []
    for drv, dlaps in laps.groupby('Driver'):
        dlaps = dlaps.sort_values('NumberOfLaps')
        valid = dlaps[~dlaps['invalid']]
        if valid.empty:
            continue
        best = valid['seconds'].min()
        # Stint'ler pit çıkışında bölünür; bir stintte en az 5 geçerli tur varsa "uzun run"
        dlaps['stint'] = dlaps['PitOutTime'].notna().cumsum()
        long_run_laps = []
        for _, stint in dlaps.groupby('stint'):
            runs = stint[(~stint['invalid']) & (stint['seconds'] <= best * 1.07)]
            if len(runs) >= 5:
                long_run_laps.append(runs['seconds'])
        drivers.append({
            'year': year,
            'round': rnd,
            'car_number': str(drv),
            'long_run': pd.concat(long_run_laps).median() if long_run_laps else np.nan,
            'best': best,
        })

    pace = pd.DataFrame(drivers, columns=['year', 'round', 'car_number', 'long_run', 'best'])
    pace['practice_pace_gap'] = (pace['long_run'] / pace['long_run'].min() - 1) * 100
    pace['practice_best_gap'] = (pace['best'] / pace['best'].min() - 1) * 100
    return pace[['year', 'round', 'car_number', 'practice_pace_gap', 'practice_best_gap']]


practice_frames, practice_fails = [], []
for year, rnd in raw[['year', 'round']].drop_duplicates().itertuples(index=False):
    if year not in PRACTICE_YEARS:
        continue
    try:
        practice_frames.append(practice_pace(year, rnd))
        print(f'✅ {year} R{rnd:02d} antrenman temposu')
    except fastf1.exceptions.RateLimitExceededError:
        print('⚠️ FastF1 istek limiti doldu — kalan antrenman verisi atlanıyor')
        break
    except Exception as e:
        practice_fails.append(f'{year} R{rnd}: {type(e).__name__}')

practice = (pd.concat(practice_frames, ignore_index=True) if practice_frames
            else pd.DataFrame(columns=['year', 'round', 'car_number', 'practice_pace_gap', 'practice_best_gap']))
raw = raw.merge(practice, on=['year', 'round', 'car_number'], how='left')
# Boş antrenman verisinde sütunlar all-NaN/object kalır; feature hücresindeki median/rolling için float şart
raw[['practice_pace_gap', 'practice_best_gap']] = raw[['practice_pace_gap', 'practice_best_gap']].astype('float64')
raw.to_csv('f1_raw_results.csv', index=False)
print(raw.shape)
if practice_fails:
    print(f'⚠️ {len(practice_fails)} yarışta antrenman verisi alınamadı:', practice_fails[:5])
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
# Antrenman temposu: uzun run farkı, tek tur farkı ve son 5 yarıştaki ortalama
# has_practice ve form, doldurmadan ÖNCE hesaplanır (yarışta antrenman verisi var mı?)
df['has_practice'] = df['practice_pace_gap'].notna().astype(int)
df['practice_form_5'] = past_rolling(df, 'driver', 'practice_pace_gap', 5)
# Eksik antrenman verisi: önce o yarışın medyanı, sonra tüm verinin medyanı
for c in ['practice_pace_gap', 'practice_best_gap']:
    df[c] = df[c].fillna(df.groupby('race_id')[c].transform('median')).fillna(df[c].median())
df['practice_form_5'] = df['practice_form_5'].fillna(df['practice_pace_gap'].median())

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
    'practice_pace_gap', 'practice_best_gap', 'practice_form_5', 'has_practice',
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

md(r"""
## 3b. Galibiyet ve podyum olasılığı

Konum tahmininin yanına iki sınıflandırıcı eklenir: pilot yarışı kazanır mı (`finish = 1`) ve podyuma çıkar mı (`finish <= 3`). Olasılıklar her yarışta normalize edilir — kazanma olasılıklarının toplamı 1'e, podyum olasılıklarının toplamı 3'e eşitlenir — ve grid pozisyonuna dayalı ampirik bir temel ile karşılaştırılır. Değerlendirme yine walk-forward yapılır; LogLoss, Brier skoru ve favori isabeti (en yüksek olasılıklı pilotun hedefi tutturup tuttmadığı) 2025 ve 2026 için ayrı ayrı raporlanır.
""")

code(r"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, brier_score_loss

CLASS_TARGETS = {'win': 1, 'podium': 3}   # hedef: finish <= değer


def fit_classifiers(train, features, half_life):
    # Kazanma ve podyum sınıflandırıcıları; örnek ağırlığı fit_models ile aynı formül
    w = None if half_life is None else 0.5 ** ((train['year'].max() - train['year']) / half_life)
    clfs = {}
    for name, k in CLASS_TARGETS.items():
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=10, n_jobs=-1, random_state=42)
        clf.fit(train[features], train['finish'] <= k, sample_weight=w)
        clfs[name] = clf
    return clfs


def normalize(frame, prob, total):
    # Aynı yarıştaki olasılıklar toplamı 'total' olacak şekilde ölçeklenir,
    # sonra sıfır olasılıkların log_loss'u patlatmaması için [1e-3, 1 - 1e-3]'e kırpılır
    out = frame[['race_id']].assign(_p=np.asarray(prob, dtype=float))
    sums = out.groupby('race_id')['_p'].transform('sum')
    sizes = out.groupby('race_id')['_p'].transform('size')
    out['_p'] = np.where(sums > 0, out['_p'] * total / sums, total / sizes)
    return np.clip(out['_p'].to_numpy(), 1e-3, 1 - 1e-3)


def grid_baseline(train, frame, k):
    # Ampirik P(finish <= k | grid), add-one smoothing; görülmeyen grid → genel oran
    g = train.assign(_hit=(train['finish'] <= k).astype(int)).groupby('grid')['_hit'].agg(['sum', 'count'])
    rate = (g['sum'] + 1) / (g['count'] + 2)
    prob = frame['grid'].map(rate).fillna((train['finish'] <= k).mean())
    return normalize(frame, prob, k)   # toplam da hedef değeriyle aynı: win'de 1, podyumda 3


# 3. bölümdeki gibi walk-forward: her yarış sadece önceki verilerle eğitilen modelle değerlendirilir
class_races = sorted(df.loc[df['year'].isin((2025, 2026)), 'race_id'].unique())
model_out, base_out = [], []
for i in range(0, len(class_races), 2):
    chunk = class_races[i:i + 2]
    train = df[df['race_id'] < chunk[0]]
    test = df[df['race_id'].isin(chunk)]
    clfs = fit_classifiers(train, best_features, BEST_HALF_LIFE)
    for name, k in CLASS_TARGETS.items():
        cols = test[['race_id', 'year', 'driver', 'grid', 'finish']]
        model_out.append(cols.assign(target=name, prob=normalize(test, clfs[name].predict_proba(test[best_features])[:, 1], k)))
        base_out.append(cols.assign(target=name, prob=grid_baseline(train, test, k)))

model_preds = pd.concat(model_out, ignore_index=True)
baseline_preds = pd.concat(base_out, ignore_index=True)


def class_report(frame, label):
    # Favori isabeti: en yüksek olasılıklı pilot gerçekten kazandı mı / podyuma girdi mi
    rows = []
    for (target, year), r in frame.groupby(['target', 'year']):
        k = CLASS_TARGETS[target]
        favorites = r.groupby('race_id').apply(lambda g: g.loc[g['prob'].idxmax(), 'finish'] <= k)
        rows.append({'model': label, 'target': target, 'yıl': int(year),
                     'LogLoss': round(log_loss(r['finish'] <= k, r['prob']), 3),
                     'Brier': round(brier_score_loss(r['finish'] <= k, r['prob']), 3),
                     'Favori isabet': round(favorites.mean(), 3)})
    return pd.DataFrame(rows)


class_table = pd.concat([class_report(model_preds, 'model'), class_report(baseline_preds, 'baseline (grid)')],
                        ignore_index=True)
class_metrics = {f"{r['model']}/{r['target']}": {k: r[k] for k in ('LogLoss', 'Brier', 'Favori isabet')}
                 for _, r in class_table[class_table['yıl'] == 2026].iterrows()}
class_table
""")

md("## 4. Tüm veriyle son eğitim ve kaydetme")

code(r"""
import joblib, sklearn
import matplotlib.pyplot as plt
from datetime import datetime

final = fit_models(df, best_features, TARGET_MODE, BEST_HALF_LIFE)
final_clf = fit_classifiers(df, best_features, BEST_HALF_LIFE)

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
    'classifiers': final_clf,
    'class_metrics': class_metrics,
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
