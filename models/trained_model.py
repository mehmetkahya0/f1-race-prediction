"""
Trained ML model loader for F1 Race Simulation.

Loads the ensemble trained on real F1 results (see
notebooks/train_f1_model_colab.ipynb) and rebuilds the same features at
prediction time from the bundled training history:
- Driver form (last 5 races and current season), DNF rate, track history
- Team form (last 5 races and current season), team DNF rate
- Pace relative to the teammate (qualifying gap and grid)
- Championship standing before the race
- Circuit overtaking profile (average position change)
"""

import os
import re
import unicodedata
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

TRAINED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained')
MODEL_PATH = os.path.join(TRAINED_DIR, 'f1_model.joblib')
HISTORY_PATH = os.path.join(TRAINED_DIR, 'f1_training_data.csv')

# Simulator track city -> Jolpica/Ergast circuitId
CIRCUIT_IDS = {
    'Melbourne': 'albert_park', 'Shanghai': 'shanghai', 'Suzuka': 'suzuka',
    'Sakhir': 'bahrain', 'Jeddah': 'jeddah', 'Miami': 'miami',
    'Montreal': 'villeneuve', 'Monte Carlo': 'monaco', 'Barcelona': 'catalunya',
    'Spielberg': 'red_bull_ring', 'Silverstone': 'silverstone', 'Spa': 'spa',
    'Budapest': 'hungaroring', 'Zandvoort': 'zandvoort', 'Monza': 'monza',
    'Madrid': 'madring', 'Baku': 'baku', 'Singapore': 'marina_bay',
    'Austin': 'americas', 'Mexico City': 'rodriguez', 'São Paulo': 'interlagos',
    'Las Vegas': 'vegas', 'Lusail': 'losail', 'Abu Dhabi': 'yas_marina',
}

# Simulator team name -> Jolpica/Ergast constructorId (after training aliases)
TEAM_IDS = {
    'red_bull_racing': 'red_bull',
    'racing_bulls': 'rb',
}


def _slug(text: str) -> str:
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


class TrainedF1Model:
    """Ensemble (RF + GB + XGBoost) trained on 2018+ real race results."""

    def __init__(self, bundle: Dict, history):
        import xgboost as xgb

        self.features: List[str] = bundle['features']
        self.weights: Dict[str, float] = bundle['weights']
        # 'delta': models predict position change relative to the grid slot
        self.target_mode: str = bundle.get('target_mode', 'abs')
        self.metadata = {k: bundle.get(k) for k in ('trained_on', 'created', 'versions', 'test_metrics')}

        self.models = dict(bundle['models'])
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(bytearray(bundle['xgb_json'], 'utf-8'))
        self.models['xgb'] = xgb_model

        self.history = history.sort_values(['race_id', 'finish']).reset_index(drop=True)
        self.season = int(self.history['year'].max())
        self._known_drivers = set(self.history['driver'])

        latest = self.history[self.history['year'] == self.season]
        self._gap_by_grid = latest.groupby('grid')['quali_gap_pct'].median()
        self._mean_dnf = float(self.history['dnf'].mean())

        finishers = self.history[self.history['dnf'] == 0]
        self._circuit_change = finishers.assign(abs_gain=finishers['gain'].abs()) \
            .groupby('circuit')['abs_gain'].mean()

    @property
    def feature_importances(self) -> Dict[str, float]:
        return dict(zip(self.features, self.models['rf'].feature_importances_))

    def driver_id(self, driver) -> Optional[str]:
        parts = _slug(driver.name).split('_')
        for candidate in (parts[-1], '_'.join(parts)):
            if candidate in self._known_drivers:
                return candidate
        return None

    def _team_id(self, team_name: str) -> str:
        slug = _slug(team_name)
        return TEAM_IDS.get(slug, slug)

    def _team_form(self, team_id: str) -> Dict[str, float]:
        team_rows = self.history[self.history['team'] == team_id]
        if team_rows.empty:
            return {'team_form_5': 12.0, 'team_points_5': 0.0, 'team_dnf_10': self._mean_dnf,
                    'team_season_form': 12.0}
        per_race = team_rows.groupby('race_id').agg(
            t_finish=('finish', 'mean'), t_points=('points', 'sum'), t_dnf=('dnf', 'mean'))
        this_season = per_race[per_race.index // 100 == self.season]['t_finish']
        team_form = per_race['t_finish'].tail(5).mean()
        return {
            'team_form_5': team_form,
            'team_points_5': per_race['t_points'].tail(5).mean(),
            'team_dnf_10': per_race['t_dnf'].tail(10).mean(),
            'team_season_form': this_season.mean() if len(this_season) else team_form,
        }

    def build_features(self, grid_positions: List, driver_teams: Dict, track,
                       is_wet: bool, round_no: int):
        import pandas as pd

        circuit = CIRCUIT_IDS.get(track.city)
        season_rows = self.history[self.history['year'] == self.season]
        season_points = season_rows.groupby('driver')['points'].sum()

        rows = []
        for i, driver in enumerate(grid_positions):
            grid = i + 1
            did = self.driver_id(driver)
            past = self.history[self.history['driver'] == did] if did else self.history.iloc[0:0]
            team = driver_teams.get(driver)
            team_name = team.name if team is not None else driver.team

            if past.empty:
                driver_form = float(grid)
                row = {'driver_points_5': 0.0, 'driver_dnf_10': self._mean_dnf,
                       'driver_gain_5': 0.0, 'driver_experience': 0}
            else:
                driver_form = past['finish'].tail(5).mean()
                row = {
                    'driver_points_5': past['points'].tail(5).mean(),
                    'driver_dnf_10': past['dnf'].tail(10).mean(),
                    'driver_gain_5': past['gain'].tail(5).mean(),
                    'driver_experience': len(past),
                }
            at_track = past[past['circuit'] == circuit]['finish']
            this_season = past[past['year'] == self.season]['finish']
            team_id = self._team_id(team_name)

            row.update({
                'grid': grid,
                'quali_gap_pct': self._gap_by_grid.get(grid, self._gap_by_grid.max()),
                'is_wet': int(is_wet),
                'round': round_no,
                'driver_form_5': driver_form,
                'driver_season_form': this_season.mean() if len(this_season) else driver_form,
                'driver_track_avg': at_track.mean() if len(at_track) else driver_form,
                'season_points_before': season_points.get(did, 0.0),
                'circuit_pos_change': self._circuit_change.get(circuit, self._circuit_change.median()),
                'reg_era_2026': int(self.season >= 2026),
                'team_id': team_id,
                **self._team_form(team_id),
            })
            rows.append(row)

        frame = pd.DataFrame(rows)
        frame['season_rank_before'] = frame['season_points_before'].rank(ascending=False, method='min')
        by_team = frame.groupby('team_id')
        frame['teammate_quali_delta'] = frame['quali_gap_pct'] - by_team['quali_gap_pct'].transform('mean')
        frame['grid_vs_teammate'] = frame['grid'] - by_team['grid'].transform('mean')
        frame['grid_vs_team_form'] = frame['grid'] - frame['team_form_5']
        return frame[self.features]

    def predict(self, X) -> Dict[str, np.ndarray]:
        """Return per-model predicted finishing positions plus the weighted ensemble."""
        offset = X['grid'].to_numpy() if self.target_mode == 'delta' else 0
        preds = {name: model.predict(X) + offset for name, model in self.models.items()}
        preds['ensemble'] = sum(preds[name] * w for name, w in self.weights.items())
        return preds


def load_trained_model() -> Optional[TrainedF1Model]:
    """Load the trained model bundle, or None if it is missing or unusable."""
    if not (os.path.exists(MODEL_PATH) and os.path.exists(HISTORY_PATH)):
        return None
    try:
        import joblib
        import pandas as pd
        model = TrainedF1Model(joblib.load(MODEL_PATH), pd.read_csv(HISTORY_PATH))
        logger.info(f"Loaded trained ML model ({model.metadata['trained_on']})")
        return model
    except Exception as e:
        logger.warning(f"Could not load trained ML model, using synthetic fallback: {e}")
        return None
