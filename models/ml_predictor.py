"""
Machine Learning Prediction Module for F1 Race Simulation.

This module implements multiple ML algorithms to enhance race predictions:
- Random Forest for position prediction
- Gradient Boosting for lap time estimation
- Neural Network (MLP) for race outcome classification

When models/trained/ contains a model trained on real F1 results
(notebooks/train_f1_model_colab.ipynb), that model is used. Otherwise the
models are trained on synthetic data built from F1 domain knowledge.
Predictions are combined with Monte Carlo simulation for improved accuracy.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from models.trained_model import load_trained_model

logger = logging.getLogger(__name__)


@dataclass
class MLPrediction:
    """Stores ML prediction results with confidence metrics."""
    predicted_position: int
    confidence: float  # 0-1 scale
    position_probabilities: Dict[int, float]  # Position -> Probability
    feature_importance: Dict[str, float]
    model_agreement: float  # How much models agree


class FeatureExtractor:
    """Extracts ML features from driver, team, and track data."""
    
    FEATURE_NAMES = [
        'driver_skill_dry', 'driver_skill_wet', 'driver_consistency',
        'driver_experience', 'driver_qualifying_pace', 'driver_tire_management',
        'driver_racecraft', 'driver_overtaking', 'driver_aggression',
        'team_performance', 'team_reliability', 'team_aero', 'team_power',
        'team_active_aero', 'team_energy_recovery', 'team_pit_efficiency',
        'track_length', 'track_corners', 'track_downforce', 'track_tyre_wear',
        'track_overtaking_difficulty', 'track_active_aero_advantage',
        'track_energy_demand', 'grid_position', 'is_wet'
    ]
    
    def __init__(self):
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self._fitted = False
    
    def extract_features(self, driver, team, track, grid_position: int, 
                        is_wet: bool = False) -> np.ndarray:
        """Extract feature vector from race entities."""
        features = [
            driver.skill_dry,
            driver.skill_wet,
            driver.consistency if hasattr(driver, 'consistency') else 80,
            driver.experience if hasattr(driver, 'experience') else 5,
            driver.qualifying_pace if hasattr(driver, 'qualifying_pace') else 80,
            driver.tire_management if hasattr(driver, 'tire_management') else 80,
            driver.racecraft if hasattr(driver, 'racecraft') else 80,
            driver.skill_overtaking,
            driver.aggression if hasattr(driver, 'aggression') else 70,
            team.performance,
            team.reliability,
            team.aerodynamics,
            team.power,
            team.active_aero if hasattr(team, 'active_aero') else 80,
            team.energy_recovery if hasattr(team, 'energy_recovery') else 80,
            team.pit_efficiency,
            track.length_km,
            track.corners,
            track.downforce_level,
            track.tyre_wear,
            track.overtaking_difficulty,
            track.active_aero_advantage if hasattr(track, 'active_aero_advantage') else 5,
            track.energy_demand if hasattr(track, 'energy_demand') else 5,
            grid_position,
            1 if is_wet else 0
        ]
        return np.array(features).reshape(1, -1)
    
    def extract_all_features(self, drivers: List, teams: Dict, track, 
                            grid_positions: List, is_wet: bool = False) -> np.ndarray:
        """Extract features for all drivers."""
        all_features = []
        for i, driver in enumerate(grid_positions):
            team = teams.get(driver)
            if team:
                features = self.extract_features(
                    driver, team, track, i + 1, is_wet
                )
                all_features.append(features.flatten())
        return np.array(all_features)


class F1MLPredictor:
    """
    Ensemble ML predictor for F1 race outcomes.
    
    Uses multiple models:
    1. Random Forest: Captures non-linear relationships
    2. Gradient Boosting: Handles complex feature interactions
    3. Neural Network: Learns deep patterns
    
    Final prediction is weighted ensemble of all models.
    """
    
    # Ensemble weights (tuned for F1 prediction)
    RF_WEIGHT = 0.40  # Random Forest
    GB_WEIGHT = 0.35  # Gradient Boosting
    NN_WEIGHT = 0.25  # Neural Network
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self._models_trained = False
        # Model trained on real race results; None → synthetic fallback below
        self.trained_model = load_trained_model() if SKLEARN_AVAILABLE else None

        if SKLEARN_AVAILABLE:
            # Initialize models with optimized hyperparameters
            self.rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            self.gb_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                min_samples_split=5,
                random_state=42
            )
            
            self.nn_model = MLPRegressor(
                hidden_layer_sizes=(64, 32, 16),
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                max_iter=500,
                random_state=42,
                early_stopping=True
            )
            
            self.scaler = StandardScaler()
        else:
            self.rf_model = None
            self.gb_model = None
            self.nn_model = None
            self.scaler = None
        
        # Historical training data (synthetic based on F1 patterns)
        self._training_data = None
        self._training_labels = None
    
    def _generate_training_data(self, num_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training data based on F1 domain knowledge.
        
        This creates realistic driver-team-track combinations with
        corresponding race finishing positions based on known F1 patterns.
        """
        np.random.seed(42)
        
        X = []
        y = []
        
        for _ in range(num_samples):
            # Generate realistic feature values
            driver_skill_dry = np.random.normal(85, 8)
            driver_skill_wet = np.random.normal(82, 10)
            driver_consistency = np.random.normal(80, 10)
            driver_experience = np.random.randint(1, 20)
            driver_qualifying_pace = np.random.normal(83, 8)
            driver_tire_management = np.random.normal(80, 10)
            driver_racecraft = np.random.normal(82, 9)
            driver_overtaking = np.random.normal(80, 10)
            driver_aggression = np.random.normal(75, 12)
            
            team_performance = np.random.normal(85, 10)
            team_reliability = np.random.normal(88, 8)
            team_aero = np.random.normal(85, 10)
            team_power = np.random.normal(87, 8)
            team_active_aero = np.random.normal(83, 10)
            team_energy_recovery = np.random.normal(85, 9)
            team_pit_efficiency = np.random.normal(88, 7)
            
            track_length = np.random.uniform(3.3, 7.0)
            track_corners = np.random.randint(10, 27)
            track_downforce = np.random.randint(1, 10)
            track_tyre_wear = np.random.randint(1, 10)
            track_overtaking = np.random.randint(1, 10)
            track_aero_adv = np.random.randint(1, 10)
            track_energy = np.random.randint(1, 10)
            
            grid_position = np.random.randint(1, 23)
            is_wet = np.random.choice([0, 1], p=[0.85, 0.15])
            
            # Clip values to realistic ranges
            features = np.clip([
                driver_skill_dry, driver_skill_wet, driver_consistency,
                driver_experience, driver_qualifying_pace, driver_tire_management,
                driver_racecraft, driver_overtaking, driver_aggression,
                team_performance, team_reliability, team_aero, team_power,
                team_active_aero, team_energy_recovery, team_pit_efficiency,
                track_length, track_corners, track_downforce, track_tyre_wear,
                track_overtaking, track_aero_adv, track_energy,
                grid_position, is_wet
            ], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 5, 1, 1, 1, 1, 1, 1, 0],
               [100, 100, 100, 25, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 7, 30, 10, 10, 10, 10, 10, 22, 1])
            
            # Calculate expected finishing position based on F1 domain knowledge
            # Grid position is the strongest predictor
            base_position = grid_position
            
            # Performance factors
            performance_score = (
                driver_skill_dry * 0.20 +
                driver_consistency * 0.15 +
                driver_racecraft * 0.10 +
                team_performance * 0.25 +
                team_reliability * 0.10 +
                team_aero * 0.10 +
                team_power * 0.10
            ) / 100
            
            # Position delta based on performance
            position_delta = (1 - performance_score) * 8 - 4  # -4 to +4 positions
            
            # Wet weather impact
            if is_wet:
                wet_factor = (driver_skill_wet - 80) / 20 * 3
                position_delta -= wet_factor
            
            # Overtaking opportunity
            overtake_factor = (10 - track_overtaking) / 10
            if grid_position > 10:
                position_delta *= (1 + overtake_factor * 0.3)
            
            # Add some randomness (race incidents, luck, etc.)
            noise = np.random.normal(0, 1.5)
            
            final_position = base_position + position_delta + noise
            final_position = np.clip(final_position, 1, 22)
            
            X.append(features)
            y.append(final_position)
        
        return np.array(X), np.array(y)
    
    def train(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None):
        """
        Train all ML models on historical/synthetic data.
        
        If no data provided, generates synthetic training data.
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, ML predictions disabled")
            return
        
        if X is None or y is None:
            logger.info("Generating synthetic training data...")
            X, y = self._generate_training_data(5000)
        
        self._training_data = X
        self._training_labels = y
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest
        logger.info("Training Random Forest model...")
        self.rf_model.fit(X_scaled, y)
        
        # Train Gradient Boosting
        logger.info("Training Gradient Boosting model...")
        self.gb_model.fit(X_scaled, y)
        
        # Train Neural Network
        logger.info("Training Neural Network model...")
        self.nn_model.fit(X_scaled, y)
        
        self._models_trained = True
        logger.info("All ML models trained successfully")
    
    def predict(self, drivers: List, teams: Dict, track, 
                grid_positions: List, weather) -> List[MLPrediction]:
        """
        Generate ML predictions for all drivers.
        
        Returns list of MLPrediction objects with confidence scores.
        """
        if not SKLEARN_AVAILABLE:
            return self._fallback_predictions(drivers, grid_positions)

        is_wet = weather.is_wet if hasattr(weather, 'is_wet') else False

        if self.trained_model is not None:
            try:
                X = self.trained_model.build_features(
                    grid_positions, teams, track, is_wet, self._round_for(track)
                )
                preds = self.trained_model.predict(X)
                return self._build_predictions(
                    preds['ensemble'],
                    [preds[name] for name in self.trained_model.weights],
                    grid_positions,
                    self.trained_model.feature_importances
                )
            except Exception as e:
                logger.warning(f"Trained model prediction failed, using synthetic models: {e}")

        if not self._models_trained:
            self.train()

        # Extract features
        X = self.feature_extractor.extract_all_features(
            drivers, teams, track, grid_positions, is_wet
        )
        
        if len(X) == 0:
            return self._fallback_predictions(drivers, grid_positions)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from each model
        rf_preds = self.rf_model.predict(X_scaled)
        gb_preds = self.gb_model.predict(X_scaled)
        nn_preds = self.nn_model.predict(X_scaled)
        
        # Ensemble prediction (weighted average)
        ensemble_preds = (
            rf_preds * self.RF_WEIGHT +
            gb_preds * self.GB_WEIGHT +
            nn_preds * self.NN_WEIGHT
        )
        
        # Get feature importances from Random Forest
        feature_importance = dict(zip(
            FeatureExtractor.FEATURE_NAMES,
            self.rf_model.feature_importances_
        ))

        return self._build_predictions(
            ensemble_preds, [rf_preds, gb_preds, nn_preds],
            grid_positions, feature_importance
        )

    @staticmethod
    def _round_for(track) -> int:
        """Calendar round of the track (1-based), used as a season-progress feature."""
        from data.tracks import get_calendar
        names = [t.name for t in get_calendar()]
        return names.index(track.name) + 1 if track.name in names else len(names) // 2

    def _build_predictions(self, ensemble_preds, model_preds: List[np.ndarray],
                           grid_positions: List, feature_importance: Dict[str, float]) -> List[MLPrediction]:
        """Turn raw position estimates into MLPrediction objects with confidence."""
        # Calculate model agreement (standard deviation between models)
        model_std = np.std(model_preds, axis=0)
        model_agreement = 1 - np.clip(model_std / 5, 0, 1)  # Normalize

        # Create predictions
        predictions = []
        for i, driver in enumerate(grid_positions):
            pred_position = max(1, min(22, round(ensemble_preds[i])))
            
            # Calculate confidence based on model agreement and position certainty
            base_confidence = model_agreement[i]
            position_certainty = 1 - abs(ensemble_preds[i] - pred_position) / 10
            confidence = (base_confidence * 0.6 + position_certainty * 0.4)
            
            # Position probabilities (simplified Gaussian around prediction)
            position_probs = {}
            for pos in range(1, 23):
                prob = np.exp(-0.5 * ((pos - ensemble_preds[i]) / 2) ** 2)
                position_probs[pos] = prob
            
            # Normalize probabilities
            total_prob = sum(position_probs.values())
            position_probs = {k: v/total_prob for k, v in position_probs.items()}
            
            predictions.append(MLPrediction(
                predicted_position=pred_position,
                confidence=float(confidence),
                position_probabilities=position_probs,
                feature_importance=feature_importance,
                model_agreement=float(model_agreement[i])
            ))
        
        return predictions
    
    def _fallback_predictions(self, drivers: List, grid_positions: List) -> List[MLPrediction]:
        """Fallback predictions when sklearn is not available."""
        predictions = []
        for i, driver in enumerate(grid_positions):
            predictions.append(MLPrediction(
                predicted_position=i + 1,
                confidence=0.5,
                position_probabilities={j: 1/22 for j in range(1, 23)},
                feature_importance={},
                model_agreement=0.5
            ))
        return predictions
    
    def get_prediction_summary(self, predictions: List[MLPrediction], 
                               drivers: List) -> Dict:
        """Generate summary of ML predictions."""
        summary = {
            'total_drivers': len(predictions),
            'high_confidence_predictions': sum(1 for p in predictions if p.confidence > 0.7),
            'average_confidence': np.mean([p.confidence for p in predictions]),
            'model_agreement': np.mean([p.model_agreement for p in predictions]),
            'top_predictions': []
        }
        
        # Top 3 predictions with highest confidence
        sorted_preds = sorted(
            zip(predictions, drivers),
            key=lambda x: x[0].confidence,
            reverse=True
        )[:3]
        
        for pred, driver in sorted_preds:
            summary['top_predictions'].append({
                'driver': driver.name,
                'predicted_position': pred.predicted_position,
                'confidence': pred.confidence
            })
        
        return summary
    
    def evaluate_prediction_accuracy(self, predictions: List[MLPrediction],
                                    actual_results: List) -> Dict:
        """
        Evaluate prediction accuracy against actual results.
        
        Returns various accuracy metrics.
        """
        if len(predictions) != len(actual_results):
            return {'error': 'Mismatched lengths'}
        
        correct_positions = 0
        within_1 = 0
        within_3 = 0
        total_error = 0
        
        for pred, actual_pos in zip(predictions, actual_results):
            error = abs(pred.predicted_position - actual_pos)
            total_error += error
            
            if error == 0:
                correct_positions += 1
            if error <= 1:
                within_1 += 1
            if error <= 3:
                within_3 += 1
        
        n = len(predictions)
        return {
            'exact_accuracy': correct_positions / n,
            'within_1_accuracy': within_1 / n,
            'within_3_accuracy': within_3 / n,
            'mean_absolute_error': total_error / n,
            'total_predictions': n
        }


# Global predictor instance
_global_predictor = None

def get_ml_predictor() -> F1MLPredictor:
    """Get or create global ML predictor instance."""
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = F1MLPredictor()
    return _global_predictor
