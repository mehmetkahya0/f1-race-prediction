"""
Enhanced Race Model with Real F1 Data Integration and ML Predictions.

This enhanced model uses real F1 performance data and machine learning
to improve the accuracy of race predictions and provide more realistic results.

Updated for 2026 regulations with active aero and new power units.
"""

import random
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
import logging
from models.race_model import RaceSimulator, DriverRaceResult, RaceIncident, format_time, assign_points

# Import ML predictor
try:
    from models.ml_predictor import get_ml_predictor, F1MLPredictor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnhancedRaceSimulator(RaceSimulator):
    """Enhanced race simulator that uses real F1 data and ML for accurate predictions."""
    
    # Enhanced weighting for 2026 regulations
    # Share of the driver's real per-lap gap to the field median (last
    # season, this track) applied to the simulated pace
    REAL_DATA_WEIGHT = 0.50
    ML_WEIGHT = 0.25  # New: ML prediction weight
    # Per-lap time gap that corresponds to one finishing position: the
    # simulator's typical per-lap gap between consecutive finishers
    # (measured ~0.6 s/lap at Monza, Monaco, Silverstone), so ML_WEIGHT
    # is the share of that gap the ML can shift.
    ML_SECONDS_PER_POSITION = 0.6
    
    def __init__(self, track, drivers, teams, weather, real_data_enhancer=None, use_ml=True):
        super().__init__(track, drivers, teams, weather)
        self.real_data_enhancer = real_data_enhancer
        self.track_insights = {}
        self._cached_driver_data = None
        self._cached_team_data = None
        self.use_ml = use_ml and ML_AVAILABLE
        self.ml_predictor = None
        self.ml_predictions = None
        # Per-lap pace offsets (seconds) derived from ML predictions, keyed by driver
        self._ml_pace_offsets = {}
        # Per-lap pace offsets (seconds) derived from real lap times, keyed by driver
        self._real_pace_offsets = {}
        
        if self.use_ml:
            try:
                self.ml_predictor = get_ml_predictor()
                logger.info("ML predictor initialized")
            except Exception as e:
                logger.warning(f"Could not initialize ML predictor: {e}")
                self.use_ml = False
        
        if real_data_enhancer:
            self.track_insights = real_data_enhancer.get_track_insights(track.name, track.city)
            # Cache data for performance
            self._cached_driver_data = real_data_enhancer.get_enhanced_driver_data()
            self._cached_team_data = real_data_enhancer.get_enhanced_team_data()
            logger.info(f"Loaded track insights for {track.name}")
    
    def get_ml_predictions(self):
        """Get ML predictions for the race."""
        if not self.use_ml or not self.ml_predictor:
            return None
        
        if self.ml_predictions is None and self.grid_positions:
            try:
                self.ml_predictions = self.ml_predictor.predict(
                    self.drivers,
                    self.driver_teams,
                    self.track,
                    self.grid_positions,
                    self.weather
                )
                logger.info("ML predictions generated")
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}")
                self.ml_predictions = None
        
        return self.ml_predictions
    
    def _compute_ml_pace_offsets(self) -> Dict:
        """
        Compute per-lap pace offsets in seconds from ML predictions.
        
        For each driver the ML prediction gives a distribution over finishing
        positions. The expected position is converted to a per-lap time offset,
        centred on the field mean so the offsets sum to ~0 and nobody is
        artificially slowed down. A positive offset means slower, i.e. the ML
        expects the driver to finish further back than the field average.
        
        Returns:
            Dict mapping driver -> offset in seconds per lap
        """
        if not self.use_ml:
            return {}
        
        predictions = self.get_ml_predictions()
        if not predictions or not self.grid_positions:
            return {}
        
        # Expected finishing position per driver from the ML probability
        # distribution (predictions align with grid positions by index)
        expected_positions = {}
        for pred, driver in zip(predictions, self.grid_positions):
            expected = sum(pos * p for pos, p in pred.position_probabilities.items())
            expected_positions[driver] = expected
        
        # Centre on the field mean so the offsets sum to ~0
        mean_expected = sum(expected_positions.values()) / len(expected_positions)
        
        offsets = {}
        for driver in self.grid_positions:
            if driver not in expected_positions:
                continue
            # Positive offset = slower (ML expects the driver further back)
            offsets[driver] = (self.ML_WEIGHT * 
                               self.ML_SECONDS_PER_POSITION * 
                               (expected_positions[driver] - mean_expected))
        
        if offsets:
            min_offset = min(offsets.values())
            max_offset = max(offsets.values())
            logger.info(f"ML pace offsets applied: min={min_offset:+.3f}s/lap, max={max_offset:+.3f}s/lap")
        
        return offsets
    
    def get_ml_summary(self):
        """Get summary of ML predictions."""
        if not self.ml_predictions:
            return None
        
        return self.ml_predictor.get_prediction_summary(
            self.ml_predictions,
            self.grid_positions
        )
    
    def _compute_real_pace_offsets(self) -> Dict:
        """
        Compute per-lap pace offsets in seconds from real lap times.
        
        Uses the real mean lap times from last season at this track
        (track_insights['lap_times']). For each driver with data the offset is
        the driver's real per-lap gap to the field median, scaled by
        REAL_DATA_WEIGHT, so real data shifts pace relative to the field
        without changing the absolute lap time scale. A positive offset means
        slower, i.e. the driver was slower than the field median last season.
        Offsets are clipped to +/- 2.0 s to contain outliers from incidents.
        
        Returns:
            Dict mapping driver -> offset in seconds per lap
        """
        if not self.real_data_enhancer:
            return {}
        
        lap_times = self.track_insights.get('lap_times', {})
        if len(lap_times) < 5:
            return {}
        
        median_lap_time = float(np.median(list(lap_times.values())))
        
        offsets = {}
        for driver in self.grid_positions:
            abbr = self.real_data_enhancer._get_driver_abbreviation(driver.name)
            if abbr in lap_times:
                offset = self.REAL_DATA_WEIGHT * (lap_times[abbr] - median_lap_time)
                # Clip outliers (real lap times may include incident laps)
                offsets[driver] = max(-2.0, min(2.0, offset))
        
        if offsets:
            min_offset = min(offsets.values())
            max_offset = max(offsets.values())
            logger.info(f"Real pace offsets applied: {len(offsets)} drivers, min={min_offset:+.3f}s/lap, max={max_offset:+.3f}s/lap")
        
        return offsets
    
    def _calculate_race_pace(self, driver, team, lap_number=1, tire_age=0):
        """Calculate race pace with ML-derived and real-data per-lap pace offsets applied."""
        base_time = super()._calculate_race_pace(driver, team, lap_number, tire_age)
        return (base_time +
                self._ml_pace_offsets.get(driver, 0.0) +
                self._real_pace_offsets.get(driver, 0.0))
    
    def simulate_qualifying(self):
        """Enhanced qualifying simulation with real data consideration for 2026."""
        print("🏎️ Running enhanced qualifying simulation...")
        
        # Qualifying can be re-run, so predictions from a previous grid are stale
        self.ml_predictions = None
        
        # Get base qualifying results (list of drivers)
        grid_drivers = super().simulate_qualifying()
        
        # Apply real data insights for qualifying
        if self.real_data_enhancer and self._cached_driver_data:
            real_driver_data = self._cached_driver_data
            
            # Create qualifying results with times for adjustment
            qualifying_performances = []
            
            for position, driver in enumerate(grid_drivers):
                team = self.driver_teams[driver]
                
                # Get qualifying-specific performance
                if hasattr(driver, 'get_qualifying_rating'):
                    driver_rating = driver.get_qualifying_rating()
                else:
                    driver_rating = driver.get_overall_rating()
                
                if hasattr(team, 'get_qualifying_pace'):
                    team_rating = team.get_qualifying_pace()
                else:
                    team_rating = team.get_car_rating()
                
                # Calculate qualifying time
                base_time = self.BASE_LAP_TIME + (self.track.length_km - 5) * 4.5
                driver_factor = 4.0 * (1 - (driver_rating / 100))
                team_factor = 3.0 * (1 - (team_rating / 100))
                
                # Weather and track conditions
                weather_factor = 1.0
                if self.weather.condition == "wet":
                    wet_skill = driver.skill_wet / 100.0 if hasattr(driver, 'skill_wet') else 0.8
                    weather_factor *= 1.0 + (0.12 * (1 - wet_skill))
                elif self.weather.condition == "mixed":
                    weather_factor *= random.uniform(1.0, 1.06)
                
                lap_time = (base_time + driver_factor + team_factor) * weather_factor
                
                # Add qualifying session improvement
                q1_time = lap_time * random.uniform(1.002, 1.008)
                q2_time = lap_time * random.uniform(0.997, 1.003)
                q3_time = lap_time * random.uniform(0.994, 1.001)
                best_time = min(q1_time, q2_time, q3_time)
                
                # Apply real data adjustments
                driver_abbrev = self.real_data_enhancer._get_driver_abbreviation(driver.name)
                
                if driver_abbrev in real_driver_data:
                    real_data = real_driver_data[driver_abbrev]
                    
                    # Apply performance adjustment based on real consistency
                    consistency_factor = real_data.get('consistency', 80) / 100
                    qualifying_pace = real_data.get('qualifying_pace', 80) / 100 if 'qualifying_pace' in real_data else consistency_factor
                    
                    # Better qualifiers perform more consistently
                    if qualifying_pace > 0.88:
                        best_time *= random.uniform(0.996, 1.001)
                    elif qualifying_pace < 0.75:
                        best_time *= random.uniform(0.998, 1.008)
                    
                    logger.debug(f"Qualifying adjustment for {driver.name}: pace={qualifying_pace:.2f}")
                
                qualifying_performances.append((driver, best_time))
            
            # Re-sort by adjusted times
            qualifying_performances.sort(key=lambda x: x[1])
            
            # Update grid positions with adjusted order
            self.grid_positions = [driver for driver, _ in qualifying_performances]
            
            return self.grid_positions
        
        return grid_drivers
    
    def simulate_race(self):
        """Enhanced race simulation with real data insights for 2026."""
        print("🏁 Running enhanced race simulation...")
        
        # Ensure the grid exists before applying ML pace offsets,
        # so the offsets always match the grid that the race runs on
        if not self.grid_positions:
            self.simulate_qualifying()
        self._ml_pace_offsets = self._compute_ml_pace_offsets()
        self._real_pace_offsets = self._compute_real_pace_offsets()
        
        # Get base race results
        results = super().simulate_race()
        
        # Apply real data post-processing
        if self.real_data_enhancer:
            results = self._apply_real_data_race_adjustments(results)
        
        return results
    
    def _apply_real_data_race_adjustments(self, results: List[DriverRaceResult]) -> List[DriverRaceResult]:
        """Apply race adjustments based on real F1 data with optimized logic."""
        real_driver_data = self._cached_driver_data or self.real_data_enhancer.get_enhanced_driver_data()
        real_team_data = self._cached_team_data or self.real_data_enhancer.get_enhanced_team_data()
        
        logger.info("Applying real data race adjustments...")
        
        # Build team name lookup cache for efficiency
        team_lookup = {}
        for real_team_name in real_team_data.keys():
            team_lookup[real_team_name.lower()] = real_team_name
        
        for result in results:
            driver_abbrev = self.real_data_enhancer._get_driver_abbreviation(result.driver.name)
            team_name = self.real_data_enhancer._clean_team_name(result.team.name)
            
            # Driver-based adjustments
            if driver_abbrev in real_driver_data:
                real_data = real_driver_data[driver_abbrev]
                
                # Adjust based on real overtaking ability and racecraft
                overtaking_skill = real_data.get('skill_overtaking', 80)
                racecraft = real_data.get('racecraft', 80) if 'racecraft' in real_data else overtaking_skill
                
                if overtaking_skill > 88 and result.starting_position > result.finishing_position:
                    # Strong overtakers might gain additional positions
                    if random.random() < 0.20:
                        result.finishing_position = max(1, result.finishing_position - 1)
                
                # Tire management affects late race performance
                tire_management = real_data.get('tire_management', 80) if 'tire_management' in real_data else 80
                if tire_management > 88 and result.status == 'Finished':
                    # Good tire managers might gain a position in final laps
                    if random.random() < 0.12:
                        result.finishing_position = max(1, result.finishing_position - 1)
                
                # Adjust based on real consistency (reduced incident probability)
                consistency = real_data.get('consistency', 80)
                if consistency > 88:
                    # High consistency drivers have reduced incident chance
                    if result.incident != RaceIncident.NONE and random.random() < 0.15:
                        result.incident = RaceIncident.NONE
                        result.incident_description = ""
                        result.incident_lap = None
                        result.status = 'Finished'
                elif consistency < 72 and random.random() < 0.08:
                    # Less consistent drivers more likely to have issues
                    if result.incident == RaceIncident.NONE:
                        result.incident = random.choice([
                            RaceIncident.DRIVER_ERROR,
                            RaceIncident.PUNCTURE
                        ])
                        result.incident_description = "Consistency issues led to incident"
                        result.incident_lap = random.randint(1, self.track.laps)
            
            # Team-based adjustments  
            matching_team = None
            team_name_lower = team_name.lower()
            for key in team_lookup:
                if team_name_lower in key or key in team_name_lower:
                    matching_team = team_lookup[key]
                    break
            
            if matching_team and matching_team in real_team_data:
                team_data = real_team_data[matching_team]
                
                # Reliability adjustments (2026: includes energy system)
                reliability = team_data.get('reliability', 85)
                energy_recovery = team_data.get('energy_recovery', 85) if 'energy_recovery' in team_data else 85
                
                # Combined reliability for 2026
                combined_reliability = (reliability * 0.6 + energy_recovery * 0.4)
                
                if combined_reliability < 78 and random.random() < 0.06:
                    if result.incident == RaceIncident.NONE and random.random() < 0.25:
                        result.incident = random.choice([
                            RaceIncident.MECHANICAL_FAILURE,
                            RaceIncident.ENERGY_SYSTEM_FAILURE
                        ])
                        result.incident_description = "Reliability concerns led to technical issue"
                        result.incident_lap = random.randint(1, self.track.laps)
                        result.status = "DNF"
                        result.finishing_position = len(results)
        
        # Re-sort results to account for adjustments
        finished_results = [r for r in results if r.status == 'Finished']
        dnf_results = [r for r in results if r.status != 'Finished']
        
        # Re-assign positions for finished drivers
        finished_results.sort(key=lambda x: (x.finishing_position, x.time))
        for i, result in enumerate(finished_results):
            result.finishing_position = i + 1
            result.positions_gained = result.starting_position - result.finishing_position
        
        # DNF drivers get positions after all finished drivers
        for i, result in enumerate(dnf_results):
            result.finishing_position = len(finished_results) + i + 1
            result.positions_gained = result.starting_position - result.finishing_position
        
        all_results = finished_results + dnf_results
        
        # Positions and statuses changed above, so points must be recomputed
        assign_points(all_results)
        
        logger.info(f"Applied real data adjustments to {len(all_results)} drivers")
        return all_results
    
    def get_simulation_metadata(self) -> Dict:
        """Get metadata about the simulation including real data usage."""
        metadata = {
            'enhanced_with_real_data': self.real_data_enhancer is not None,
            'track_insights_available': bool(self.track_insights),
            'simulation_type': 'Enhanced with Real F1 Data (2026 Regulations)',
            'real_data_weight': self.REAL_DATA_WEIGHT,
            'ml_weight': self.ML_WEIGHT,
            'ml_pace_applied': bool(self._ml_pace_offsets),
            'real_pace_applied': bool(self._real_pace_offsets)
        }
        
        if self.real_data_enhancer:
            real_driver_data = self._cached_driver_data or self.real_data_enhancer.get_enhanced_driver_data()
            real_team_data = self._cached_team_data or self.real_data_enhancer.get_enhanced_team_data()
            
            metadata.update({
                'drivers_with_real_data': len(real_driver_data),
                'teams_with_real_data': len(real_team_data),
                'track_lap_times_available': len(self.track_insights.get('lap_times', {}))
            })
        
        return metadata


def create_enhanced_simulator(track, drivers, teams, weather, real_data_enhancer=None):
    """Factory function to create an enhanced race simulator."""
    if real_data_enhancer:
        return EnhancedRaceSimulator(track, drivers, teams, weather, real_data_enhancer)
    else:
        return RaceSimulator(track, drivers, teams, weather)
