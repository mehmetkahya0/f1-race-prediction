"""
Real F1 Data Integration using Fast-F1 library.

This module fetches real Formula 1 data (event schedules from Fast-F1, race
results and lap times from the Jolpica/Ergast API through Fast-F1) and
integrates it with the simulation to provide more realistic race predictions.
"""

import fastf1
from fastf1.ergast import Ergast
from fastf1.exceptions import ErgastInvalidRequestError
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import warnings
import time
import sys
import os

# Add utils to path for loading screen
utils_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils')
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

try:
    from loading_screen import suppress_fastf1_logging, loading_with_animation, show_data_loading_screen, setup_global_logging_suppression
    # Apply global suppression immediately
    setup_global_logging_suppression()
except ImportError:
    # Fallback if loading screen is not available
    from contextlib import contextmanager
    
    @contextmanager
    def suppress_fastf1_logging():
        yield
    
    def loading_with_animation(message, func, *args, **kwargs):
        print(f"🔄 {message}...")
        return func(*args, **kwargs)
    
    def show_data_loading_screen(stage="Fetching data"):
        print(f"🏎️ {stage}...")
    
    def setup_global_logging_suppression():
        pass

# Configure logging
logger = logging.getLogger(__name__)

# Enable Fast-F1 caching for better performance
cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Suppress Fast-F1 logging during cache setup
with suppress_fastf1_logging():
    fastf1.Cache.enable_cache(cache_dir)

# Results and lap times come from the Jolpica (Ergast) API, which needs no
# F1 account sign-in, unlike FastF1's own timing data (Session.load)
ergast = Ergast(result_type='pandas', auto_cast=True)
ERGAST_PAGE_SIZE = 100  # maximum page size the API allows

ERGAST_RETRIES = 3  # the API rejects bursts of requests, so retry after a pause


def ergast_call(method, **kwargs):
    """Call an Ergast API method, retrying when the server rejects a burst."""
    for attempt in range(ERGAST_RETRIES):
        try:
            return method(**kwargs)
        except ErgastInvalidRequestError:
            if attempt == ERGAST_RETRIES - 1:
                raise
            time.sleep(5 * (attempt + 1))


# Laps slower than this multiple of the fastest lap are not representative
QUICKLAP_THRESHOLD = 1.07

# Ergast constructorId -> 2026 team name, where the team was renamed
CONSTRUCTOR_ALIASES = {
    'rb': 'Racing Bulls',
    'sauber': 'Audi',
}

# Track city -> FastF1 event 'Location' where the two differ
FASTF1_LOCATION_ALIASES = {
    'miami': 'miami gardens',
    'montreal': 'montréal',
    'monte carlo': 'monaco',
    'spa': 'spa-francorchamps',
    'singapore': 'marina bay',
    'abu dhabi': 'yas island',
}


class RealDataProvider:
    """Provides real F1 data from Fast-F1 API."""
    
    def __init__(self):
        self.current_year = datetime.now().year
        # Use the most recent completed season
        self.last_season_year = self.current_year - 1
        
    def get_current_season_schedule(self) -> pd.DataFrame:
        """Get the current F1 season schedule."""
        def _fetch_schedule():
            try:
                # Try the last completed season first for more reliable data
                schedule = fastf1.get_event_schedule(self.last_season_year)
                logger.info(f"Using {self.last_season_year} season data")
                return schedule
            except Exception as e:
                logger.warning(f"Could not fetch {self.last_season_year} schedule: {e}")
                # Fallback to current year
                try:
                    schedule = fastf1.get_event_schedule(self.current_year)
                    logger.info(f"Using {self.current_year} season data as fallback")
                    return schedule
                except Exception as e2:
                    logger.error(f"Could not fetch any schedule: {e2}")
                    return pd.DataFrame()
        
        return loading_with_animation("Fetching F1 season schedule", _fetch_schedule)
    
    def _load_race(self, year: int, round_number: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load race results and lap times from the Jolpica (Ergast) API.

        FastF1's own timing data needs an F1 account sign-in, Jolpica does not.

        Returns:
            (results, laps). results has one row per driver (driverCode,
            constructorId, constructorName, grid, position, points, status).
            laps has one row per driver per lap (driverCode, lap, position,
            seconds).
        """
        results = ergast_call(ergast.get_race_results, season=year, round=round_number).content[0]

        # The API returns at most 100 laps per request, so page through them
        pages = []
        offset = 0
        while True:
            response = ergast_call(ergast.get_lap_times, season=year, round=round_number,
                                   limit=ERGAST_PAGE_SIZE, offset=offset)
            if response.content:
                pages.append(response.content[0])
            offset += ERGAST_PAGE_SIZE
            if offset >= response.total_results:
                break

        laps = pd.concat(pages, ignore_index=True) if pages else pd.DataFrame(
            columns=['number', 'driverId', 'position', 'time'])
        codes = results.set_index('driverId')['driverCode']
        laps = pd.DataFrame({
            'driverCode': laps['driverId'].map(codes),
            'lap': laps['number'],
            'position': laps['position'],
            'seconds': pd.to_timedelta(laps['time']).dt.total_seconds(),
        }).dropna(subset=['driverCode', 'seconds'])
        return results, laps

    @staticmethod
    def _quick_laps(laps: pd.DataFrame) -> pd.DataFrame:
        """Keep laps within 107% of the race's fastest lap.

        Drops pit in/out laps, safety car laps and the standing start lap,
        like FastF1's pick_quicklaps.
        """
        if laps.empty:
            return laps
        return laps[laps['seconds'] <= laps['seconds'].min() * QUICKLAP_THRESHOLD]

    def _recent_races(self, year: int, count: int) -> pd.DataFrame:
        """Last `count` race events of a season (pre-season testing excluded)."""
        schedule = fastf1.get_event_schedule(year)
        return schedule[schedule['EventFormat'] != 'testing'].tail(count)

    def get_driver_performance_data(self, year: Optional[int] = None) -> Dict[str, Dict]:
        """
        Extract real driver performance data from recent races.

        Returns:
            Dictionary with driver performance metrics
        """
        def _fetch_driver_data():
            driver_stats = {}

            try:
                # Try to get data from multiple recent years for better statistics
                years_to_try = [self.last_season_year, self.last_season_year - 1]

                for year in years_to_try:
                    try:
                        # Get last few races for statistics
                        recent_races = self._recent_races(year, 5)

                        for _, event in recent_races.iterrows():
                            try:
                                results, laps = self._load_race(year, int(event['RoundNumber']))

                                if laps.empty:
                                    continue

                                quick_laps = self._quick_laps(laps)

                                # Extract driver performance metrics
                                for _, driver_result in results.iterrows():
                                    driver_name = driver_result['driverCode']
                                    driver_laps = laps[laps['driverCode'] == driver_name]

                                    if driver_laps.empty:
                                        continue

                                    if driver_name not in driver_stats:
                                        driver_stats[driver_name] = {
                                            'lap_times': [],
                                            'positions': [],
                                            'consistency_scores': [],
                                            'fastest_laps': [],
                                            'overtakes': 0,
                                            'grid_positions': [],
                                            'race_positions': [],
                                            'total_races': 0
                                        }

                                    # Collect lap times (exclude outliers)
                                    lap_times = quick_laps.loc[quick_laps['driverCode'] == driver_name, 'seconds']
                                    if not lap_times.empty:
                                        driver_stats[driver_name]['lap_times'].extend(lap_times.tolist())

                                        # Calculate consistency (lower std deviation = more consistent)
                                        if len(lap_times) > 1:
                                            consistency = 100 - (np.std(lap_times) * 10)  # Scale to 0-100
                                            driver_stats[driver_name]['consistency_scores'].append(max(0, min(100, consistency)))

                                    # Track positions throughout race
                                    positions = driver_laps['position'].dropna()
                                    if not positions.empty:
                                        driver_stats[driver_name]['positions'].extend(positions.tolist())

                                    # Get grid and race positions (grid 0 = pit lane start)
                                    if pd.notna(driver_result['grid']) and driver_result['grid'] > 0:
                                        driver_stats[driver_name]['grid_positions'].append(float(driver_result['grid']))
                                    if pd.notna(driver_result['position']):
                                        driver_stats[driver_name]['race_positions'].append(float(driver_result['position']))

                                    driver_stats[driver_name]['total_races'] += 1

                            except Exception as e:
                                logger.warning(f"Could not process event {event['EventName']} in {year}: {e}")
                                continue

                    except Exception as e:
                        logger.warning(f"Could not get schedule for {year}: {e}")
                        continue

            except Exception as e:
                logger.error(f"Error extracting driver performance data: {e}")

            return driver_stats

        return loading_with_animation("Analyzing driver performance data", _fetch_driver_data)

    def get_team_performance_data(self, year: Optional[int] = None) -> Dict[str, Dict]:
        """Extract real team performance data."""
        if year is None:
            year = self.last_season_year

        team_stats = {}

        try:
            recent_races = self._recent_races(year, 3)

            for _, event in recent_races.iterrows():
                try:
                    results = ergast_call(ergast.get_race_results, season=year, round=int(event['RoundNumber'])).content[0]

                    for _, driver_result in results.iterrows():
                        team_name = CONSTRUCTOR_ALIASES.get(driver_result['constructorId'],
                                                            driver_result['constructorName'])

                        if team_name not in team_stats:
                            team_stats[team_name] = {
                                'points': [],
                                'grid_positions': [],
                                'race_positions': [],
                                'reliability_issues': 0,
                                'total_races': 0,
                                'pit_stop_times': []
                            }

                        # Track points and positions
                        if pd.notna(driver_result['points']):
                            team_stats[team_name]['points'].append(float(driver_result['points']))

                        if pd.notna(driver_result['grid']) and driver_result['grid'] > 0:
                            team_stats[team_name]['grid_positions'].append(float(driver_result['grid']))

                        if pd.notna(driver_result['position']):
                            team_stats[team_name]['race_positions'].append(float(driver_result['position']))

                        # Check for reliability issues (DNF). Classified finishers
                        # have status "Finished", "Lapped" or "+N Lap(s)".
                        status = str(driver_result['status'])
                        if status not in ('Finished', 'Lapped') and not status.startswith('+'):
                            team_stats[team_name]['reliability_issues'] += 1

                        team_stats[team_name]['total_races'] += 1

                except Exception as e:
                    logger.warning(f"Could not process team data for event {event['EventName']}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error extracting team performance data: {e}")

        return team_stats

    def get_track_specific_data(self, track_name: str, year: Optional[int] = None,
                                city: Optional[str] = None) -> Dict:
        """Get track-specific performance data."""
        if year is None:
            year = self.last_season_year

        track_data = {
            'lap_times': {},
            'weather_history': [],
            'tire_strategies': {},
            'overtaking_zones': []
        }

        try:
            # Try to find the event for this track
            schedule = fastf1.get_event_schedule(year)

            # Match the circuit to an event. Event names are "<Country> Grand Prix",
            # not circuit names, so match the track's city against the event
            # Location, then fall back to a name substring match.
            location = FASTF1_LOCATION_ALIASES.get(city.lower(), city.lower()) if city else None
            races = schedule[schedule['EventFormat'] != 'testing']
            matching_event = None
            for _, event in races.iterrows():
                if location and str(event['Location']).lower() == location:
                    matching_event = event
                    break
            if matching_event is None:
                for _, event in races.iterrows():
                    if track_name.lower() in event['EventName'].lower() or event['EventName'].lower() in track_name.lower():
                        matching_event = event
                        break

            if matching_event is None:
                logger.warning(f"Could not find event for track: {track_name}")
                return track_data

            # Extract mean quick-lap time per driver
            _, laps = self._load_race(year, int(matching_event['RoundNumber']))
            quick_laps = self._quick_laps(laps)
            if not quick_laps.empty:
                track_data['lap_times'] = quick_laps.groupby('driverCode')['seconds'].mean().to_dict()

        except Exception as e:
            logger.warning(f"Could not extract track data for {track_name}: {e}")

        return track_data

    def calculate_realistic_driver_ratings(self) -> Dict[str, Dict]:
        """Calculate realistic driver ratings based on real performance data."""
        driver_perf_data = self.get_driver_performance_data()
        ratings = {}
        
        for driver, stats in driver_perf_data.items():
            if stats['total_races'] == 0:
                continue
                
            # Calculate average position performance
            avg_grid_pos = np.mean(stats['grid_positions']) if stats['grid_positions'] else 10
            avg_race_pos = np.mean(stats['race_positions']) if stats['race_positions'] else 10
            
            # Position improvement (negative means better performance in race)
            position_improvement = avg_grid_pos - avg_race_pos
            
            # Consistency score
            consistency = np.mean(stats['consistency_scores']) if stats['consistency_scores'] else 75
            
            # Calculate skill ratings (scale 0-100)
            # Better average position = higher skill
            skill_dry = max(20, min(100, 100 - (avg_race_pos - 1) * 4))  # Scale position to skill
            skill_wet = skill_dry + np.random.normal(0, 5)  # Add some variation for wet
            skill_wet = max(20, min(100, skill_wet))
            
            # Overtaking skill based on position improvement
            skill_overtaking = max(20, min(100, 70 + position_improvement * 3))
            
            # Experience estimation (simplified)
            experience = min(20, stats['total_races'] // 10)  # Rough estimate
            
            # Aggression based on overtaking ability
            aggression = max(20, min(100, skill_overtaking + np.random.normal(0, 10)))
            
            ratings[driver] = {
                'skill_dry': round(skill_dry),
                'skill_wet': round(skill_wet),
                'skill_overtaking': round(skill_overtaking),
                'consistency': round(consistency),
                'experience': round(experience),
                'aggression': round(aggression),
                'total_races': stats['total_races']
            }
        
        return ratings
    
    def calculate_realistic_team_ratings(self) -> Dict[str, Dict]:
        """Calculate realistic team ratings based on real performance data."""
        team_perf_data = self.get_team_performance_data()
        ratings = {}
        
        for team, stats in team_perf_data.items():
            if stats['total_races'] == 0:
                continue
                
            # Calculate average performance metrics
            avg_points = np.mean(stats['points']) if stats['points'] else 0
            avg_grid_pos = np.mean(stats['grid_positions']) if stats['grid_positions'] else 10
            avg_race_pos = np.mean(stats['race_positions']) if stats['race_positions'] else 10
            
            # Reliability (fewer DNFs = higher reliability)
            reliability_rate = 1 - (stats['reliability_issues'] / max(1, stats['total_races']))
            reliability = max(50, min(100, reliability_rate * 100))
            
            # Performance based on points and positions
            performance = max(20, min(100, 100 - (avg_race_pos - 1) * 4))
            
            # Aerodynamics and power (estimated from performance)
            aerodynamics = performance + np.random.normal(0, 5)
            aerodynamics = max(20, min(100, aerodynamics))
            
            power = performance + np.random.normal(0, 5)
            power = max(20, min(100, power))
            
            # Pit efficiency (simplified)
            pit_efficiency = max(60, min(100, 80 + np.random.normal(0, 10)))
            
            # Development rate (simplified)
            development_rate = max(60, min(100, 75 + np.random.normal(0, 10)))
            
            ratings[team] = {
                'performance': round(performance),
                'reliability': round(reliability),
                'aerodynamics': round(aerodynamics),
                'power': round(power),
                'pit_efficiency': round(pit_efficiency),
                'development_rate': round(development_rate),
                'total_races': stats['total_races']
            }
        
        return ratings


class RealDataEnhancer:
    """Enhances simulation with real F1 data."""
    
    def __init__(self):
        self.data_provider = RealDataProvider()
        self._driver_ratings = None
        self._team_ratings = None
        
    def get_enhanced_driver_data(self) -> Dict:
        """Get driver data enhanced with real performance metrics."""
        if self._driver_ratings is None:
            self._driver_ratings = self.data_provider.calculate_realistic_driver_ratings()
        return self._driver_ratings
        
    def get_enhanced_team_data(self) -> Dict:
        """Get team data enhanced with real performance metrics."""
        if self._team_ratings is None:
            self._team_ratings = self.data_provider.calculate_realistic_team_ratings()
        return self._team_ratings
    
    def enhance_race_prediction(self, drivers, teams, track) -> Tuple[List, List]:
        """
        Enhance race prediction with real data insights.
        
        Returns:
            Tuple of (enhanced_drivers, enhanced_teams)
        """
        real_driver_data = self.get_enhanced_driver_data()
        real_team_data = self.get_enhanced_team_data()
        
        enhanced_drivers = []
        enhanced_teams = []
        
        # Enhance drivers with real data
        for driver in drivers:
            enhanced_driver = driver
            
            # Find matching real data
            driver_abbrev = self._get_driver_abbreviation(driver.name)
            if driver_abbrev in real_driver_data:
                real_data = real_driver_data[driver_abbrev]
                
                # Update driver attributes with real data (weighted average)
                weight_real = 0.7  # 70% real data, 30% original fictional data
                weight_original = 0.3
                
                enhanced_driver.skill_dry = round(
                    real_data['skill_dry'] * weight_real + driver.skill_dry * weight_original
                )
                enhanced_driver.skill_wet = round(
                    real_data['skill_wet'] * weight_real + driver.skill_wet * weight_original
                )
                enhanced_driver.skill_overtaking = round(
                    real_data['skill_overtaking'] * weight_real + driver.skill_overtaking * weight_original
                )
                enhanced_driver.consistency = round(
                    real_data['consistency'] * weight_real + driver.consistency * weight_original
                )
                enhanced_driver.aggression = round(
                    real_data['aggression'] * weight_real + driver.aggression * weight_original
                )
                
                logger.info(f"Enhanced {driver.name} with real data from {real_data['total_races']} races")
            
            enhanced_drivers.append(enhanced_driver)
        
        # Enhance teams with real data
        for team in teams:
            enhanced_team = team
            
            # Find matching real data
            team_name_clean = self._clean_team_name(team.name)
            matching_team = None
            
            for real_team_name in real_team_data.keys():
                if team_name_clean.lower() in real_team_name.lower() or real_team_name.lower() in team_name_clean.lower():
                    matching_team = real_team_name
                    break
            
            if matching_team and matching_team in real_team_data:
                real_data = real_team_data[matching_team]
                
                # Update team attributes with real data
                weight_real = 0.7
                weight_original = 0.3
                
                enhanced_team.performance = round(
                    real_data['performance'] * weight_real + team.performance * weight_original
                )
                enhanced_team.reliability = round(
                    real_data['reliability'] * weight_real + team.reliability * weight_original
                )
                enhanced_team.aerodynamics = round(
                    real_data['aerodynamics'] * weight_real + team.aerodynamics * weight_original
                )
                enhanced_team.power = round(
                    real_data['power'] * weight_real + team.power * weight_original
                )
                enhanced_team.pit_efficiency = round(
                    real_data['pit_efficiency'] * weight_real + team.pit_efficiency * weight_original
                )
                
                logger.info(f"Enhanced {team.name} with real data from {real_data['total_races']} races")
            
            enhanced_teams.append(enhanced_team)
        
        return enhanced_drivers, enhanced_teams
    
    def get_track_insights(self, track_name: str, city: Optional[str] = None) -> Dict:
        """Get real insights for track-specific predictions."""
        return self.data_provider.get_track_specific_data(track_name, city=city)
    
    def _get_driver_abbreviation(self, full_name: str) -> str:
        """Convert full driver name to common abbreviation."""
        name_map = {
            'Max Verstappen': 'VER',
            'Lewis Hamilton': 'HAM',
            'Charles Leclerc': 'LEC',
            'George Russell': 'RUS',
            'Lando Norris': 'NOR',
            'Oscar Piastri': 'PIA',
            'Fernando Alonso': 'ALO',
            'Lance Stroll': 'STR',
            'Pierre Gasly': 'GAS',
            'Nico Hulkenberg': 'HUL',
            'Yuki Tsunoda': 'TSU',
            'Liam Lawson': 'LAW',
            'Kimi Antonelli': 'ANT',
            'Jack Doohan': 'DOO',
            'Gabriel Bortoleto': 'BOR',
            'Isack Hadjar': 'HAD'
        }
        # FIA codes are normally the first three letters of the surname
        return name_map.get(full_name, full_name.split()[-1][:3].upper())
    
    def _clean_team_name(self, team_name: str) -> str:
        """Clean team name for matching."""
        # Remove common suffixes and normalize
        team_name = team_name.replace(' Racing', '').replace(' F1 Team', '').replace(' Team', '')
        return team_name.strip()


# Global instance for easy access
real_data_enhancer = RealDataEnhancer()
