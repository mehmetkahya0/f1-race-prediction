"""
Module for simulating F1 race strategies, pit stops, and tire management.
"""

import random
from enum import Enum


class TireCompound(Enum):
    """F1 tire compounds."""
    SOFT = 1
    MEDIUM = 2
    HARD = 3
    INTERMEDIATE = 4
    WET = 5


class TireSet:
    """Represents a set of tires with degradation characteristics."""
    
    def __init__(self, compound, expected_life, track):
        self.compound = compound
        self.expected_life = expected_life  # Expected laps before significant degradation
        self.current_age = 0  # Current age in laps
        self.degradation = 0.0  # Current degradation level (0-1)
        self.track = track
        
        # Base pace factors for each compound (seconds per lap)
        self.base_pace_factors = {
            TireCompound.SOFT: 0.0,  # Baseline
            TireCompound.MEDIUM: 0.4,  # 0.4s slower per lap than soft
            TireCompound.HARD: 0.8,  # 0.8s slower per lap than soft
            TireCompound.INTERMEDIATE: 2.0,  # Only for mixed conditions
            TireCompound.WET: 3.0,  # Only for wet conditions
        }
        
        # Base degradation rates for each compound
        self.base_degradation_rates = {
            TireCompound.SOFT: 0.03,  # 3% degradation per lap
            TireCompound.MEDIUM: 0.022,  # 2.2% degradation per lap
            TireCompound.HARD: 0.015,  # 1.5% degradation per lap
            TireCompound.INTERMEDIATE: 0.025,  # Depends on conditions
            TireCompound.WET: 0.02,  # Depends on conditions
        }
    
    def update(self, track_conditions):
        """
        Update tire state after a lap.
        
        Args:
            track_conditions: Contains track status, weather, etc.
        """
        # Base degradation rate for the compound
        base_rate = self.base_degradation_rates[self.compound]
        
        # Adjust for track characteristics - high tyre wear tracks degrade tires faster
        track_factor = self.track.tyre_wear / 5  # Convert 1-10 scale to something more manageable
        
        # Adjust for weather
        if track_conditions['weather'].is_wet:
            if self.compound in [TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD]:
                # Slicks in wet conditions degrade extremely fast
                weather_factor = 5.0
            else:
                # Wet tires in wet conditions degrade normally
                weather_factor = 1.0
        else:
            if self.compound in [TireCompound.INTERMEDIATE, TireCompound.WET]:
                # Wet tires in dry conditions degrade extremely fast
                weather_factor = 6.0
            else:
                # Slicks in dry conditions degrade normally
                weather_factor = 1.0
        
        # Temperature effect
        temp = track_conditions['weather'].track_temperature
        temp_factor = 1.0
        if temp > 45:  # Very hot track
            temp_factor = 1.3
        elif temp < 15:  # Very cold track
            temp_factor = 0.8
            
        # Calculate degradation for this lap
        lap_degradation = base_rate * track_factor * weather_factor * temp_factor
        
        # Apply degradation
        self.degradation += lap_degradation
        self.current_age += 1
        
        return lap_degradation
    
    def get_pace_effect(self, track_conditions):
        """
        Calculate the effect of tire state on lap time.
        
        Args:
            track_conditions: Contains track and weather information
            
        Returns:
            Time penalty in seconds for current tire state
        """
        # Base effect from compound
        base_effect = self.base_pace_factors[self.compound]
        
        # Effect from degradation (as tires wear, they get slower)
        # Exponential increase in time as degradation increases
        # At 100% degradation, the tire is ~3s slower
        degradation_effect = 3.0 * (self.degradation ** 2)
        
        # Weather condition effect - wrong tire in wrong conditions has massive impact
        weather = track_conditions['weather']
        if weather.is_wet:
            if self.compound in [TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD]:
                # Slicks in rain are dangerously slow
                weather_effect = 10.0 + (weather.rain_intensity * 2)
            else:
                # Appropriate tires
                weather_effect = 0.0
        else:
            if self.compound in [TireCompound.INTERMEDIATE, TireCompound.WET]:
                # Rain tires in dry conditions are very slow
                weather_effect = 5.0
            else:
                # Appropriate tires
                weather_effect = 0.0
        
        return base_effect + degradation_effect + weather_effect


class PitStopStrategy:
    """Models a pit stop strategy for a race."""
    
    def __init__(self, track, weather, total_laps):
        """
        Initialize a pit stop strategy.
        
        Args:
            track: Track object
            weather: Weather conditions
            total_laps: Number of laps in the race
        """
        self.track = track
        self.weather = weather
        self.total_laps = total_laps
        self.pit_stops = []
        
        # Determine initial compound based on conditions
        if weather.is_wet:
            self.starting_compound = TireCompound.WET if weather.rain_intensity > 5 else TireCompound.INTERMEDIATE
        else:
            # In a real race, this would be a strategic choice
            # For simulation, we'll make a reasonable choice
            if total_laps < 40:  # Short race
                self.starting_compound = TireCompound.SOFT
            elif 40 <= total_laps <= 60:  # Medium race
                self.starting_compound = TireCompound.MEDIUM
            else:  # Long race
                self.starting_compound = TireCompound.HARD
    
    def generate_optimal_strategy(self):
        """
        Generate an optimal pit stop strategy based on track and conditions.
        
        Returns:
            List of (lap, compound) tuples indicating pit stops
        """
        # Expected tire life for each compound (laps)
        tire_life = {
            TireCompound.SOFT: max(15, 30 - self.track.tyre_wear),
            TireCompound.MEDIUM: max(25, 45 - self.track.tyre_wear),
            TireCompound.HARD: max(35, 60 - self.track.tyre_wear),
            TireCompound.INTERMEDIATE: 40 if self.weather.condition == 'mixed' else 20,
            TireCompound.WET: 60 if self.weather.condition == 'wet' else 10
        }
        
        # Strategy depends on track length, tire wear and weather
        pit_stops = []
        
        # Wet/intermediate conditions are special cases
        if self.weather.is_wet:
            if self.weather.condition == 'mixed':
                # Mixed conditions - might need to switch between wet and dry tires
                if self.weather.rain_intensity > 5:
                    # Start on wet, switch to inters if rain eases
                    pit_stops.append((int(self.total_laps * 0.4), TireCompound.INTERMEDIATE))
                else:
                    # Start on inters, might need to switch later
                    pit_stops.append((int(self.total_laps * 0.5), TireCompound.SOFT))
            else:  # fully wet
                # May need one stop for fresh wet tires
                if self.total_laps > tire_life[TireCompound.WET]:
                    pit_stops.append((tire_life[TireCompound.WET], TireCompound.WET))
        else:
            # Dry conditions - normal strategy
            # Number of stops depends on track tire wear and race length
            laps_covered = 0
            current_compound = self.starting_compound
            
            while laps_covered < self.total_laps:
                laps_covered += tire_life[current_compound]
                
                if laps_covered < self.total_laps:
                    # Need another stop
                    # Choose compound based on remaining laps
                    remaining_laps = self.total_laps - laps_covered
                    
                    if remaining_laps < tire_life[TireCompound.SOFT]:
                        next_compound = TireCompound.SOFT  # Fastest for the end
                    elif remaining_laps < tire_life[TireCompound.MEDIUM]:
                        next_compound = TireCompound.MEDIUM
                    else:
                        next_compound = TireCompound.HARD
                    
                    pit_stops.append((laps_covered, next_compound))
                    current_compound = next_compound
        
        return pit_stops
    
    def execute_pit_stop(self, lap):
        """
        Simulate execution of a pit stop.
        
        Args:
            lap: Current lap number
            
        Returns:
            Dictionary containing pit stop information including time lost
        """
        # Base pit stop time (seconds)
        base_time = 22.0  # Time from pit entry to pit exit including stationary time
        
        # Random variation in pit stop execution (±1 second)
        execution_variation = random.uniform(-1, 1)
        
        # Chance of pit stop error
        error_chance = 0.02  # 2% chance of an issue
        error_time = 0.0
        error_description = None
        
        if random.random() < error_chance:
            # Some form of pit stop error
            error_severity = random.random()
            
            if error_severity < 0.6:  # Minor issue
                error_time = random.uniform(1, 3)
                error_description = random.choice([
                    "Slightly slow front tire change",
                    "Brief delay connecting wheel gun",
                    "Slight hesitation releasing the car"
                ])
            elif error_severity < 0.9:  # Medium issue
                error_time = random.uniform(3, 8)
                error_description = random.choice([
                    "Problem with wheel gun",
                    "Tire not ready immediately",
                    "Traffic in pit lane causing delay"
                ])
            else:  # Major issue
                error_time = random.uniform(8, 20)
                error_description = random.choice([
                    "Cross-threaded wheel nut",
                    "Car dropped before tire fitted",
                    "Fuel hose issue"
                ])
        
        total_time_lost = base_time + execution_variation + error_time
        
        return {
            "lap": lap,
            "time_lost": total_time_lost,
            "error_time": error_time,
            "error_description": error_description
        }


def get_tire_strategy(track, weather, laps):
    """Helper function to generate a tire strategy for a given race."""
    strategy = PitStopStrategy(track, weather, laps)
    return strategy.generate_optimal_strategy()


def simulate_pit_stop(team_efficiency):
    """
    Simulate a pit stop execution with potential errors.
    
    Args:
        team_efficiency: Rating of team's pit crew efficiency (1-100)
    
    Returns:
        Dictionary with pit stop duration and any errors
    """
    # Base pit stop time (seconds)
    base_time = 2.0  # Stationary time for a perfect stop
    
    # Adjust for team efficiency
    team_factor = 1 - (team_efficiency / 100) * 0.5  # Scale effect
    adjusted_base = base_time * (1 + team_factor)
    
    # Random variation (±0.5 seconds)
    random_variation = random.uniform(-0.3, 0.7)
    
    # Error chance inversely proportional to team efficiency
    error_chance = 0.1 * (1 - team_efficiency / 100)
    
    if random.random() < error_chance:
        # Some kind of pit stop error occurred
        error_severity = random.random()
        
        if error_severity < 0.7:  # Minor issue
            error_time = random.uniform(1, 3)
            error = "Minor delay"
        elif error_severity < 0.95:  # Medium issue
            error_time = random.uniform(3, 8)
            error = "Significant delay"
        else:  # Major issue
            error_time = random.uniform(8, 20)
            error = "Major pit stop problem"
            
        stop_time = adjusted_base + random_variation + error_time
        
        return {
            "duration": stop_time,
            "error": error,
            "error_time": error_time
        }
    else:
        # Clean stop
        stop_time = adjusted_base + random_variation
        
        return {
            "duration": stop_time,
            "error": None,
            "error_time": 0
        }