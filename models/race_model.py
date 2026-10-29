"""
Core race simulation model for Formula 1 race predictions.
This model simulates race outcomes based on driver skills, car performance, track characteristics,
weather conditions, and various random events that can happen during a race.

Updated for 2026 regulations with active aero and new power units.
"""

import random
import numpy as np
from dataclasses import dataclass
from typing import Optional
from enum import Enum

from models.strategy import get_tire_strategy, simulate_pit_stop


class RaceIncident(Enum):
    """Types of incidents that can occur during a race."""
    NONE = 0
    MECHANICAL_FAILURE = 1
    DRIVER_ERROR = 2
    COLLISION = 3
    PUNCTURE = 4
    WEATHER_RELATED = 5
    PENALTY = 6
    PIT_ERROR = 7
    ENERGY_SYSTEM_FAILURE = 8  # New for 2026


@dataclass
class DriverRaceResult:
    """Stores the final race result for a driver."""
    driver: object
    team: object
    starting_position: int
    finishing_position: int
    time: float  # Race time in seconds
    status: str  # 'Finished', 'DNF', 'DSQ'
    fastest_lap: bool = False
    incident: RaceIncident = RaceIncident.NONE
    incident_description: str = ""
    incident_lap: Optional[int] = None
    points: int = 0
    positions_gained: int = 0  # New metric
    pit_stops: int = 0
    
    def __str__(self):
        if self.status == 'Finished':
            return f"{self.finishing_position}. {self.driver.name} ({self.team.name}) - {format_time(self.time)}"
        else:
            return f"{self.finishing_position}. {self.driver.name} ({self.team.name}) - {self.status}"


def format_time(time_seconds):
    """Format race time in a readable format."""
    minutes = int(time_seconds // 60)
    seconds = int(time_seconds % 60)
    milliseconds = int((time_seconds - int(time_seconds)) * 1000)
    return f"{minutes}:{seconds:02d}.{milliseconds:03d}"


# Points for P1-P10 (no fastest lap bonus point since 2025)
POINTS_SYSTEM = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]


def assign_points(results):
    """Set points from finishing position. Must be re-run whenever positions or statuses change."""
    for result in results:
        result.points = 0
        if result.status == 'Finished' and result.finishing_position <= len(POINTS_SYSTEM):
            result.points = POINTS_SYSTEM[result.finishing_position - 1]


class RaceSimulator:
    """Simulates a Formula 1 race with realistic outcomes for 2026 regulations."""
    
    # Constants for more consistent simulation
    BASE_LAP_TIME = 90.0  # Base lap time in seconds
    PIT_LANE_LOSS = 20.0  # Seconds lost driving through the pit lane (excluding stationary time)
    PIT_LAP_JITTER = 3  # Each driver pits within +/- this many laps of the planned lap
    
    def __init__(self, track, drivers, teams, weather, qualifying_results=None):
        """
        Initialize the race simulator.
        
        Args:
            track: Track object containing circuit information
            drivers: List of Driver objects
            teams: List of Team objects
            weather: WeatherCondition object
            qualifying_results: Optional list of driver positions from qualifying
        """
        self.track = track
        self.drivers = drivers
        self.teams = teams
        self.weather = weather
        
        # Random seed for reproducibility within a simulation
        self._simulation_seed = random.randint(0, 1000000)
        
        # Match drivers to their teams
        self.driver_teams = {}
        for driver in drivers:
            for team in teams:
                if driver.team == team.name:
                    self.driver_teams[driver] = team
                    break
        
        # Use provided qualifying results or generate them
        if qualifying_results:
            self.grid_positions = qualifying_results
        else:
            self.grid_positions = []
            
        self.race_results = []
        self.race_simulated = False
    
    def _get_driver_performance_factor(self, driver, is_qualifying=False):
        """Calculate normalized driver performance factor (0-1 scale)."""
        if is_qualifying:
            # Use qualifying-specific rating if available
            if hasattr(driver, 'get_qualifying_rating'):
                return driver.get_qualifying_rating() / 100.0
            return driver.get_overall_rating() / 100.0
        else:
            # Use race-specific rating if available
            if hasattr(driver, 'get_race_rating'):
                return driver.get_race_rating() / 100.0
            return driver.get_overall_rating() / 100.0
    
    def _get_car_performance_factor(self, team, is_qualifying=False):
        """Calculate normalized car performance factor (0-1 scale)."""
        if is_qualifying:
            # Use qualifying-specific rating if available
            if hasattr(team, 'get_qualifying_pace'):
                return team.get_qualifying_pace() / 100.0
            return team.get_car_rating() / 100.0
        else:
            # Use race-specific rating if available
            if hasattr(team, 'get_race_pace'):
                return team.get_race_pace() / 100.0
            return team.get_car_rating() / 100.0
    
    def _get_track_modifier(self, team, driver):
        """Calculate track-specific performance modifier."""
        modifier = 1.0
        
        # Active aero advantage (2026 specific)
        if hasattr(self.track, 'active_aero_advantage') and hasattr(team, 'active_aero'):
            aero_advantage = (self.track.active_aero_advantage / 10.0) * (team.active_aero / 100.0)
            modifier -= aero_advantage * 0.02  # Up to 2% advantage
        
        # Energy demand (2026 specific)
        if hasattr(self.track, 'energy_demand') and hasattr(team, 'energy_recovery'):
            energy_advantage = (self.track.energy_demand / 10.0) * (team.energy_recovery / 100.0)
            modifier -= energy_advantage * 0.015  # Up to 1.5% advantage
        
        # Downforce level matching
        if self.track.downforce_level >= 8:  # High downforce track
            modifier -= (team.aerodynamics / 100.0) * 0.01
        elif self.track.downforce_level <= 3:  # Low downforce track
            modifier -= (team.power / 100.0) * 0.01
        
        return modifier
    
    def simulate_qualifying(self):
        """
        Simulate qualifying session to determine grid positions.
        Enhanced for 2026 with more consistent results.
        
        Returns:
            Ordered list of drivers based on qualifying performance
        """
        qualifying_performances = []
        
        # Use consistent random state for this session
        random.seed(self._simulation_seed)
        
        for driver in self.drivers:
            team = self.driver_teams[driver]
            
            # Base qualifying time calculation (track-adjusted)
            base_time = self.BASE_LAP_TIME + (self.track.length_km - 5) * 4.5
            
            # Driver skill impact (normalized, 0-4 seconds delta)
            driver_factor = self._get_driver_performance_factor(driver, is_qualifying=True)
            driver_time_delta = 4.0 * (1 - driver_factor)
            
            # Car performance impact (normalized, 0-3 seconds delta)
            car_factor = self._get_car_performance_factor(team, is_qualifying=True)
            car_time_delta = 3.0 * (1 - car_factor)
            
            # Track modifier (can be positive or negative)
            track_modifier = self._get_track_modifier(team, driver)
            
            # Weather impact (more significant in 2026 with active aero)
            if self.weather.is_wet:
                # Wet skill and active aero both matter
                wet_skill_factor = driver.skill_wet / 100.0
                weather_impact = 3.0 * (1 - wet_skill_factor)
                if hasattr(team, 'active_aero'):
                    # Active aero less effective in wet
                    weather_impact += 0.5 * (team.active_aero / 100.0)
            else:
                weather_impact = 0.3 * (1 - (driver.skill_dry / 100.0))
            
            # Controlled random factor (smaller variance for consistency)
            # Based on driver consistency
            consistency_factor = driver.consistency / 100.0 if hasattr(driver, 'consistency') else 0.8
            random_variance = 0.15 * (1 - consistency_factor * 0.5)
            random_factor = random.gauss(0, random_variance)
            
            # Calculate base lap time
            lap_time = (base_time + driver_time_delta + car_time_delta + 
                       weather_impact + random_factor) * track_modifier
            
            # Simulate Q1, Q2, Q3 with improvement factor
            improvement_factor = 0.995 + (driver_factor * 0.003)  # Better drivers improve more
            q1_time = lap_time * random.uniform(1.002, 1.008)
            q2_time = lap_time * random.uniform(0.998, 1.003) * improvement_factor
            q3_time = lap_time * random.uniform(0.995, 1.001) * (improvement_factor ** 2)
            
            best_time = min(q1_time, q2_time, q3_time)
            qualifying_performances.append((driver, best_time))
        
        # Sort by lap time
        qualifying_performances.sort(key=lambda x: x[1])
        
        # Store grid positions
        self.grid_positions = [driver for driver, _ in qualifying_performances]
        
        # Reset random state
        random.seed()
        
        return self.grid_positions
    
    def _calculate_race_pace(self, driver, team, lap_number=1, tire_age=0):
        """Calculate the race pace for a driver-team combination with 2026 updates."""
        # Base lap time calculation
        base_time = self.BASE_LAP_TIME + (self.track.length_km - 5) * 4.5
        
        # Driver skill impact on race pace (0-2.5 seconds, less than qualifying)
        driver_factor = self._get_driver_performance_factor(driver, is_qualifying=False)
        driver_time_delta = 2.5 * (1 - driver_factor)
        
        # Car performance impact (0-2 seconds)
        car_factor = self._get_car_performance_factor(team, is_qualifying=False)
        car_time_delta = 2.0 * (1 - car_factor)
        
        # Track modifier
        track_modifier = self._get_track_modifier(team, driver)
        
        # Weather impact
        if self.weather.is_wet:
            wet_skill_factor = driver.skill_wet / 100.0
            weather_impact = 2.0 * (1 - wet_skill_factor)
        else:
            weather_impact = 0.3 * (1 - (driver.skill_dry / 100.0))
        
        # Tire degradation impact (uses driver tire management skill)
        tire_management = driver.tire_management / 100.0 if hasattr(driver, 'tire_management') else 0.8
        tire_impact = (tire_age * 0.03) * (1.3 - tire_management * 0.3) * (self.track.tyre_wear / 5)
        
        # Fuel load impact (cars get faster as fuel burns)
        total_laps = self.track.laps
        fuel_impact = -0.025 * (lap_number / (total_laps / 10))  # Negative = faster
        
        # Controlled randomness (based on consistency)
        consistency = driver.consistency / 100.0 if hasattr(driver, 'consistency') else 0.8
        random_variance = 0.1 * (1 - consistency * 0.6)
        random_factor = random.gauss(0, random_variance)
        
        # Calculate lap time
        lap_time = (base_time + driver_time_delta + car_time_delta + 
                   weather_impact + tire_impact + fuel_impact + random_factor) * track_modifier
        
        return lap_time
    
    def _simulate_incidents(self, driver, team, lap, total_laps):
        """Simulate race incidents for a driver with 2026 regulation considerations."""
        # Base chance of incident per lap (balanced for realism)
        base_incident_chance = 0.0008  # 0.08% chance of incident per lap per driver
        
        # Adjust for driver consistency (inconsistent drivers have more incidents)
        consistency = driver.consistency / 100.0 if hasattr(driver, 'consistency') else 0.8
        driver_incident_factor = 1.8 - (consistency * 1.2)
        
        # Adjust for car reliability - significant impact
        reliability_factor = team.reliability / 100.0 if hasattr(team, 'reliability') else 0.85
        car_incident_factor = 1.8 - (reliability_factor * 1.4)
        
        # 2026: Energy system reliability (new failure mode)
        energy_reliability = team.energy_recovery / 100.0 if hasattr(team, 'energy_recovery') else 0.85
        energy_factor = 1.3 - (energy_reliability * 0.5)
        
        # Weather factor - more incidents in wet conditions
        if self.weather.condition == 'wet':
            weather_factor = 3.5
        elif self.weather.condition == 'mixed':
            weather_factor = 2.2
        else:
            weather_factor = 1.0
            
        # First lap has higher incident chance
        first_lap_factor = 6.0 if lap <= 2 else 1.0
        
        # Last laps have slightly higher incident chance due to fatigue/desperate moves
        last_lap_factor = 1.8 if lap > total_laps * 0.85 else 1.0
        
        # Driver experience factor - rookies have more incidents
        experience = driver.experience if hasattr(driver, 'experience') else 5
        exp_factor = 1.4 if experience < 3 else 1.0
        
        # Aggressive drivers have more incidents (but also more overtakes)
        aggression = driver.aggression / 100.0 if hasattr(driver, 'aggression') else 0.8
        aggression_factor = 1.0 + (aggression * 0.6)
        
        # Calculate incident chance for this lap
        incident_chance = (base_incident_chance * 
                          driver_incident_factor * 
                          car_incident_factor * 
                          energy_factor *
                          weather_factor * 
                          first_lap_factor * 
                          last_lap_factor * 
                          exp_factor *
                          aggression_factor)
        
        # Safety cap to avoid unrealistic incident rates
        incident_chance = min(incident_chance, 0.10)  # Max 10% chance per lap
        
        # Check if incident occurs
        if random.random() < incident_chance:
            # Determine incident type based on various factors
            incident_roll = random.random()
            
            # 2026: Energy system failures (new for 2026 regulations)
            if incident_roll < 0.15 * (1 - energy_reliability):
                incident = RaceIncident.ENERGY_SYSTEM_FAILURE
                descriptions = [
                    f"Battery cell failure for {driver.name}",
                    f"MGU-K overheating forces {driver.name} to retire",
                    f"Energy deployment issue for {driver.name}",
                    f"Hybrid system failure for {driver.name}",
                    f"Power unit electrical failure for {driver.name}"
                ]
            
            # Mechanical failures are more likely for less reliable cars
            elif incident_roll < 0.30 * (1 - reliability_factor) + 0.15:
                incident = RaceIncident.MECHANICAL_FAILURE
                descriptions = [
                    f"Engine failure for {driver.name}",
                    f"Gearbox issue forces {driver.name} to retire",
                    f"Hydraulic system failure for {driver.name}",
                    f"Power unit problem for {driver.name}",
                    f"Brake failure for {driver.name}"
                ]
            
            # Driver errors more common in wet conditions or for less experienced drivers
            elif incident_roll < 0.50:
                incident = RaceIncident.DRIVER_ERROR
                descriptions = [
                    f"{driver.name} spins off track",
                    f"{driver.name} locks up and goes into the gravel",
                    f"Racing incident involving {driver.name}",
                    f"{driver.name} exceeds track limits and damages the car",
                    f"Driving error forces {driver.name} to retire"
                ]
                
            # Collisions more common on first lap or in overtaking
            elif incident_roll < 0.70:
                incident = RaceIncident.COLLISION
                descriptions = [
                    f"Collision damage forces {driver.name} to retire",
                    f"{driver.name} involved in racing incident",
                    f"Contact with another car damages {driver.name}'s suspension",
                    f"Multi-car collision involves {driver.name}",
                    f"Wing damage from contact forces {driver.name} to retire"
                ]
                
            # Punctures
            elif incident_roll < 0.82:
                incident = RaceIncident.PUNCTURE
                descriptions = [
                    f"Puncture for {driver.name}",
                    f"{driver.name} suffers tire failure",
                    f"Debris causes puncture for {driver.name}",
                    f"Tire delamination for {driver.name}",
                    f"Slow puncture affects {driver.name}'s race"
                ]
            
            # Weather related
            elif incident_roll < 0.92 and self.weather.is_wet:
                incident = RaceIncident.WEATHER_RELATED
                descriptions = [
                    f"{driver.name} aquaplanes off track",
                    f"Poor visibility causes {driver.name} to crash",
                    f"{driver.name} slides off in wet conditions",
                    f"Standing water causes {driver.name} to lose control",
                    f"Wet track catches {driver.name} out"
                ]
                
            # Pit errors or penalties
            else:
                if random.random() < 0.5:
                    incident = RaceIncident.PIT_ERROR
                    descriptions = [
                        f"Pit stop error costs {driver.name} the race",
                        f"Wheel not attached properly for {driver.name}",
                        f"Fire during pit stop forces {driver.name} to retire",
                        f"Major delay in the pits for {driver.name}",
                        f"Unsafe release leads to retirement for {driver.name}"
                    ]
                else:
                    incident = RaceIncident.PENALTY
                    descriptions = [
                        f"{driver.name} black flagged for rule infringement",
                        f"Technical infringement disqualifies {driver.name}",
                        f"Safety violation forces {driver.name} to retire",
                        f"Stewards give {driver.name} black flag",
                        f"Disqualification for {driver.name}"
                    ]
            
            # Pick a random description
            description = random.choice(descriptions)
            
            return incident, description
            
        return RaceIncident.NONE, ""
    
    def simulate_race(self):
        """
        Simulate the complete race with 2026 regulations.
        Enhanced for better consistency and realism.
        
        Returns:
            List of DriverRaceResult objects with final race results
        """
        # If no qualifying results, simulate qualifying first
        if not self.grid_positions:
            self.simulate_qualifying()
        
        # Initialize race variables
        total_laps = self.track.laps
        driver_status = {driver: {'active': True, 'incident': RaceIncident.NONE, 
                                 'description': "", 'incident_lap': None, 'current_position': i+1,
                                 'tire_age': 0, 'pit_stops': 0} 
                        for i, driver in enumerate(self.grid_positions)}
        
        driver_times = {driver: 0.0 for driver in self.grid_positions}
        driver_lap_times = {driver: [] for driver in self.grid_positions}
        
        # Build the pit plan from the strategy module, adding a small random spread
        # so that drivers do not all pit on exactly the same lap.
        planned_stops = get_tire_strategy(self.track, self.weather, total_laps)
        driver_pit_laps = {}
        for driver in self.grid_positions:
            pit_laps = []
            for planned_lap, _compound in planned_stops:
                pit_lap = planned_lap + random.randint(-self.PIT_LAP_JITTER, self.PIT_LAP_JITTER)
                pit_lap = max(2, min(pit_lap, total_laps - 1))
                pit_laps.append(pit_lap)
            driver_pit_laps[driver] = sorted(set(pit_laps))
        
        # Track the fastest lap
        fastest_lap = {'driver': None, 'time': float('inf')}
        
        # Simulate lap by lap
        for lap in range(1, total_laps + 1):
            active_drivers = [d for d in self.grid_positions if driver_status[d]['active']]
            
            for driver in active_drivers:
                team = self.driver_teams[driver]
                tire_age = driver_status[driver]['tire_age']
                
                # Calculate base lap time with tire consideration
                lap_time = self._calculate_race_pace(driver, team, lap, tire_age)
                
                # Traffic factor - cars in lower positions might be held up
                position = driver_status[driver]['current_position']
                if position > 1:
                    # Overtaking difficulty affects how much time is lost in traffic
                    traffic_base = 0.08 * max(0, (position - 3) / 10)
                    # Better overtakers lose less time
                    overtaking_skill = driver.skill_overtaking / 100.0 if hasattr(driver, 'skill_overtaking') else 0.8
                    traffic_factor = traffic_base * (1.2 - overtaking_skill * 0.4)
                else:
                    traffic_factor = 0
                
                # Active aero advantage (2026) - X-Mode helps on straights
                aero_advantage = 0
                if hasattr(self.track, 'active_aero_advantage') and hasattr(team, 'active_aero'):
                    aero_efficiency = team.active_aero / 100.0
                    track_aero_benefit = self.track.active_aero_advantage / 10.0
                    aero_advantage = -0.1 * aero_efficiency * track_aero_benefit  # Negative = faster
                
                # Final lap time calculation
                final_lap_time = lap_time * (1 + traffic_factor) + aero_advantage
                
                # Ensure minimum realistic lap time
                min_lap_time = 60.0  # No lap should be under 60 seconds
                final_lap_time = max(final_lap_time, min_lap_time)
                
                # Record lap time
                driver_lap_times[driver].append(final_lap_time)
                
                # Check for fastest lap (only count clean laps after lap 5)
                if lap > 5 and final_lap_time < fastest_lap['time']:
                    fastest_lap['driver'] = driver
                    fastest_lap['time'] = final_lap_time
                
                # Apply lap time to cumulative race time
                driver_times[driver] += final_lap_time
                
                # Pit stop on a planned lap: add pit lane loss and stationary time,
                # then fit a fresh set of tires (counted as one lap old below).
                if lap in driver_pit_laps[driver]:
                    pit_efficiency = getattr(team, 'pit_efficiency', 85)
                    stop = simulate_pit_stop(pit_efficiency)
                    driver_times[driver] += self.PIT_LANE_LOSS + stop['duration']
                    driver_status[driver]['tire_age'] = 0
                    driver_status[driver]['pit_stops'] += 1
                
                # Update tire age
                driver_status[driver]['tire_age'] += 1
                
                # Check for incidents
                incident, description = self._simulate_incidents(driver, team, lap, total_laps)
                if incident != RaceIncident.NONE:
                    driver_status[driver]['active'] = False
                    driver_status[driver]['incident'] = incident
                    driver_status[driver]['description'] = description
                    driver_status[driver]['incident_lap'] = lap
            
            # Update positions based on race times
            active_drivers = [d for d in self.grid_positions if driver_status[d]['active']]
            sorted_drivers = sorted(active_drivers, key=lambda d: driver_times[d])
            
            for i, driver in enumerate(sorted_drivers):
                driver_status[driver]['current_position'] = i + 1
        
        # Compile final results
        race_results = []
        
        # First, sort all drivers by status (active first) then by position/time
        active_drivers = [d for d in self.grid_positions if driver_status[d]['active']]
        inactive_drivers = [d for d in self.grid_positions if not driver_status[d]['active']]
        
        # Sort active drivers by race time
        active_drivers.sort(key=lambda d: driver_times[d])
        
        # For inactive drivers, sort by the number of completed laps (approximated by time)
        inactive_drivers.sort(key=lambda d: driver_times[d], reverse=True)
        
        # Combine the lists - active drivers (finished) followed by inactive (DNF)
        final_positions = active_drivers + inactive_drivers
        
        # Create race results objects
        for position, driver in enumerate(final_positions, 1):
            team = self.driver_teams[driver]
            starting_pos = self.grid_positions.index(driver) + 1
            
            # Determine status
            if driver_status[driver]['active']:
                status = 'Finished'
            elif driver_status[driver]['incident'] == RaceIncident.PENALTY:
                status = 'DSQ'  # Disqualified
            else:
                status = 'DNF'  # Did Not Finish
            
            # Calculate positions gained
            positions_gained = starting_pos - position
                
            # Create result object
            result = DriverRaceResult(
                driver=driver,
                team=team,
                starting_position=starting_pos,
                finishing_position=position,
                time=driver_times[driver],
                status=status,
                fastest_lap=(driver == fastest_lap['driver']),
                incident=driver_status[driver]['incident'],
                incident_description=driver_status[driver]['description'],
                incident_lap=driver_status[driver]['incident_lap'],
                positions_gained=positions_gained,
                pit_stops=driver_status[driver]['pit_stops']
            )
            
            race_results.append(result)
        
        assign_points(race_results)
        self.race_results = race_results
        self.race_simulated = True
        
        return race_results
    
    def get_results(self):
        """Get race results, simulating first if needed."""
        if not self.race_simulated:
            self.simulate_race()
        return self.race_results
    
    def calculate_lap_time(self, driver, team, lap_number, weather, tire_degradation=0):
        """Calculate lap time for enhanced model compatibility."""
        return self._calculate_race_pace(driver, team, lap_number, tire_degradation)