"""
Core race simulation model for Formula 1 race predictions.
This model simulates race outcomes based on driver skills, car performance, track characteristics,
weather conditions, and various random events that can happen during a race.
"""

import random
import numpy as np
from dataclasses import dataclass
from enum import Enum


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
    points: int = 0
    
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


class RaceSimulator:
    """Simulates a Formula 1 race with realistic outcomes."""
    
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
            # Will be generated during simulation
            self.grid_positions = []
            
        self.race_results = []
        self.race_simulated = False
    
    def simulate_qualifying(self):
        """
        Simulate qualifying session to determine grid positions.
        
        Returns:
            Ordered list of drivers based on qualifying performance
        """
        qualifying_performances = []
        
        for driver in self.drivers:
            team = self.driver_teams[driver]
            
            # Base qualifying time calculation
            # Average lap time varies by track, approximately 90-110 seconds, adjusted for track length
            base_time = 90 + (self.track.length_km - 5) * 5
            
            # Driver skill impact (0-5 seconds)
            driver_skill = driver.get_overall_rating()
            driver_factor = 5 * (1 - (driver_skill / 100))
            
            # Car performance impact (0-3 seconds)
            car_rating = team.get_car_rating()
            car_factor = 3 * (1 - (car_rating / 100))
            
            # Track specialization - some drivers perform better at certain tracks
            track_specialization = random.uniform(-0.5, 0.5)
            
            # Weather impact
            if self.weather.is_wet:
                weather_impact = 2 * (1 - (driver.skill_wet / 100))
            else:
                weather_impact = 0.5 * (1 - (driver.skill_dry / 100))
                
            # Random factor (up to 0.3 seconds) - represents the unpredictability of qualifying
            random_factor = random.uniform(-0.2, 0.3)
            
            # Calculate lap time
            lap_time = base_time + driver_factor + car_factor + track_specialization + weather_impact + random_factor
            
            # Add some variation for the 3 qualifying laps
            q1_time = lap_time * random.uniform(1.001, 1.01)
            q2_time = lap_time * random.uniform(0.995, 1.005)
            q3_time = lap_time * random.uniform(0.99, 1.005)
            
            # Best time is the minimum of the three
            best_time = min(q1_time, q2_time, q3_time)
            
            qualifying_performances.append((driver, best_time))
        
        # Sort by lap time
        qualifying_performances.sort(key=lambda x: x[1])
        
        # Store grid positions
        self.grid_positions = [driver for driver, _ in qualifying_performances]
        
        return self.grid_positions
    
    def _calculate_race_pace(self, driver, team):
        """Calculate the race pace for a driver-team combination."""
        # Base lap time calculation
        base_time = 90 + (self.track.length_km - 5) * 5
        
        # Driver skill impact on race pace (0-3 seconds)
        driver_skill = driver.get_overall_rating()
        driver_factor = 3 * (1 - (driver_skill / 100))
        
        # Car performance impact (0-2.5 seconds)
        car_rating = team.get_car_rating()
        car_factor = 2.5 * (1 - (car_rating / 100))
        
        # Weather impact
        if self.weather.is_wet:
            weather_impact = 1.5 * (1 - (driver.skill_wet / 100))
        else:
            weather_impact = 0.5 * (1 - (driver.skill_dry / 100))
        
        # Randomness factor (±0.2 seconds)
        random_factor = random.uniform(-0.2, 0.2)
        
        # Calculate lap time
        lap_time = base_time + driver_factor + car_factor + weather_impact + random_factor
        
        return lap_time
    
    def _simulate_incidents(self, driver, team, lap, total_laps):
        """Simulate race incidents for a driver."""
        # Base chance of incident per lap
        base_incident_chance = 0.0005  # 0.05% chance of incident per lap per driver
        
        # Adjust for driver consistency (inconsistent drivers have more incidents)
        driver_incident_factor = 1 - (driver.consistency / 100)
        
        # Adjust for car reliability
        car_incident_factor = 1 - (team.reliability / 100)
        
        # Weather factor - more incidents in wet conditions
        if self.weather.condition == 'wet':
            weather_factor = 3.0
        elif self.weather.condition == 'mixed':
            weather_factor = 1.8
        else:
            weather_factor = 1.0
            
        # First lap has higher incident chance
        first_lap_factor = 5.0 if lap < 3 else 1.0
        
        # Last laps have slightly higher incident chance due to fatigue/desperate moves
        last_lap_factor = 1.5 if lap > total_laps * 0.9 else 1.0
        
        # Calculate incident chance for this lap
        incident_chance = (base_incident_chance * 
                          driver_incident_factor * 
                          car_incident_factor * 
                          weather_factor * 
                          first_lap_factor * 
                          last_lap_factor)
        
        # Check if incident occurs
        if random.random() < incident_chance:
            # Determine incident type based on various factors
            incident_roll = random.random()
            
            # Mechanical failures are more likely for less reliable cars
            if incident_roll < 0.3 * (1 - team.reliability/100) * 2:
                incident = RaceIncident.MECHANICAL_FAILURE
                descriptions = [
                    f"Engine failure for {driver.name}",
                    f"Gearbox issue forces {driver.name} to retire",
                    f"Hydraulic system failure for {driver.name}",
                    f"Power unit problem for {driver.name}",
                    f"Brake failure for {driver.name}"
                ]
            
            # Driver errors more common in wet conditions or for less experienced drivers
            elif incident_roll < 0.5:
                incident = RaceIncident.DRIVER_ERROR
                descriptions = [
                    f"{driver.name} spins off track",
                    f"{driver.name} locks up and goes into the gravel",
                    f"Racing incident involving {driver.name}",
                    f"{driver.name} exceeds track limits and damages the car",
                    f"Driving error forces {driver.name} to retire"
                ]
                
            # Collisions more common on first lap or in overtaking
            elif incident_roll < 0.7:
                incident = RaceIncident.COLLISION
                descriptions = [
                    f"Collision damage forces {driver.name} to retire",
                    f"{driver.name} involved in racing incident",
                    f"Contact with another car damages {driver.name}'s suspension",
                    f"Multi-car collision involves {driver.name}",
                    f"Wing damage from contact forces {driver.name} to retire"
                ]
                
            # Punctures
            elif incident_roll < 0.8:
                incident = RaceIncident.PUNCTURE
                descriptions = [
                    f"Puncture for {driver.name}",
                    f"{driver.name} suffers tire failure",
                    f"Debris causes puncture for {driver.name}",
                    f"Tire delamination for {driver.name}",
                    f"Slow puncture affects {driver.name}'s race"
                ]
            
            # Weather related
            elif incident_roll < 0.9 and self.weather.is_wet:
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
        Simulate the complete race.
        
        Returns:
            List of DriverRaceResult objects with final race results
        """
        # If no qualifying results, simulate qualifying first
        if not self.grid_positions:
            self.simulate_qualifying()
        
        # Initialize race variables
        total_laps = self.track.laps
        driver_status = {driver: {'active': True, 'incident': RaceIncident.NONE, 
                                 'description': "", 'current_position': i+1} 
                        for i, driver in enumerate(self.grid_positions)}
        
        driver_times = {driver: 0.0 for driver in self.grid_positions}
        
        # Track the fastest lap
        fastest_lap = {'driver': None, 'time': float('inf')}
        
        # Simulate lap by lap
        for lap in range(1, total_laps + 1):
            active_drivers = [d for d in self.grid_positions if driver_status[d]['active']]
            
            for driver in active_drivers:
                team = self.driver_teams[driver]
                
                # Calculate base lap time
                lap_time = self._calculate_race_pace(driver, team)
                
                # Adjust for tire wear - lap times increase as the race progresses
                tire_degradation = 0.05 * (lap / 20) * (self.track.tyre_wear / 10)
                
                # Adjust for fuel load - cars get faster as fuel burns (about 0.2-0.3s improvement per 10 laps)
                fuel_factor = -0.02 * (lap / 10)
                
                # Traffic factor - cars in lower positions might be held up
                position = driver_status[driver]['current_position']
                traffic_factor = 0.1 * max(0, (position - 5) / 10)
                
                # Random variation per lap
                random_variation = random.uniform(-0.3, 0.3)
                
                # Final lap time
                final_lap_time = lap_time * (1 + tire_degradation + fuel_factor + traffic_factor + random_variation)
                
                # Check for fastest lap
                if final_lap_time < fastest_lap['time']:
                    fastest_lap['driver'] = driver
                    fastest_lap['time'] = final_lap_time
                
                # Apply lap time to cumulative race time
                driver_times[driver] += final_lap_time
                
                # Check for incidents
                incident, description = self._simulate_incidents(driver, team, lap, total_laps)
                if incident != RaceIncident.NONE:
                    driver_status[driver]['active'] = False
                    driver_status[driver]['incident'] = incident
                    driver_status[driver]['description'] = description
            
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
                incident_description=driver_status[driver]['description']
            )
            
            # Calculate points (2023 system)
            if status == 'Finished':
                points_system = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
                if position <= len(points_system):
                    result.points = points_system[position-1]
                    
                # Add point for fastest lap if in top 10
                if result.fastest_lap and position <= 10:
                    result.points += 1
            
            race_results.append(result)
            
        self.race_results = race_results
        self.race_simulated = True
        
        return race_results
    
    def get_results(self):
        """Get race results, simulating first if needed."""
        if not self.race_simulated:
            self.simulate_race()
        return self.race_results