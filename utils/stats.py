"""
Statistics and analysis utilities for F1 race simulation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class SeasonStats:
    """Track driver and team statistics across a season."""
    driver_points: Dict  # Driver name -> points
    team_points: Dict    # Team name -> points
    race_wins: Dict      # Driver name -> number of wins
    podiums: Dict        # Driver name -> number of podiums
    fastest_laps: Dict   # Driver name -> number of fastest laps
    pole_positions: Dict # Driver name -> number of poles
    
    @classmethod
    def new_season(cls):
        """Create a new empty season statistics object."""
        return cls(
            driver_points={},
            team_points={},
            race_wins={},
            podiums={},
            fastest_laps={},
            pole_positions={}
        )
    
    def update_with_race_results(self, results, grid_positions):
        """Update season statistics with results from a race."""
        # Update pole positions
        if grid_positions:
            pole_driver = grid_positions[0].name
            self.pole_positions[pole_driver] = self.pole_positions.get(pole_driver, 0) + 1
        
        # Update other stats
        for result in results:
            driver_name = result.driver.name
            team_name = result.team.name
            
            # Update points
            self.driver_points[driver_name] = self.driver_points.get(driver_name, 0) + result.points
            self.team_points[team_name] = self.team_points.get(team_name, 0) + result.points
            
            # Race winner
            if result.finishing_position == 1:
                self.race_wins[driver_name] = self.race_wins.get(driver_name, 0) + 1
            
            # Podium finishers
            if result.finishing_position <= 3:
                self.podiums[driver_name] = self.podiums.get(driver_name, 0) + 1
            
            # Fastest lap
            if result.fastest_lap:
                self.fastest_laps[driver_name] = self.fastest_laps.get(driver_name, 0) + 1
    
    def driver_standings(self):
        """Get the current driver standings."""
        # Convert to pandas DataFrame for easy sorting
        drivers = []
        for driver, points in self.driver_points.items():
            drivers.append({
                'Driver': driver,
                'Points': points,
                'Wins': self.race_wins.get(driver, 0),
                'Podiums': self.podiums.get(driver, 0),
                'Fastest Laps': self.fastest_laps.get(driver, 0),
                'Poles': self.pole_positions.get(driver, 0)
            })
        
        df = pd.DataFrame(drivers)
        if len(df) > 0:
            # Sort by points (descending) then by wins
            df = df.sort_values(['Points', 'Wins'], ascending=[False, False])
        
        return df
    
    def team_standings(self):
        """Get the current constructor standings."""
        # Convert to pandas DataFrame for easy sorting
        teams = []
        for team, points in self.team_points.items():
            teams.append({
                'Team': team,
                'Points': points
            })
        
        df = pd.DataFrame(teams)
        if len(df) > 0:
            df = df.sort_values('Points', ascending=False)
        
        return df


def analyze_qualifying_performance(qualifying_results, race_results):
    """
    Analyze qualifying performance vs race results.
    
    Args:
        qualifying_results: List of drivers in qualifying order
        race_results: List of DriverRaceResult objects
        
    Returns:
        DataFrame with analysis metrics
    """
    analysis = []
    
    for i, driver in enumerate(qualifying_results):
        quali_pos = i + 1
        
        # Find corresponding race result
        race_result = next((r for r in race_results if r.driver == driver), None)
        
        if race_result:
            race_pos = race_result.finishing_position
            pos_change = quali_pos - race_pos
            status = race_result.status
            
            analysis.append({
                'Driver': driver.name,
                'Qualifying': quali_pos,
                'Race': race_pos,
                'Change': pos_change,
                'Status': status
            })
    
    return pd.DataFrame(analysis)


def calculate_performance_metrics(results):
    """
    Calculate performance metrics from race results.
    
    Args:
        results: List of DriverRaceResult objects
        
    Returns:
        Dictionary with various performance metrics
    """
    # Count finished drivers
    finished = sum(1 for r in results if r.status == 'Finished')
    dnf = sum(1 for r in results if r.status == 'DNF')
    dsq = sum(1 for r in results if r.status == 'DSQ')
    
    # Calculate average time gap between positions
    times = [r.time for r in results if r.status == 'Finished']
    if len(times) > 1:
        winner_time = times[0]
        avg_gap = np.mean([(t - winner_time) / winner_time * 100 for t in times[1:]])
    else:
        avg_gap = 0
    
    return {
        'Finished': finished,
        'DNF': dnf,
        'DSQ': dsq,
        'DNF Rate': dnf / len(results) if results else 0,
        'Average Gap': avg_gap
    }


def driver_comparison(driver_name1, driver_name2, races):
    """
    Compare performance between two drivers across multiple races.
    
    Args:
        driver_name1: First driver's name
        driver_name2: Second driver's name
        races: List of dictionaries containing race results
        
    Returns:
        DataFrame with comparison metrics
    """
    comparison = []
    
    for race in races:
        track = race['track']
        results = race['results']
        
        # Find both drivers' results
        result1 = next((r for r in results if r.driver.name == driver_name1), None)
        result2 = next((r for r in results if r.driver.name == driver_name2), None)
        
        if result1 and result2:
            # Both drivers have results for this race
            comparison.append({
                'Track': track.name,
                f'{driver_name1} Position': result1.finishing_position,
                f'{driver_name1} Status': result1.status,
                f'{driver_name2} Position': result2.finishing_position,
                f'{driver_name2} Status': result2.status,
                'Position Difference': result2.finishing_position - result1.finishing_position,
                'Points Difference': result1.points - result2.points
            })
    
    return pd.DataFrame(comparison)