"""
Advanced visualization functions for F1 race predictions.
This module provides additional graph types beyond the basic race progress visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os

# Create visualized-graphs directory if it doesn't exist
GRAPH_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "visualized-graphs")
if not os.path.exists(GRAPH_DIR):
    os.makedirs(GRAPH_DIR)

def plot_tire_degradation(laps, results, track_name):
    """
    Plot tire degradation over race distance with compound changes.
    
    Args:
        laps: Number of laps in the race
        results: List of race results
        track_name: Name of the track for the title
    """
    plt.figure(figsize=(12, 8))
    
    # Get top 5 drivers to display
    top_drivers = [r.driver for r in results[:5]]
    
    # Generate tire compound changes
    # We'll simulate typically 1-3 stops
    compounds = ['SOFT', 'MEDIUM', 'HARD', 'SOFT', 'MEDIUM']
    compounds_colors = {'SOFT': 'red', 'MEDIUM': 'yellow', 'HARD': 'white', 
                        'INTERMEDIATE': 'green', 'WET': 'blue'}
    
    # Track segments per driver
    segments = {}
    legends = []
    
    for i, driver in enumerate(top_drivers):
        # Number of pit stops for this driver
        n_stops = random.randint(1, 3)
        stop_laps = sorted([random.randint(15, laps-15) for _ in range(n_stops)])
        stop_laps = [0] + stop_laps + [laps]
        
        # Assign compounds
        driver_compounds = [compounds[j % len(compounds)] for j in range(n_stops + 1)]
        
        # Store segments for this driver
        segments[driver.name] = list(zip(stop_laps, driver_compounds))
        
        # Initial performance is around 100%
        performance = 100
        x_values = [0]
        y_values = [performance]
        
        # Color changes for compound changes
        colors = []
        
        # Plot each stint
        for j in range(len(stop_laps) - 1):
            start_lap = stop_laps[j]
            end_lap = stop_laps[j+1]
            compound = driver_compounds[j]
            
            # Degradation factors vary by compound
            if compound == 'SOFT':
                deg_factor = 0.6
            elif compound == 'MEDIUM':
                deg_factor = 0.4
            else:  # HARD
                deg_factor = 0.25
            
            # Plot each lap in the stint
            for lap in range(start_lap + 1, end_lap + 1):
                # Tire starts at ~100% and degrades based on compound
                # Random element to create realistic curves
                random_factor = random.uniform(-0.05, 0.05)
                
                # Performance drop accelerates slightly as tires age
                age_factor = ((lap - start_lap) / 10) ** 1.2
                performance = 100 - (deg_factor * age_factor * (lap - start_lap) + random_factor)
                
                x_values.append(lap)
                y_values.append(performance)
                colors.append(compounds_colors[compound])
                
            # After pit stop, performance resets to ~100%
            if j < len(stop_laps) - 2:  # If not the last stint
                x_values.append(end_lap)
                y_values.append(100)
        
        # Plot the performance curve
        plt.plot(x_values, y_values, label=f"{driver.name} - {driver.team}", 
                 linewidth=2, alpha=0.8)
        
        # Add compound indicators
        for j, (lap, compound) in enumerate(segments[driver.name]):
            if j < len(segments[driver.name]) - 1:
                next_lap = segments[driver.name][j+1][0]
                plt.axvspan(lap, next_lap, alpha=0.1, 
                            color=compounds_colors[compound], ymin=0.98-i*0.05, ymax=1)
    
    # Add a legend for compounds
    compound_patches = [Patch(color=color, alpha=0.5, label=comp) 
                       for comp, color in compounds_colors.items() 
                       if comp in ['SOFT', 'MEDIUM', 'HARD']]
    
    # Create legend with drivers and compounds
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(f'Tire Performance Degradation - {track_name}', fontsize=16)
    plt.xlabel('Lap', fontsize=12)
    plt.ylabel('Tire Performance (%)', fontsize=12)
    plt.ylim(70, 102)
    
    # Add a second legend for tire compounds
    ax = plt.gca()
    second_legend = plt.legend(handles=compound_patches, loc='lower left', 
                               title='Tire Compounds')
    ax.add_artist(second_legend)
    
    plt.tight_layout()
    save_path = os.path.join(GRAPH_DIR, 'tire_degradation.png')
    plt.savefig(save_path, dpi=300)
    return save_path

def plot_lap_time_progression(results, laps, track_name):
    """
    Plot lap time progression throughout the race for top drivers.
    
    Args:
        results: List of race results
        laps: Number of laps in the race
        track_name: Name of the track for the title
    """
    plt.figure(figsize=(12, 8))
    
    # Get top drivers to display
    top_drivers = [r.driver for r in results[:5]]
    
    # Generate realistic lap times with tire, fuel effects
    lap_times = {}
    
    # Base lap time around 90 seconds (adjust based on track if needed)
    base_time = 90.0
    
    # Team colors for consistency with other charts
    team_colors = {
        'Red Bull Racing': '#0600EF',  # Dark blue
        'Ferrari': '#DC0000',          # Red
        'Mercedes': '#00D2BE',         # Teal
        'McLaren': '#FF8700',          # Orange
        'Aston Martin': '#006F62',     # Green
    }
    
    for driver in top_drivers:
        # Driver-specific base time (better drivers are faster)
        skill_factor = (driver.skill_dry / 100)
        driver_base = base_time * (1 - ((skill_factor - 0.8) / 10))
        
        # Consistency affects variation
        consistency = driver.consistency / 100
        
        # Generate lap times
        times = []
        pit_laps = []
        # Generate random pit stops
        n_stops = random.randint(1, 3)
        for _ in range(n_stops):
            pit_laps.append(random.randint(15, laps-15))
        
        for lap in range(1, laps + 1):
            # Effect of fuel load (decreases over time, making car faster)
            fuel_effect = 0.2 * (1 - min(lap/laps, 1))
            
            # Effect of tire wear (increases over time until pit stop)
            last_pit = 0
            for pit in sorted(pit_laps):
                if lap > pit:
                    last_pit = pit
            
            laps_since_pit = lap - last_pit
            
            # Tire wear effect (initially improves slightly then degrades)
            if laps_since_pit <= 3:
                # New tires take a few laps to warm up
                tire_effect = 0.3 * (1 - laps_since_pit/3)
            else:
                # Then performance degrades
                tire_effect = 0.01 * (laps_since_pit - 3) ** 1.5
            
            # Random variation (more consistent drivers have less variation)
            variation = random.normalvariate(0, 0.3 * (1 - consistency))
            
            # Combined time
            lap_time = driver_base + fuel_effect + tire_effect + variation
            
            # Pit stop adds time
            if lap in pit_laps:
                # Mark pit stops with a spike in lap time
                lap_time += 20  # Approx time lost in pits
                
            times.append(lap_time)
        
        # Plot lap times for this driver
        color = team_colors.get(driver.team, None)
        plt.plot(range(1, laps + 1), times, 
                 label=f"{driver.name} ({driver.team})", 
                 linewidth=1.5, alpha=0.8, color=color)
        
        # Mark pit stops
        for pit in pit_laps:
            plt.axvline(x=pit, alpha=0.2, linestyle='--', 
                        color=color or 'gray')
        
        lap_times[driver.name] = times
    
    # Styling
    plt.grid(True, alpha=0.3)
    plt.title(f'Lap Time Progression - {track_name}', fontsize=16)
    plt.xlabel('Lap', fontsize=12)
    plt.ylabel('Lap Time (seconds)', fontsize=12)
    plt.legend()
    
    # Add shaded areas for different race phases
    plt.axvspan(0, laps * 0.1, alpha=0.1, color='green', label='Start Phase')
    plt.axvspan(laps * 0.1, laps * 0.7, alpha=0.1, color='blue', label='Mid Race')
    plt.axvspan(laps * 0.7, laps, alpha=0.1, color='red', label='End Phase')
    
    plt.tight_layout()
    save_path = os.path.join(GRAPH_DIR, 'lap_time_progression.png')
    plt.savefig(save_path, dpi=300)
    return save_path

def plot_driver_comparison(results, track_name):
    """
    Create a performance comparison chart for top drivers.
    
    Args:
        results: List of race results
        track_name: Name of the track for the title
    """
    # Extract relevant metrics
    drivers = []
    
    for r in results[:10]:  # Top 10 drivers
        # Create a data structure for each driver
        drivers.append({
            'Name': r.driver.name,
            'Team': r.driver.team,
            'Position': r.finishing_position,
            'Start': r.starting_position,
            'Gain/Loss': r.starting_position - r.finishing_position,
            'Qualification': r.starting_position,
            'Skill': r.driver.get_overall_rating(),
            'Consistency': r.driver.consistency,
            'Overtaking': r.driver.skill_overtaking,
            'Points': r.points
        })
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(drivers)
    
    # Create a radar chart for top 5 drivers
    plt.figure(figsize=(12, 10))
    
    # Radar chart attributes
    categories = ['Skill', 'Consistency', 'Overtaking', 'Points', 'Start']
    N = len(categories)
    
    # Create angle variables for radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the polygon
    
    # Initialize the spider plot
    ax = plt.subplot(111, polar=True)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=10)
    plt.ylim(0, 100)
    
    # Plotting for top 5 drivers
    for i, driver in enumerate(drivers[:5]):
        # Get values for this driver, scale as needed
        values = [
            driver['Skill'],
            driver['Consistency'],
            driver['Overtaking'],
            max(driver['Points'] * 4, 10),  # Scale points to 0-100
            100 - (driver['Start'] - 1) * 5  # Convert qualification to 0-100 scale
        ]
        values += values[:1]  # Close the polygon
        
        # Plot the driver as a polygon
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=driver['Name'])
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(f'Driver Performance Comparison - {track_name}', size=20)
    
    save_path = os.path.join(GRAPH_DIR, 'driver_comparison_radar.png')
    plt.savefig(save_path, dpi=300)
    
    # Additional bar chart comparing position gains/losses
    plt.figure(figsize=(12, 6))
    
    # Sort by position for better readability
    df_sorted = df.sort_values('Position')
    
    # Bar chart for position changes
    colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in df_sorted['Gain/Loss']]
    plt.bar(df_sorted['Name'], df_sorted['Gain/Loss'], color=colors)
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('Position Gains/Losses During Race', size=16)
    plt.xlabel('Driver')
    plt.ylabel('Positions Gained/Lost')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path2 = os.path.join(GRAPH_DIR, 'position_changes.png')
    plt.savefig(save_path2, dpi=300)
    
    return save_path, save_path2

def plot_team_performance(results):
    """
    Create visualization of team performance.
    
    Args:
        results: List of race results
    """
    # Extract team data
    teams = {}
    
    for result in results:
        team_name = result.team.name
        if team_name not in teams:
            teams[team_name] = {
                'Points': 0,
                'BestPosition': float('inf'),
                'Drivers': [],
                'DriverPositions': [],
                'CarPerformance': result.team.performance,
                'Reliability': result.team.reliability
            }
        
        # Update team stats
        teams[team_name]['Points'] += result.points
        teams[team_name]['BestPosition'] = min(teams[team_name]['BestPosition'], result.finishing_position)
        teams[team_name]['Drivers'].append(result.driver.name)
        teams[team_name]['DriverPositions'].append(result.finishing_position)
    
    # Convert to DataFrame
    team_data = []
    for team_name, stats in teams.items():
        team_data.append({
            'Team': team_name,
            'Points': stats['Points'],
            'BestPosition': stats['BestPosition'] if stats['BestPosition'] != float('inf') else 20,
            'AvgPosition': sum(stats['DriverPositions']) / len(stats['DriverPositions']),
            'CarPerformance': stats['CarPerformance'],
            'Reliability': stats['Reliability'],
            'Drivers': ', '.join(stats['Drivers'])
        })
    
    team_df = pd.DataFrame(team_data)
    
    # Sort by points for better display
    team_df = team_df.sort_values('Points', ascending=False)
    
    # Team Performance Heatmap
    plt.figure(figsize=(12, 8))
    
    # Prepare data for heatmap
    heatmap_data = team_df[['Team', 'CarPerformance', 'Reliability', 'Points']]
    heatmap_data = heatmap_data.set_index('Team')
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.1f', linewidths=.5)
    plt.title('Team Performance Metrics', size=16)
    
    save_path = os.path.join(GRAPH_DIR, 'team_performance_heatmap.png') 
    plt.savefig(save_path, dpi=300)
    
    # Team Points Bar Chart
    plt.figure(figsize=(12, 6))
    
    # Team colors
    team_colors = {
        'Red Bull Racing': '#0600EF',  # Dark blue
        'Ferrari': '#DC0000',          # Red
        'Mercedes': '#00D2BE',         # Teal
        'McLaren': '#FF8700',          # Orange
        'Aston Martin': '#006F62',     # Green
        'Alpine': '#0090FF',           # Blue
        'Williams': '#005AFF',         # Blue
        'Racing Bulls': '#052C5A',     # Navy
        'Kick Sauber': '#52E252',      # Green
        'Haas': '#FFFFFF'              # White
    }
    
    # Create color list matching the order of teams in the DataFrame
    colors = [team_colors.get(team, '#333333') for team in team_df['Team']]
    
    # Create the bar chart
    plt.bar(team_df['Team'], team_df['Points'], color=colors)
    plt.title('Team Points', size=16)
    plt.xlabel('Team')
    plt.ylabel('Points')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add point values at top of bars
    for i, points in enumerate(team_df['Points']):
        plt.text(i, points + 0.5, str(points), ha='center')
    
    plt.tight_layout()
    save_path2 = os.path.join(GRAPH_DIR, 'team_points.png')
    plt.savefig(save_path2, dpi=300)
    
    return save_path, save_path2

def generate_all_visualizations(results, laps, track_name):
    """
    Generate all visualization types and return paths to saved files.
    
    Args:
        results: Race results
        laps: Number of laps in race
        track_name: Name of the track
        
    Returns:
        Dictionary of paths to generated visualization files
    """
    visualizations = {}
    
    print("Generating tire degradation chart...")
    visualizations['tire_degradation'] = plot_tire_degradation(laps, results, track_name)
    
    print("Generating lap time progression chart...")
    visualizations['lap_times'] = plot_lap_time_progression(results, laps, track_name)
    
    print("Generating driver comparison charts...")
    radar_chart, position_chart = plot_driver_comparison(results, track_name)
    visualizations['driver_radar'] = radar_chart
    visualizations['position_changes'] = position_chart
    
    print("Generating team performance charts...")
    team_heatmap, team_points = plot_team_performance(results)
    visualizations['team_heatmap'] = team_heatmap
    visualizations['team_points'] = team_points
    
    return visualizations