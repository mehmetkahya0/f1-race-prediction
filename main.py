"""
Formula 1 Race Prediction Simulator
Main application entry point for simulating F1 race outcomes.

This application simulates F1 race results based on:
- Driver statistics and skills
- Team/car performance
- Track characteristics
- Weather conditions
"""

import os
import sys
import random
import argparse
from tabulate import tabulate as tabulate_func  # Import and rename for clarity

# Import our data models
from data.drivers import get_all_drivers, get_driver_by_name
from data.teams import get_all_teams, get_team_by_name
from data.tracks import get_all_tracks, get_track_by_name, get_calendar

# Import our simulation models
from models.race_model import RaceSimulator
from models.weather import generate_weather
from models.strategy import TireCompound, get_tire_strategy

# Import visualization and stats utilities
from utils.visualization import (
    display_race_header, display_qualifying_results, display_race_results,
    plot_race_progress, simulate_live_race_progress
)
from utils.visualization_graphs import (
    plot_tire_degradation, plot_lap_time_progression,
    plot_driver_comparison, plot_team_performance,
    generate_all_visualizations
)
from utils.stats import SeasonStats, analyze_qualifying_performance, calculate_performance_metrics


def print_welcome():
    """Print welcome message with app information."""
    print("\n" + "=" * 80)
    print("FORMULA 1 RACE PREDICTION SIMULATOR - 2025 SEASON")
    print("=" * 80)
    print("This application simulates F1 race outcomes based on driver skills, car performance,")
    print("track characteristics, and variable weather conditions.")
    print("\nAll data is fictional and represents a hypothetical 2025 F1 season.")
    print("=" * 80 + "\n")


def select_track():
    """Prompt user to select a track from the 2025 calendar."""
    tracks = get_calendar()
    
    print("\nAvailable races in the 2025 F1 Calendar:")
    print("-" * 60)
    
    table_data = []
    for i, track in enumerate(tracks, 1):
        table_data.append([i, track.name, track.country, track.date])
    
    headers = ["Option", "Circuit", "Country", "Race Date"]
    print(tabulate_func(table_data, headers=headers, tablefmt="pipe"))
    
    while True:
        try:
            choice = input("\nSelect a race by number (or 'q' to quit): ")
            if choice.lower() == 'q':
                sys.exit(0)
                
            choice = int(choice)
            if 1 <= choice <= len(tracks):
                return tracks[choice - 1]
            else:
                print(f"Please enter a number between 1 and {len(tracks)}")
        except ValueError:
            print("Please enter a valid number")


def select_weather_option(track):
    """Allow user to choose weather options."""
    print("\nChoose weather conditions for the race:")
    print("1. Realistic (based on track location and season)")
    print("2. Dry race")
    print("3. Wet race")
    print("4. Mixed conditions")
    
    while True:
        choice = input("\nSelect an option (1-4): ")
        
        # Strip any whitespace and check if it's a digit
        choice = choice.strip()
        if choice.isdigit():
            choice_num = int(choice)
            
            if choice_num == 1:
                return generate_weather(track)
            elif choice_num == 2:
                return generate_weather(track, forced_condition="dry")
            elif choice_num == 3:
                return generate_weather(track, forced_condition="wet")
            elif choice_num == 4:
                return generate_weather(track, forced_condition="mixed")
            else:
                print("Please enter a number between 1 and 4")
        else:
            print("Please enter a valid number (1-4)")


def run_race_simulation(track, weather_option):
    """Run the complete race simulation."""
    # Get all drivers and teams
    drivers = get_all_drivers()
    teams = get_all_teams()
    
    # Create race simulator
    weather = weather_option
    simulator = RaceSimulator(track, drivers, teams, weather)
    
    # Simulate qualifying
    qualifying_results = simulator.simulate_qualifying()
    
    # Simulate race
    race_results = simulator.simulate_race()
    
    # Display results
    display_race_header(track, weather)
    display_qualifying_results(qualifying_results)
    display_race_results(race_results)
    
    # Offer further analysis
    return simulator, qualifying_results, race_results


def show_analysis_menu(simulator, qualifying_results, race_results):
    """Show menu for additional analysis options."""
    track = simulator.track
    
    while True:
        print("\nAnalysis Options:")
        print("1. Show qualifying vs race performance analysis")
        print("2. Show detailed race statistics")
        print("3. Visualize basic race progress")
        print("4. Generate advanced visualizations")
        print("5. Generate all visualizations (saves to visualized-graphs folder)")
        print("6. Simulate another race")
        print("7. Quit")
        
        try:
            choice = input("\nSelect an option (1-7): ")
            choice = int(choice)
            
            if choice == 1:
                # Qualifying vs race analysis
                analysis_df = analyze_qualifying_performance(qualifying_results, race_results)
                print("\nQualifying vs Race Performance:")
                print(tabulate_func(analysis_df, headers='keys', tablefmt='pipe', showindex=False))
                
            elif choice == 2:
                # Race statistics
                metrics = calculate_performance_metrics(race_results)
                print("\nRace Statistics:")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        print(f"{key}: {value:.2f}")
                    else:
                        print(f"{key}: {value}")
                        
            elif choice == 3:
                # Visualize race progress
                print("\nGenerating race progress visualization...")
                plot_race_progress(race_results, track.laps)
                simulate_live_race_progress(race_results, track.laps, track.name)
                
            elif choice == 4:
                # Advanced visualizations submenu
                show_visualization_menu(race_results, track)
                
            elif choice == 5:
                # Generate all visualizations
                print("\nGenerating all visualizations...")
                visualizations = generate_all_visualizations(race_results, track.laps, track.name)
                print("\nAll visualizations saved to the 'visualized-graphs' folder:")
                for viz_type, path in visualizations.items():
                    print(f"- {viz_type}: {os.path.basename(path)}")
                
            elif choice == 6:
                # Simulate another race
                return True
                
            elif choice == 7:
                return False
                
            else:
                print("Please enter a number between 1 and 7")
                
        except ValueError:
            print("Please enter a valid number")


def show_visualization_menu(race_results, track):
    """Show submenu for advanced visualization options."""
    while True:
        print("\nAdvanced Visualization Options:")
        print("1. Tire degradation chart")
        print("2. Lap time progression")
        print("3. Driver performance comparison")
        print("4. Team performance analysis")
        print("5. Return to main menu")
        
        try:
            choice = input("\nSelect visualization type (1-5): ")
            choice = int(choice)
            
            if choice == 1:
                print("\nGenerating tire degradation visualization...")
                path = plot_tire_degradation(track.laps, race_results, track.name)
                print(f"Visualization saved to: {path}")
                
            elif choice == 2:
                print("\nGenerating lap time progression visualization...")
                path = plot_lap_time_progression(race_results, track.laps, track.name)
                print(f"Visualization saved to: {path}")
                
            elif choice == 3:
                print("\nGenerating driver comparison visualizations...")
                paths = plot_driver_comparison(race_results, track.name)
                print(f"Visualizations saved to:")
                for path in paths:
                    print(f"- {path}")
                    
            elif choice == 4:
                print("\nGenerating team performance visualizations...")
                paths = plot_team_performance(race_results)
                print(f"Visualizations saved to:")
                for path in paths:
                    print(f"- {path}")
                    
            elif choice == 5:
                return
                
            else:
                print("Please enter a number between 1 and 5")
                
        except ValueError:
            print("Please enter a valid number")


def main():
    """Main application entry point."""
    print_welcome()
    
    while True:
        try:
            # Let user select track
            track = select_track()
            
            # Let user select weather conditions
            weather = select_weather_option(track)
            
            # Run simulation
            simulator, qualifying_results, race_results = run_race_simulation(track, weather)
            
            # Show analysis menu
            continue_simulation = show_analysis_menu(simulator, qualifying_results, race_results)
            if not continue_simulation:
                break
                
        except KeyboardInterrupt:
            print("\nExiting application...")
            break
            
    print("\nThank you for using the F1 Race Prediction Simulator!")


if __name__ == "__main__":
    # Check for required packages
    try:
        import numpy
        import pandas
        import matplotlib
        import tabulate
        import colorama
        import seaborn
    except ImportError:
        print("Missing required packages. Installing dependencies...")
        import subprocess
        subprocess.call([sys.executable, "-m", "pip", "install", 
                        "numpy", "pandas", "matplotlib", "tabulate", "colorama", "seaborn"])
        print("Dependencies installed. Starting application...")
    
    main()