"""
Visualization utilities for F1 race simulation.
"""

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate as tabulate_func
import time
import sys
from colorama import init, Fore, Style
import random
# Initialize colorama for cross-platform colored terminal output
init()

def display_race_header(track, weather):
    """Display a styled header for race information."""
    print("\n" + "=" * 80)
    print(f"{Fore.CYAN}2025 FORMULA 1 GRAND PRIX - {track.name.upper()}{Style.RESET_ALL}")
    print(f"Location: {track.city}, {track.country}")
    print(f"Track Length: {track.length_km}km - {track.laps} laps ({int(track.length_km * track.laps)}km)")
    print(f"Weather: {weather}")
    if weather.is_wet:
        print(f"{Fore.BLUE}Wet conditions - Rain intensity: {weather.rain_intensity:.1f}/10{Style.RESET_ALL}")
    print("=" * 80 + "\n")

def display_qualifying_results(qualifying_results):
    """Display the qualifying results in a formatted table."""
    print(f"\n{Fore.YELLOW}QUALIFYING RESULTS{Style.RESET_ALL}")
    print("-" * 60)
    
    table_data = []
    for i, driver in enumerate(qualifying_results, 1):
        table_data.append([
            f"{i:2d}",
            f"{driver.name}",
            f"{driver.team}",
            f"{driver.number}"
        ])
    
    headers = ["Pos", "Driver", "Team", "No."]
    print(tabulate_func(table_data, headers=headers, tablefmt="pipe"))
    print("")

def display_race_results(results):
    """Display the race results in a formatted table."""
    print(f"\n{Fore.GREEN}RACE RESULTS{Style.RESET_ALL}")
    print("-" * 80)
    
    table_data = []
    for result in results:
        pos_change = result.starting_position - result.finishing_position
        if pos_change > 0:
            pos_indicator = f"{Fore.GREEN}↑{pos_change}{Style.RESET_ALL}"
        elif pos_change < 0:
            pos_indicator = f"{Fore.RED}↓{abs(pos_change)}{Style.RESET_ALL}"
        else:
            pos_indicator = "→"
            
        status = result.status
        if status == "Finished":
            if result.fastest_lap:
                time_display = f"{format_race_time(result.time)} {Fore.MAGENTA}FL{Style.RESET_ALL}"
            else:
                time_display = format_race_time(result.time)
        else:
            time_display = f"{Fore.RED}{status}{Style.RESET_ALL}"
            
        table_data.append([
            f"{result.finishing_position:2d}",
            f"{result.driver.name}",
            f"{result.driver.team}",
            f"{result.starting_position:2d}",
            pos_indicator,
            time_display,
            f"{result.points}" if result.points > 0 else ""
        ])
    
    headers = ["Pos", "Driver", "Team", "Start", "Change", "Time/Status", "Pts"]
    print(tabulate_func(table_data, headers=headers, tablefmt="pipe"))
    
    # Display incidents
    incidents = [r for r in results if r.incident_description]
    if incidents:
        print(f"\n{Fore.YELLOW}RACE INCIDENTS:{Style.RESET_ALL}")
        for incident in incidents:
            print(f"  • Lap {incident.incident_description}")

def format_race_time(seconds):
    """Format race time in minutes:seconds.milliseconds format."""
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{minutes:02d}:{remaining_seconds:02d}.{milliseconds:03d}"

def plot_race_progress(results, laps):
    """Plot race progress showing positions by lap."""
    top_drivers = [r.driver.name for r in results[:10]]
    positions = np.random.randint(1, 21, size=(laps, len(top_drivers)))
    
    # Ensure final positions match results
    positions[-1] = list(range(1, len(top_drivers) + 1))
    
    plt.figure(figsize=(12, 8))
    x = list(range(1, laps + 1))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_drivers)))
    
    for i, driver in enumerate(top_drivers):
        plt.plot(x, positions[:, i], label=driver, color=colors[i], linewidth=2)
    
    plt.gca().invert_yaxis()  # Invert Y-axis so position 1 is at the top
    plt.yticks(range(1, len(top_drivers) + 1))
    plt.title('Race Position by Lap')
    plt.xlabel('Lap')
    plt.ylabel('Position')
    plt.grid(True)
    plt.legend(loc='center right')
    
    plt.tight_layout()
    plt.savefig('race_progress.png')
    print(f"\nRace progress plot saved to 'race_progress.png'")

def simulate_live_race_progress(results, laps, track_name):
    """Simulate live race progress with animated console updates."""
    print(f"\n{Fore.GREEN}SIMULATING LIVE RACE - {track_name}{Style.RESET_ALL}")
    print("-" * 60)
    
    # Get top 10 drivers for display
    top_drivers = [r.driver for r in results[:10]]
    
    # Generate random position changes during the race (more realistic would use actual sim data)
    positions = {}
    for driver in top_drivers:
        # Start at qualifying position and end at final position
        start_pos = next(r.starting_position for r in results if r.driver == driver)
        end_pos = next(r.finishing_position for r in results if r.driver == driver)
        
        # Create a plausible position progression
        current_pos = start_pos
        driver_positions = [current_pos]
        
        for lap in range(1, laps):
            # More position changes early in the race
            change_probability = 0.1 if lap < laps * 0.2 else 0.03
            
            if random.random() < change_probability:
                # Move towards final position
                if current_pos < end_pos:
                    current_pos += random.randint(0, 2)
                else:
                    current_pos -= random.randint(0, 2)
                    
                # Keep within bounds
                current_pos = max(1, min(current_pos, len(top_drivers)))
            
            driver_positions.append(current_pos)
            
        # Ensure final position is correct
        driver_positions[-1] = end_pos
        positions[driver] = driver_positions
    
    # Simulate race progress
    lap_times = {}
    for driver in top_drivers:
        # Base lap time around 90 seconds with some variation
        base_time = 90 + random.uniform(-2, 2)
        lap_times[driver] = [base_time]
        
        # Generate lap times with some consistent variation per driver
        driver_consistency = random.uniform(0.3, 1.0)
        for lap in range(1, laps):
            variation = random.uniform(-0.5, 0.5) * driver_consistency
            lap_times[driver].append(base_time + variation)
    
    # Display each lap with a small delay
    laps_to_show = min(laps, 20)  # Limit to avoid spamming console
    lap_interval = max(1, laps // laps_to_show)
    
    for lap in range(0, laps, lap_interval):
        if lap > 0:
            time.sleep(0.5)  # Small delay between lap updates
            
        lap_display = lap + 1
        print(f"\nLap {lap_display}/{laps}")
        print("-" * 30)
        
        # Get current positions for this lap
        current_positions = [(driver, positions[driver][lap]) for driver in top_drivers]
        current_positions.sort(key=lambda x: x[1])
        
        # Display current order
        for pos, (driver, _) in enumerate(current_positions, 1):
            lap_time = lap_times[driver][lap]
            print(f"{pos:2d}. {driver.name:20s} - {lap_time:.3f}s")
    
    print(f"\n{Fore.GREEN}RACE COMPLETED!{Style.RESET_ALL}")

