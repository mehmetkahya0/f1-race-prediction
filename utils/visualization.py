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
    """Plot race progress showing positions by lap with realistic race dynamics."""
    # Get top drivers to display
    top_drivers = [r.driver for r in results[:10]]
    driver_names = [r.driver.name for r in results[:10]]
    
    # Generate realistic position data
    positions = np.zeros((laps, len(top_drivers)), dtype=int)
    
    # Starting positions
    for i, driver in enumerate(top_drivers):
        start_pos = next(r.starting_position for r in results if r.driver == driver)
        # Convert position to array index (0-based)
        if start_pos <= 10:  # Only track top 10
            positions[0, i] = start_pos
    
    # Generate realistic position changes considering:
    # - Strong starts/poor starts
    # - Gradual position changes rather than random jumps
    # - Strategy-based position changes (pit stops)
    # - Convergence to final positions
    
    # Simulate pit stop laps (typically 1-3 stops depending on track)
    pit_stop_count = random.randint(1, 3)
    race_distance = 1/3 if pit_stop_count == 1 else 1/4 if pit_stop_count == 2 else 1/5
    
    pit_stops = {}
    for i, driver in enumerate(top_drivers):
        # Create realistic pit stop laps
        driver_pit_stops = []
        for stop in range(pit_stop_count):
            # Distribute stops throughout race with some variation between drivers
            base_lap = int((stop + 1) * race_distance * laps)
            variation = random.randint(-3, 3)
            pit_lap = min(max(base_lap + variation, 5), laps-5)  # Keep within sensible bounds
            driver_pit_stops.append(pit_lap)
        pit_stops[i] = driver_pit_stops
    
    # Fill in position progression
    for i, driver in enumerate(top_drivers):
        start_pos = next(r.starting_position for r in results if r.driver == driver)
        if start_pos > 10:  # Skip drivers who start outside top 10
            continue
            
        end_pos = next(r.finishing_position for r in results if r.driver == driver)
        
        # Convert to 0-based for array indexing but keep 1-based for display
        positions[0, i] = start_pos
        
        current_pos = start_pos
        
        # Start phase (first 10% of race) - more position changes
        first_phase = int(laps * 0.1)
        # Middle phase - more stable with occasional position changes
        middle_phase = int(laps * 0.7)
        
        # Simulate position changes throughout the race
        for lap in range(1, laps):
            # Check if this is a pit stop lap
            pit_effect = 0
            if lap in pit_stops[i]:
                # Lose positions during pit stop
                pit_effect = random.randint(2, 4)  # Lose 2-4 places
            
            # Position change probability varies by race phase
            if lap < first_phase:
                # Early race - more position changes, especially for better drivers
                driver_skill = next(d.skill_overtaking for d in top_drivers if d == driver)
                skill_factor = (driver_skill - 80) / 20  # Normalize to approximately -1 to 1
                change_prob = 0.2 + (0.1 * skill_factor)
                max_change = 2
            elif lap < middle_phase:
                # Middle race - more stable
                change_prob = 0.05
                max_change = 1
            else:
                # End race - some final position battles
                change_prob = 0.1
                max_change = 1
            
            # Direction of position change tends toward final position
            direction = -1 if current_pos > end_pos else 1 if current_pos < end_pos else 0
            
            # Determine position change
            if random.random() < change_prob and direction != 0:
                position_change = random.randint(1, max_change) * direction
            else:
                position_change = 0
                
            # Apply pit effect (always negative - lose positions)
            if pit_effect > 0:
                position_change = pit_effect
            
            # Calculate new position
            new_pos = current_pos + position_change
            
            # Ensure position remains in valid range
            new_pos = max(1, min(new_pos, 10))
            
            # Force convergence to final position in last 10% of race
            if lap > laps * 0.9:
                if new_pos < end_pos:
                    new_pos += min(1, end_pos - new_pos)
                elif new_pos > end_pos:
                    new_pos -= min(1, new_pos - end_pos)
            
            positions[lap, i] = new_pos
            current_pos = new_pos
            
        # Ensure final position is correct
        positions[-1, i] = end_pos
    
    # Create the visualization
    plt.figure(figsize=(14, 9))
    
    # Use team colors for drivers (approximated)
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
    
    # Create x-axis ticks at regular intervals
    xticks_step = max(1, laps // 10)
    xticks = list(range(0, laps, xticks_step))
    if laps - 1 not in xticks:
        xticks.append(laps - 1)
    
    # Plot each driver's position
    for i, driver in enumerate(top_drivers):
        # Skip drivers who don't start in top 10
        if next(r.starting_position for r in results if r.driver == driver) > 10:
            continue
            
        color = team_colors.get(driver.team, f'C{i}')  # Use team color or fallback
        
        # Plot the main line
        plt.plot(range(laps), positions[:, i], label=driver.name, color=color, linewidth=2.5)
        
        # Mark pit stops with vertical lines
        for pit_lap in pit_stops[i]:
            if 0 <= pit_lap < laps:
                plt.axvline(x=pit_lap, ymin=0, ymax=0.03, color=color, linewidth=1.5)
        
        # Add points to mark major events (race start, race end)
        plt.plot(0, positions[0, i], 'o', color=color, markersize=8)
        plt.plot(laps-1, positions[-1, i], 'D', color=color, markersize=8)
    
    # Set plot properties
    plt.gca().invert_yaxis()  # Invert Y-axis so position 1 is at the top
    plt.yticks(range(1, 11))
    plt.title('Race Position Progression by Lap', fontsize=16, fontweight='bold')
    plt.xlabel('Lap', fontsize=12)
    plt.ylabel('Position', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Add grid in the background
    plt.grid(True, linestyle='-', alpha=0.1)
    
    # Highlight first/last laps with vertical lines
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3, label='_nolegend_')
    plt.axvline(x=laps-1, color='black', linestyle='-', alpha=0.3, label='_nolegend_')
    
    # Add text annotations for race phases
    plt.text(laps*0.05, 0.5, 'START', ha='center', fontsize=8, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    plt.text(laps*0.5, 0.5, 'MID RACE', ha='center', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    plt.text(laps*0.95, 0.5, 'FINISH', ha='center', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # For each pit stop window, add a shaded area
    for window in range(pit_stop_count):
        window_start = int((window + 0.7) * race_distance * laps)
        window_end = int((window + 1.3) * race_distance * laps)
        plt.axvspan(window_start, window_end, alpha=0.1, color='gray')
        plt.text(
            (window_start + window_end) / 2, 0.5, 
            f'PIT WINDOW {window+1}', ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
    
    # Add a title with race information
    plt.suptitle('2025 F1 Race Position Progression', fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('race_progress.png', dpi=300, bbox_inches='tight')
    print(f"\n{Fore.GREEN}Enhanced race progress plot saved to 'race_progress.png'{Style.RESET_ALL}")

def simulate_live_race_progress(results, laps, track_name):
    """Simulate live race progress with animated console updates."""
    print(f"\n{Fore.GREEN}SIMULATING LIVE RACE - {track_name}{Style.RESET_ALL}")
    print("-" * 60)
    
    # Get top 10 drivers for display
    top_drivers = [r.driver for r in results[:10]]
    
    # Generate realistic position changes during the race
    positions = {}
    last_position = {}
    sector_times = {}
    fastest_sectors = [{} for _ in range(3)]  # 3 sectors
    fastest_lap = {"driver": None, "time": float('inf')}
    
    for driver in top_drivers:
        # Start at qualifying position and end at final position
        start_pos = next(r.starting_position for r in results if r.driver == driver)
        end_pos = next(r.finishing_position for r in results if r.driver == driver)
        last_position[driver] = start_pos
        
        # Create a realistic position progression
        current_pos = start_pos
        driver_positions = [current_pos]
        
        # Initial pace based on driver/car performance
        driver_result = next(r for r in results if r.driver == driver)
        
        # Simulate race performance with more realistic patterns
        for lap in range(1, laps):
            # First lap has more position changes
            if lap == 1:
                change_probability = 0.35
            # Early race has more changes as cars settle into rhythm
            elif lap < laps * 0.2:
                change_probability = 0.15
            # Mid race is more stable
            elif lap < laps * 0.7:
                change_probability = 0.05
            # Late race has some changes due to strategy/tire wear
            else:
                change_probability = 0.08
            
            # Adjust probability based on driver's overtaking skill and current position
            overtaking_factor = driver.skill_overtaking / 100
            position_factor = 1 + (current_pos - 5) / 10  # Higher positions have fewer changes
            final_probability = change_probability * overtaking_factor * position_factor
            
            if random.random() < final_probability:
                # Strategic progression toward final position with varied step sizes
                if current_pos > end_pos:  # Need to move up
                    # Early in race, bigger position changes are possible
                    max_step = 2 if lap < laps * 0.3 else 1
                    step = random.randint(1, max_step)
                    current_pos = max(1, current_pos - step)
                elif current_pos < end_pos:  # Will drop back
                    # More likely to drop back in second half of race (tire wear/strategy)
                    drop_probability = 0.3 if lap > laps * 0.5 else 0.1
                    if random.random() < drop_probability:
                        step = random.randint(1, 2)
                        current_pos = min(len(top_drivers), current_pos + step)
            
            # Keep within bounds and ensure progression toward final position
            if lap > laps * 0.8:  # Force convergence to final position in last 20%
                if current_pos < end_pos:
                    current_pos += min(1, end_pos - current_pos)
                elif current_pos > end_pos:
                    current_pos -= min(1, current_pos - end_pos)
                    
            driver_positions.append(current_pos)
        
        # Ensure final position is correct
        driver_positions[-1] = end_pos
        positions[driver] = driver_positions
    
    # Simulate sector and lap times with realistic patterns
    for driver in top_drivers:
        # Base lap time varies by track but typically around 90 seconds
        base_lap_time = 90.0 + random.uniform(-3, 3)
        driver_consistency = driver.consistency / 100
        
        # Better drivers have slightly faster base times
        skill_factor = driver.skill_dry / 100
        base_lap_time -= (skill_factor - 0.8) * 2.0
        
        sector_times[driver] = []
        
        for lap in range(laps):
            # Fuel effect - cars get faster as fuel burns off (about 0.1s per lap)
            fuel_effect = -0.1 * min(lap, laps * 0.7)  # Effect diminishes after 70%
            
            # Tire effect - performance drops after peak (around lap 6-10 for softer compounds)
            if lap < 5:  # Warming up
                tire_effect = -0.05 * lap
            elif lap < 10:  # Peak performance
                tire_effect = -0.25
            else:  # Degradation (more pronounced for less consistent drivers)
                tire_effect = 0.02 * (lap - 10) * (1.2 - driver_consistency)
            
            # Random variation inversely proportional to consistency
            variation = random.uniform(-0.8, 0.8) * (1.1 - driver_consistency)
            
            # Calculated lap time with all effects
            lap_time = base_lap_time + fuel_effect + tire_effect + variation
            
            # Generate realistic sector times (sum to lap time)
            s1_percent = random.uniform(0.28, 0.32)
            s2_percent = random.uniform(0.38, 0.42)
            s3_percent = 1 - s1_percent - s2_percent
            
            s1 = lap_time * s1_percent
            s2 = lap_time * s2_percent
            s3 = lap_time * s3_percent
            
            # Track fastest sectors
            if lap > 0:  # Skip formation lap
                for i, sector_time in enumerate([s1, s2, s3]):
                    if driver not in fastest_sectors[i] or sector_time < fastest_sectors[i].get(driver, float('inf')):
                        fastest_sectors[i][driver] = sector_time
            
            sector_times[driver].append((s1, s2, s3, lap_time))
            
            # Track fastest lap
            if lap_time < fastest_lap["time"]:
                fastest_lap["driver"] = driver
                fastest_lap["time"] = lap_time
    
    # Display race progress with enhanced information
    laps_to_show = min(laps, 20)  # Limit to avoid console spam
    lap_interval = max(1, laps // laps_to_show)
    
    for lap in range(0, laps, lap_interval):
        if lap > 0:
            time.sleep(0.5)  # Small delay between lap updates
            
        lap_display = lap + 1
        print(f"\n{Fore.CYAN}Lap {lap_display}/{laps}{Style.RESET_ALL}")
        print("-" * 60)
        
        # Get current positions for this lap
        current_positions = [(driver, positions[driver][lap]) for driver in top_drivers]
        current_positions.sort(key=lambda x: x[1])
        
        # Prepare display table
        table_data = []
        
        # Track position changes for display
        for pos, (driver, position) in enumerate(current_positions, 1):
            prev_pos = last_position.get(driver, pos)
            pos_change = prev_pos - pos
            
            if pos_change > 0:
                pos_indicator = f"{Fore.GREEN}↑{pos_change}{Style.RESET_ALL}"
            elif pos_change < 0:
                pos_indicator = f"{Fore.RED}↓{abs(pos_change)}{Style.RESET_ALL}"
            else:
                pos_indicator = "  "
                
            # Get sector times
            if lap < len(sector_times[driver]):
                s1, s2, s3, lap_time = sector_times[driver][lap]
                
                # Highlight fastest sectors if this is the lap with fastest sector
                s1_color = Fore.MAGENTA if fastest_sectors[0].get(driver) == s1 else ""
                s2_color = Fore.MAGENTA if fastest_sectors[1].get(driver) == s2 else ""
                s3_color = Fore.MAGENTA if fastest_sectors[2].get(driver) == s3 else ""
                lap_color = Fore.MAGENTA if fastest_lap["driver"] == driver and fastest_lap["time"] == lap_time else ""
                
                time_display = f"{s1_color}{s1:.3f}s{Style.RESET_ALL} | {s2_color}{s2:.3f}s{Style.RESET_ALL} | {s3_color}{s3:.3f}s{Style.RESET_ALL} | {lap_color}{lap_time:.3f}s{Style.RESET_ALL}"
            else:
                time_display = "Starting grid"
                
            # Get gap to leader
            if pos == 1:
                gap_display = "LEADER"
            else:
                leader_driver = current_positions[0][0]
                if lap < len(sector_times[leader_driver]) and lap < len(sector_times[driver]):
                    # Cumulative time gap to leader
                    leader_total = sum(t[3] for t in sector_times[leader_driver][:lap+1])
                    driver_total = sum(t[3] for t in sector_times[driver][:lap+1])
                    gap = driver_total - leader_total
                    gap_display = f"+{gap:.3f}s"
                else:
                    gap_display = ""
            
            table_data.append([
                f"{pos:2d}",
                f"{pos_indicator}",
                f"{driver.name:20s}",
                f"{driver.team:15s}",
                time_display,
                gap_display
            ])
            
            # Update last position for next lap
            last_position[driver] = pos
        
        # Display the table
        headers = ["Pos", "Δ", "Driver", "Team", "Sectors | Lap Time", "Gap"]
        print(tabulate_func(table_data, headers=headers, tablefmt="pipe"))
        
        # Show race highlights occasionally
        if random.random() < 0.2 and lap > 0:
            highlight_types = [
                f"{Fore.YELLOW}Good defending from {random.choice(top_drivers).name} against {random.choice(top_drivers).name}!{Style.RESET_ALL}",
                f"{Fore.CYAN}DRS enabled for {random.choice(top_drivers).name} - closing in!{Style.RESET_ALL}",
                f"{Fore.GREEN}Great overtake at Turn {random.randint(1, 15)}!{Style.RESET_ALL}",
                f"{Fore.RED}Lock-up for {random.choice(top_drivers).name} at Turn {random.randint(1, 15)}{Style.RESET_ALL}",
                f"{Fore.YELLOW}Yellow flags in sector {random.randint(1, 3)} - incident being investigated{Style.RESET_ALL}"
            ]
            if lap > laps // 3:
                highlight_types.append(f"{Fore.CYAN}Pit window is now open - expecting stops soon{Style.RESET_ALL}")
            if lap > laps // 2:
                highlight_types.append(f"{Fore.CYAN}Teams reporting tire degradation becoming significant{Style.RESET_ALL}")
                
            print(f"\n{random.choice(highlight_types)}")
    
    # Display race summary
    print(f"\n{Fore.GREEN}RACE COMPLETED!{Style.RESET_ALL}")
    
    # Display final fastest lap and sectors
    print(f"\n{Fore.MAGENTA}FASTEST LAP: {fastest_lap['driver'].name} - {fastest_lap['time']:.3f}s{Style.RESET_ALL}")
    print("\nFastest Sectors:")
    for i, sector in enumerate(fastest_sectors, 1):
        fastest_driver = min(sector.items(), key=lambda x: x[1])
        print(f"Sector {i}: {fastest_driver[0].name} - {fastest_driver[1]:.3f}s")

