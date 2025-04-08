# F1 Race Prediction Simulator
![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Fmehmetkahya0%2Ff1-race-prediction&label=VISITORS&labelColor=%23333333&countColor=%23f34324&style=flat)



A sophisticated Formula 1 race simulation tool that models and predicts F1 race outcomes with realistic parameters based on driver skills, team performance, track characteristics, and dynamic weather conditions.

![F1 Simulation](race_progress.png)

## 🏎️ Features

- **Complete Race Weekend Simulation**: Simulate qualifying sessions and full race events
- **Realistic Driver & Team Modeling**: Uses attributes like driver skill, consistency, wet weather ability, team performance, and reliability
- **Dynamic Weather System**: Simulates realistic weather patterns based on track location and season
- **Advanced Race Strategy**: Models tire compounds, degradation, and pit stop strategies
- **Race Incidents**: Simulates mechanical failures, driver errors, collisions, and other race-affecting events
- **Detailed Analysis**: Provides in-depth statistics and visualizations of race results
- **Advanced Visualizations**: Generates multiple graph types including tire degradation, lap times, driver comparisons, and team performance
- **2025 Season Data**: Includes fictional data for the hypothetical 2025 F1 season
- **Live Race Simulation**: Watch races unfold with lap-by-lap position updates and race highlights
- **Driver Comparison Tools**: Compare any two drivers across multiple races and metrics
- **Team Performance Analytics**: Detailed heatmap analysis of team performance factors
- **Enhanced Weather Effects**: Weather now affects tire degradation, grip levels, and incident probability
- **Realistic Pit Stop Simulation**: Includes potential pit stop errors and time losses based on team efficiency
- **Consistent Visualization Naming**: All generated graphs use a standardized naming format for better organization

## 📊 Simulation Parameters

The simulation models numerous aspects of F1 racing:

### Drivers
- Overall skill rating
- Wet and dry condition performance
- Overtaking ability
- Consistency rating
- Experience level
- Aggression factor

### Teams & Cars
- Overall car performance
- Reliability metrics
- Aerodynamic efficiency
- Power unit performance
- Pit crew efficiency

### Tracks
- Circuit characteristics (length, corners, straights)
- Overtaking difficulty
- Tire wear intensity
- Downforce requirements
- Braking severity

### Weather
- Dynamic weather patterns based on location and season
- Temperature effects on tire performance
- Rain intensity modeling
- Changing track conditions

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Required packages: numpy, pandas, matplotlib, tabulate, colorama, seaborn

### Installation

1. Clone this repository:
```bash
git clone https://github.com/mehmetkahya0/f1-prediction.git
cd f1-prediction
```

2. Install dependencies:
```bash
pip install numpy pandas matplotlib tabulate colorama seaborn
```

3. Run the simulator:
```bash
python main.py
```

## 🎮 Usage

The application provides an interactive console interface:

1. Choose a race from the 2025 F1 calendar
2. Select weather conditions (realistic, dry, wet, or mixed)
3. View qualifying and race results
4. Explore various analysis options:
   - Compare qualifying and race performance
   - View detailed race statistics
   - Visualize basic race progress
   - Generate advanced visualizations
   - Generate all visualizations at once
   - Simulate live race with commentary and sector times
   - Compare driver performances across multiple races

### 📊 Visualization Types

The simulator now offers multiple visualization types:

1. **Race Progress Visualization**
   - Shows position changes lap by lap
   - Highlights pit stop windows
   - Uses official F1 team colors
   - Marks race phases (start, mid-race, finish)

2. **Tire Degradation Chart**
   - Models tire performance over race distance
   - Shows different tire compounds (soft, medium, hard)
   - Displays performance drop-off for each stint
   - Simulates compound-specific degradation rates

3. **Lap Time Progression**
   - Tracks lap times throughout the race
   - Shows effects of fuel load and tire wear
   - Highlights pit stops and their effect on lap times
   - Displays sector-specific performance variations

4. **Driver Performance Radar Chart**
   - Compares top drivers across multiple attributes
   - Visualizes driver strengths and weaknesses
   - Provides insights into race performance factors
   - Includes qualification performance metrics

5. **Position Changes Chart**
   - Shows positions gained or lost during the race
   - Highlights over-performers and under-performers
   - Color-coded for quick visual interpretation

6. **Team Performance Analysis**
   - Heatmap of key team performance metrics
   - Bar chart of team points using official colors
   - Comparative analysis of car performance vs. reliability vs. points scored

7. **Live Race Simulation Display**
   - Real-time position updates with delta indicators
   - Sector-by-sector timing information
   - Highlighted fastest sector and lap times
   - Dynamic race commentary and incident reporting
   - Gap to leader calculation for each driver

All visualizations are saved to the `visualized-graphs` folder with a consistent naming format: `graphtype_circuit_date.png`

## 📂 Project Structure

```
├── main.py                  # Main application entry point
├── data/                    # Data models for teams, drivers, and tracks
│   ├── drivers.py           # 2025 F1 driver attributes
│   ├── teams.py             # 2025 F1 team characteristics
│   └── tracks.py            # 2025 F1 track information
├── models/                  # Core simulation models
│   ├── race_model.py        # Race simulation engine
│   ├── strategy.py          # Tire and pit stop strategy simulation
│   └── weather.py           # Weather simulation system
├── utils/                   # Utility modules
│   ├── stats.py             # Statistical analysis tools
│   ├── visualization.py     # Basic visualization functions
│   └── visualization_graphs.py # Advanced graph generation
└── visualized-graphs/       # Directory containing generated visualizations
```

## 📈 Sample Output

The simulator provides detailed race results with position changes, timing data, and race incidents:

```
=================================================================================
2025 FORMULA 1 GRAND PRIX - MONACO GRAND PRIX
Location: Monte Carlo, Monaco
Track Length: 3.337km - 78 laps (260km)
Weather: Dry - 24.3°C, Rain: 0%
=================================================================================

RACE RESULTS
--------------------------------------------------------------------------------
| Pos | Driver           | Team           | Start | Change | Time/Status     | Pts |
|-----|------------------|----------------|-------|--------|-----------------|-----|
|  1  | Max Verstappen   | Red Bull Racing|   1   | →      | 01:42:23.456    | 25  |
|  2  | Charles Leclerc  | Ferrari        |   2   | →      | 01:42:28.791    | 18  |
|  3  | Lando Norris     | McLaren        |   4   | ↑1     | 01:42:34.102 FL | 16  |
|  4  | Lewis Hamilton   | Ferrari        |   3   | ↓1     | 01:42:39.889    | 12  |
|  5  | George Russell   | Mercedes       |   5   | →      | 01:42:45.231    | 10  |
|  6  | Oscar Piastri    | McLaren        |   7   | ↑1     | 01:42:50.778    | 8   |
|  7  | Fernando Alonso  | Aston Martin   |   6   | ↓1     | 01:42:55.446    | 6   |
|  8  | Carlos Sainz     | Williams       |   8   | →      | 01:43:02.112    | 4   |
|  9  | Esteban Ocon     | Haas           |  13   | ↑4     | 01:43:10.993    | 2   |
| 10  | Yuki Tsunoda     | Red Bull Racing|  10   | →      | 01:43:15.776    | 1   |
| 11  | Gabriel Bortoleto| Kick Sauber    |  11   | →      | 01:43:20.444    |     |
```

## 🔧 Customization

The simulation parameters can be customized:
- Edit driver attributes in `data/drivers.py`
- Modify team characteristics in `data/teams.py`
- Adjust track properties in `data/tracks.py`
- Fine-tune simulation algorithms in the models directory
- Create custom weather scenarios by modifying the weather generation parameters
- Adjust incident probability factors in the race simulation engine

## 🆕 Recent Updates

- **Enhanced Race Incident System**: More realistic simulation of mechanical failures, driver errors and race-affecting events
- **Improved Visualization Engine**: Better color schemes and more informative legends
- **Live Race Commentary**: Dynamic race updates with realistic F1-style commentary
- **Sector-by-Sector Analysis**: Break down driver performance by individual track sectors
- **Race History Graph**: Visualize the complete race story with gap-to-leader progression
- **Weather Impact Modeling**: More sophisticated effects of weather on race performance and strategy
- **Team Reliability Factors**: Car reliability now plays a more significant role in race outcomes
- **Driver Consistency Modeling**: Driver performance varies more realistically throughout the race
- **Enhanced Pit Stop Strategy**: More sophisticated pit stop timing and compound selection algorithms

## 📝 License

This project is released under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Formula 1 for inspiration
- The open-source Python community for amazing libraries
- Coffee, for making this project possible

## 📬 Contact

For any questions, suggestions, or feedback, feel free to reach out:

- **Author**: Mehmet Kahya  
- **Email**: mehmetkahyakas5@gmail.com 
- **GitHub**: [mehmetkahya0](https://github.com/mehmetkahya0)  
- 
We'd love to hear from you!

---

*Note: This is a simulation project using fictional data for the 2025 season. It is not affiliated with or endorsed by Formula 1, FIA, or any F1 team.*