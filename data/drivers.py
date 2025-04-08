"""
Module containing data for F1 2025 drivers with their attributes and statistics.
"""

class Driver:
    def __init__(self, name, team, number, nationality, age, experience, 
                 skill_wet, skill_dry, skill_overtaking, consistency, aggression):
        self.name = name
        self.team = team
        self.number = number
        self.nationality = nationality
        self.age = age
        self.experience = experience  # Years in F1
        self.skill_wet = skill_wet  # Performance in wet conditions (1-100)
        self.skill_dry = skill_dry  # Performance in dry conditions (1-100)
        self.skill_overtaking = skill_overtaking  # Overtaking ability (1-100)
        self.consistency = consistency  # Consistency during race (1-100)
        self.aggression = aggression  # Aggressive driving style (1-100)
        
    def __str__(self):
        return f"{self.name} ({self.team})"
    
    def get_overall_rating(self):
        """Calculate the overall driver rating based on their skills."""
        return (self.skill_dry * 0.35 + 
                self.skill_wet * 0.15 + 
                self.skill_overtaking * 0.20 + 
                self.consistency * 0.20 + 
                self.experience * 0.10)


# 2025 F1 Drivers with realistic attributes
# Data is based on 2025 season performance and standings
DRIVERS = {
    "verstappen": Driver("Max Verstappen", "Red Bull Racing", 1, "Dutch", 28, 11, 94, 96, 93, 92, 85),
    "tsunoda": Driver("Yuki Tsunoda", "Red Bull Racing", 22, "Japanese", 25, 5, 78, 80, 78, 76, 82),
    "leclerc": Driver("Charles Leclerc", "Ferrari", 16, "Monegasque", 28, 8, 88, 90, 87, 84, 82),
    "hamilton": Driver("Lewis Hamilton", "Ferrari", 44, "British", 40, 19, 85, 86, 85, 84, 75),
    "russell": Driver("George Russell", "Mercedes", 63, "British", 27, 6, 92, 93, 90, 89, 83),
    "antonelli": Driver("Kimi Antonelli", "Mercedes", 87, "Italian", 19, 0, 90, 89, 85, 82, 83),
    "norris": Driver("Lando Norris", "McLaren", 4, "British", 26, 7, 96, 97, 94, 95, 87),
    "piastri": Driver("Oscar Piastri", "McLaren", 81, "Australian", 24, 3, 92, 93, 90, 91, 85),
    "alonso": Driver("Fernando Alonso", "Aston Martin", 14, "Spanish", 44, 24, 85, 87, 86, 82, 88),
    "stroll": Driver("Lance Stroll", "Aston Martin", 18, "Canadian", 26, 9, 82, 83, 80, 79, 75),
    "gasly": Driver("Pierre Gasly", "Alpine", 10, "French", 29, 9, 80, 82, 80, 79, 78),
    "doohan": Driver("Jack Doohan", "Alpine", 5, "Australian", 21, 0, 76, 77, 75, 72, 70),
    "hulkenberg": Driver("Nico Hulkenberg", "Kick Sauber", 27, "German", 38, 12, 83, 84, 82, 84, 78),
    "bortoleto": Driver("Gabriel Bortoleto", "Kick Sauber", 20, "Brazilian", 20, 0, 75, 76, 74, 70, 72),
    "lawson": Driver("Liam Lawson", "Racing Bulls", 15, "New Zealander", 23, 2, 77, 78, 77, 75, 76),
    "hadjar": Driver("Isack Hadjar", "Racing Bulls", 38, "French", 19, 0, 80, 81, 79, 76, 80),
    "albon": Driver("Alexander Albon", "Williams", 23, "Thai", 29, 7, 86, 88, 84, 85, 79),
    "sainz": Driver("Carlos Sainz", "Williams", 55, "Spanish", 31, 11, 90, 92, 89, 90, 78),
    "ocon": Driver("Esteban Ocon", "Haas", 31, "French", 29, 9, 86, 87, 85, 84, 82),
    "bearman": Driver("Oliver Bearman", "Haas", 50, "British", 20, 1, 83, 86, 84, 80, 85)
}

def get_driver_by_name(name):
    """Get a driver by their full or partial name."""
    name = name.lower()
    for driver_id, driver in DRIVERS.items():
        if name in driver.name.lower():
            return driver
    return None

def get_all_drivers():
    """Return all drivers."""
    return list(DRIVERS.values())

def get_drivers_by_team(team_name):
    """Get all drivers from a specific team."""
    team_name = team_name.lower()
    return [driver for driver in DRIVERS.values() if team_name in driver.team.lower()]