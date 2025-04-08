"""
Module containing data for F1 2025 teams with their attributes and statistics.
"""

class Team:
    def __init__(self, name, constructor, engine, performance, reliability, 
                 pit_efficiency, development_rate, aerodynamics, power):
        self.name = name
        self.constructor = constructor
        self.engine = engine
        self.performance = performance  # Overall car performance (1-100)
        self.reliability = reliability  # Car reliability rating (1-100)
        self.pit_efficiency = pit_efficiency  # Pit stop efficiency (1-100)
        self.development_rate = development_rate  # In-season development rate (1-100)
        self.aerodynamics = aerodynamics  # Aerodynamic efficiency (1-100)
        self.power = power  # Engine power (1-100)
    
    def __str__(self):
        return f"{self.name} ({self.engine})"
    
    def get_car_rating(self):
        """Calculate the overall car rating."""
        return (self.performance * 0.3 + 
                self.reliability * 0.2 + 
                self.aerodynamics * 0.25 + 
                self.power * 0.25)


# 2025 F1 Teams with realistic attributes
# Data is fictional for the 2025 season
TEAMS = {
    "red_bull": Team("Red Bull Racing", "Red Bull", "Red Bull Powertrains", 95, 92, 94, 93, 97, 94),
    "ferrari": Team("Ferrari", "Ferrari", "Ferrari", 93, 88, 92, 90, 93, 95),
    "mercedes": Team("Mercedes", "Mercedes", "Mercedes", 92, 94, 95, 91, 92, 93),
    "mclaren": Team("McLaren", "McLaren", "Mercedes", 94, 90, 93, 92, 95, 93),
    "aston_martin": Team("Aston Martin", "Aston Martin", "Mercedes", 88, 86, 90, 87, 89, 93),
    "alpine": Team("Alpine", "Alpine", "Renault", 84, 82, 88, 85, 86, 87),
    "williams": Team("Williams", "Williams", "Mercedes", 83, 85, 87, 86, 85, 93),
    "rb": Team("RB", "RB", "Red Bull Powertrains", 82, 87, 85, 85, 84, 94),
    "sauber": Team("Sauber", "Sauber", "Ferrari", 80, 83, 84, 80, 82, 95),
    "haas": Team("Haas", "Haas", "Ferrari", 81, 80, 81, 82, 83, 95)
}

def get_team_by_name(name):
    """Get a team by their full or partial name."""
    name = name.lower()
    for team_id, team in TEAMS.items():
        if name in team.name.lower():
            return team
    return None

def get_all_teams():
    """Return all teams."""
    return list(TEAMS.values())

def get_team_by_engine(engine):
    """Get all teams using a specific engine manufacturer."""
    engine = engine.lower()
    return [team for team in TEAMS.values() if engine in team.engine.lower()]