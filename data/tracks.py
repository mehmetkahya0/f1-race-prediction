"""
Module containing data for F1 2025 tracks with their attributes and statistics.
"""

class Track:
    def __init__(self, name, country, city, length_km, laps, corners, 
                 straight_count, top_speed, downforce_level, tyre_wear, 
                 braking_severity, overtaking_difficulty, date):
        self.name = name
        self.country = country
        self.city = city
        self.length_km = length_km  # Circuit length in km
        self.laps = laps  # Number of laps in the race
        self.corners = corners  # Number of corners
        self.straight_count = straight_count  # Number of straights
        self.top_speed = top_speed  # Expected top speed in km/h
        self.downforce_level = downforce_level  # Required downforce (1-10) where 10 is maximum
        self.tyre_wear = tyre_wear  # Tyre degradation (1-10) where 10 is highest wear
        self.braking_severity = braking_severity  # Braking intensity (1-10) where 10 is highest
        self.overtaking_difficulty = overtaking_difficulty  # Difficulty to overtake (1-10) where 10 is hardest
        self.date = date  # Race date for 2025 season
    
    def __str__(self):
        return f"{self.name} ({self.country})"
    
    def race_distance(self):
        """Calculate total race distance in km."""
        return self.length_km * self.laps
    
    def track_type(self):
        """Determine track type based on characteristics."""
        if self.downforce_level >= 8:
            return "High Downforce"
        elif self.downforce_level <= 4:
            return "Low Downforce"
        else:
            return "Medium Downforce"
    
    def track_summary(self):
        """Return a summary of track characteristics."""
        return {
            "name": self.name,
            "country": self.country,
            "length": self.length_km,
            "laps": self.laps,
            "corners": self.corners,
            "type": self.track_type(),
            "top_speed": self.top_speed,
            "tyre_wear": self.tyre_wear,
            "overtaking": 10 - self.overtaking_difficulty,  # Convert to overtaking potential (10 is best)
            "date": self.date
        }


# 2025 F1 Calendar with track data
# Data is fictional for the 2025 season, dates are approximate
TRACKS = {
    "bahrain": Track("Bahrain International Circuit", "Bahrain", "Sakhir", 5.412, 57, 15, 4, 330, 6, 7, 7, 5, "March 2, 2025"),
    "jeddah": Track("Jeddah Corniche Circuit", "Saudi Arabia", "Jeddah", 6.174, 50, 27, 3, 350, 4, 5, 8, 6, "March 9, 2025"),
    "australia": Track("Albert Park Circuit", "Australia", "Melbourne", 5.278, 58, 14, 4, 325, 5, 6, 7, 5, "March 23, 2025"),
    "japan": Track("Suzuka Circuit", "Japan", "Suzuka", 5.807, 53, 18, 2, 315, 9, 6, 7, 7, "April 6, 2025"),
    "china": Track("Shanghai International Circuit", "China", "Shanghai", 5.451, 56, 16, 3, 327, 6, 7, 8, 5, "April 13, 2025"),
    "miami": Track("Miami International Autodrome", "USA", "Miami", 5.412, 57, 19, 3, 340, 5, 6, 7, 4, "May 4, 2025"),
    "imola": Track("Autodromo Enzo e Dino Ferrari", "Italy", "Imola", 4.909, 63, 19, 2, 320, 7, 5, 9, 8, "May 18, 2025"),
    "monaco": Track("Circuit de Monaco", "Monaco", "Monte Carlo", 3.337, 78, 19, 1, 290, 10, 3, 10, 10, "May 25, 2025"),
    "canada": Track("Circuit Gilles Villeneuve", "Canada", "Montreal", 4.361, 70, 14, 3, 330, 6, 8, 8, 4, "June 8, 2025"),
    "spain": Track("Circuit de Barcelona-Catalunya", "Spain", "Barcelona", 4.675, 66, 16, 2, 325, 8, 7, 6, 7, "June 22, 2025"),
    "austria": Track("Red Bull Ring", "Austria", "Spielberg", 4.318, 71, 10, 3, 340, 5, 6, 7, 3, "June 29, 2025"),
    "britain": Track("Silverstone Circuit", "Great Britain", "Silverstone", 5.891, 52, 18, 2, 330, 8, 7, 7, 5, "July 6, 2025"),
    "hungary": Track("Hungaroring", "Hungary", "Budapest", 4.381, 70, 14, 1, 315, 9, 5, 7, 9, "July 20, 2025"),
    "belgium": Track("Circuit de Spa-Francorchamps", "Belgium", "Spa", 7.004, 44, 19, 2, 350, 6, 5, 9, 4, "July 27, 2025"),
    "netherlands": Track("Circuit Zandvoort", "Netherlands", "Zandvoort", 4.259, 72, 14, 2, 315, 8, 7, 7, 7, "August 24, 2025"),
    "monza": Track("Autodromo Nazionale Monza", "Italy", "Monza", 5.793, 53, 11, 4, 360, 1, 8, 9, 4, "September 7, 2025"),
    "azerbaijan": Track("Baku City Circuit", "Azerbaijan", "Baku", 6.003, 51, 20, 2, 350, 4, 5, 8, 6, "September 14, 2025"),
    "singapore": Track("Marina Bay Street Circuit", "Singapore", "Singapore", 4.94, 62, 23, 2, 325, 8, 9, 9, 7, "September 21, 2025"),
    "usa": Track("Circuit of the Americas", "USA", "Austin", 5.513, 56, 20, 3, 330, 7, 6, 7, 5, "October 19, 2025"),
    "mexico": Track("Autódromo Hermanos Rodríguez", "Mexico", "Mexico City", 4.304, 71, 17, 3, 350, 6, 7, 8, 6, "October 26, 2025"),
    "brazil": Track("Autódromo José Carlos Pace", "Brazil", "São Paulo", 4.309, 71, 15, 2, 335, 7, 8, 7, 4, "November 9, 2025"),
    "las_vegas": Track("Las Vegas Strip Circuit", "USA", "Las Vegas", 6.12, 50, 17, 3, 345, 3, 6, 7, 5, "November 16, 2025"),
    "qatar": Track("Losail International Circuit", "Qatar", "Lusail", 5.38, 57, 16, 1, 330, 8, 9, 7, 6, "November 30, 2025"),
    "abu_dhabi": Track("Yas Marina Circuit", "UAE", "Abu Dhabi", 5.281, 58, 16, 2, 335, 7, 6, 7, 7, "December 7, 2025")
}

def get_track_by_name(name):
    """Get a track by its full or partial name."""
    name = name.lower()
    for track_id, track in TRACKS.items():
        if name in track.name.lower() or name in track.country.lower():
            return track
    return None

def get_all_tracks():
    """Return all tracks."""
    return list(TRACKS.values())

def get_calendar():
    """Return the tracks in calendar order (by date)."""
    import datetime
    
    def parse_date(date_string):
        return datetime.datetime.strptime(date_string, "%B %d, %Y")
    
    return sorted(TRACKS.values(), key=lambda track: parse_date(track.date))