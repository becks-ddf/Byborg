import pandas as pd

tags = [
    "Apollo 11", "Moon", "Spacecraft", "Orbiter", "Lander", "Rover", "Probe", "Shuttle", "Satellite", "Space Station",
    "Space Capsule", "Space Telescope", "Interstellar", "Extraterrestrial", "NASA", "ESA", "Roscosmos", "ISRO", "CNSA",
    "Launch Vehicle", "Rocket", "Booster", "Thruster", "Propulsion", "Fuel", "Solar Panel", "Antenna", "Docking",
    "Command Module", "Service Module", "Heat Shield", "Parachute", "Escape System", "Crewed Mission",
    "Uncrewed Mission", "Space Exploration", "Deep Space", "Low Earth Orbit", "Geostationary Orbit", "Spacewalk",
    "Extravehicular Activity", "Gravity Assist", "Orbital Insertion", "Reentry", "Landing", "Mars Mission",
    "Moon Mission", "Venus Mission", "Jupiter Mission", "Saturn Mission", "Asteroid Belt", "Kuiper Belt", "Oort Cloud",
    "Exoplanet", "Habitat", "Colonization", "Terraforming", "Space Mining", "Space Tourism", "Commercial Spaceflight",
    "Reusable Rocket", "Launch Pad", "Mission Control", "Telemetry", "Trajectory", "Orbital Mechanics",
    "Thrust-to-Weight Ratio", "Payload", "Fairing", "Stage Separation", "Vacuum", "Microgravity", "Artificial Gravity",
    "Radiation Shielding", "Life Support", "Space Suit", "EVA Suit", "MMU", "Space Habitat", "Space Colony",
    "Lunar Base", "Mars Base", "Astrobiology", "Astrophysics", "Cosmology", "Stellar Navigation",
    "In-Situ Resource Utilization", "Cryogenic Fuel", "Ion Propulsion", "Plasma Propulsion", "Fusion Propulsion",
    "Antimatter Propulsion", "Space Debris", "Orbital Decay", "Space Junk", "Space Weather", "Solar Flare",
    "Cosmic Rays", "Van Allen Belts", "Magnetosphere", "Atmospheric Entry", "Suborbital Flight",
    "Trans-lunar Injection", "Trans-Martian Injection", "Aerobraking", "Gravity Well", "Lagrange Points",
    "Hohmann Transfer", "Biological Contamination", "Planetary Protection", "Deep Space Network",
    "Space Communications", "Interplanetary Travel", "Intergalactic Travel", "Space Elevator", "Launch Window",
    "Mission Timeline", "Spacecraft Design", "Spacecraft Testing", "Spacecraft Manufacturing", "Rocket Engine",
    "Liquid Fuel", "Solid Fuel", "Hybrid Fuel", "Cryogenics", "Space Law", "Space Policy", "Astronaut", "Cosmonaut",
    "Taikonaut", "Space Program", "Space Agency", "Space Race", "Space Treaty", "Outer Space", "Space History",
    "Space Future", "Space Economy", "Space Business", "Space Industry", "Space Conference", "Space Symposium",
    "Space Research", "Space Innovation", "Space Technology", "Space Science", "Space Medicine", "Space Psychology",
    "Space Education", "Space Training", "Space Simulation", "Zero Gravity", "Microgravity Research",
    "Space Experiment", "Space Laboratory", "Space Observatory", "Space Exploration Rover", "Mars Rover", "Lunar Rover",
    "Space Probe", "Deep Space Probe", "Interstellar Probe", "Planetary Rover", "Solar System", "Galaxy", "Universe",
    "Black Hole", "Wormhole", "White Dwarf", "Red Giant", "Neutron Star", "Pulsar", "Quasar", "Supernova",
    "Dark Matter", "Dark Energy", "Big Bang", "Cosmic Microwave Background", "Gravitational Waves", "Space Art",
    "Space Culture", "Science Fiction", "Space Opera", "Starship", "Alien Life", "UFO", "Space Documentary",
    "Space Movie", "Space TV Show"
]

tags_df = pd.DataFrame(tags, columns=["name"])

file_path = "dummy_tags.csv"
tags_df.to_csv(file_path, index=True)
