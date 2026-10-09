from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

TRAIN_DATA_PATH = DATA_DIR / "train-test.csv"
VALIDATION_DATA_PATH = DATA_DIR / "validation.csv"
VALIDATION_TEMPLATE_PATH = DATA_DIR / "validation-predictions-template.csv"
DECEMBER_INPUTS_PATH = DATA_DIR / "december-chart-inputs.csv"

OUTPUT_PREDICTIONS_PATH = BASE_DIR / "validation_predictions.csv"
OUTPUT_DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"
SCORER_OUTPUT_DIR = BASE_DIR / "scorer_results"
DECEMBER_CHART_PATH = SCORER_OUTPUT_DIR / "candidate_december.png"

# City Coordinates Dictionary (1:1 mapping verified from dataset)
CITY_COORDINATES = {
    "Albany": (42.65258, -73.75623),
    "Albuquerque": (35.08449, -106.65114),
    "Allentown": (40.60843, -75.49018),
    "Amarillo": (35.22199, -101.8313),
    "Atlanta": (33.74899, -84.38798),
    "Austin": (30.26715, -97.74306),
    "Bakersfield": (35.37329, -119.01871),
    "Baltimore": (39.29038, -76.61219),
    "Baton Rouge": (30.45147, -91.18715),
    "Birmingham": (33.51859, -86.81036),
    "Boise": (43.61502, -116.20231),
    "Boston": (42.36008, -71.05888),
    "Buffalo": (42.88645, -78.87837),
    "Cedar Rapids": (41.97788, -91.66562),
    "Charleston": (32.77647, -79.93105),
    "Charlotte": (35.22709, -80.84313),
    "Chattanooga": (35.04563, -85.30968),
    "Chicago": (41.87811, -87.6298),
    "Cincinnati": (39.10312, -84.51202),
    "Cleveland": (41.49932, -81.69436),
    "Columbia": (34.00071, -81.03481),
    "Columbus": (39.96118, -82.99879),
    "Dallas": (32.77666, -96.79699),
    "Denver": (39.73924, -104.99025),
    "Des Moines": (41.58684, -93.62496),
    "Detroit": (42.33143, -83.04575),
    "El Paso": (31.76188, -106.48502),
    "Fort Wayne": (41.07927, -85.13935),
    "Fresno": (36.74684, -119.77259),
    "Grand Rapids": (42.96336, -85.66809),
    "Green Bay": (44.51916, -88.01983),
    "Greensboro": (36.07264, -79.79198),
    "Greenville": (34.85262, -82.39401),
    "Harrisburg": (40.27319, -76.8867),
    "Houston": (29.76043, -95.3698),
    "Indianapolis": (39.7684, -86.15807),
    "Jackson": (32.29876, -90.18481),
    "Jacksonville": (30.33218, -81.65565),
    "Kansas City": (39.09973, -94.57857),
    "Knoxville": (35.96064, -83.92074),
    "Laredo": (27.50356, -99.50755),
    "Las Vegas": (36.16994, -115.13983),
    "Lexington": (38.04058, -84.50372),
    "Little Rock": (34.74648, -92.28959),
    "Los Angeles": (34.05223, -118.24368),
    "Louisville": (38.25266, -85.75846),
    "Lubbock": (33.57786, -101.85517),
    "Memphis": (35.14953, -90.04898),
    "Miami": (25.76168, -80.19179),
    "Milwaukee": (43.0389, -87.90647),
    "Minneapolis": (44.97775, -93.26501),
    "Mobile": (30.69537, -88.03989),
    "Nashville": (36.16266, -86.7816),
    "New Orleans": (29.95107, -90.07153),
    "Oklahoma City": (35.46756, -97.51643),
    "Omaha": (41.25654, -95.9345),
    "Orlando": (28.53834, -81.37924),
    "Philadelphia": (39.95258, -75.16522),
    "Phoenix": (33.44838, -112.07404),
    "Pittsburgh": (40.44062, -79.99589),
    "Portland": (45.51523, -122.67838),
    "Raleigh": (35.77959, -78.63818),
    "Richmond": (37.54072, -77.43605),
    "Sacramento": (38.58157, -121.4944),
    "Salt Lake City": (40.76078, -111.89105),
    "San Antonio": (29.42412, -98.49363),
    "San Diego": (32.71574, -117.16109),
    "San Francisco": (37.77493, -122.41942),
    "Savannah": (32.08089, -81.0912),
    "Seattle": (47.60621, -122.33207),
    "Shreveport": (32.52515, -93.75018),
    "Spokane": (47.65878, -117.42605),
    "St. Louis": (38.627, -90.1994),
    "Stockton": (37.9577, -121.29078),
    "Tampa": (27.95058, -82.45718),
    "Toledo": (41.65281, -83.53787),
    "Tucson": (32.22261, -110.97471),
    "Tulsa": (36.15398, -95.99277),
    "Wichita": (37.68718, -97.33005),
}

RANDOM_SEED = 42
