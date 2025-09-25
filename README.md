# Marathon & Running Data Analysis

This project creates a suite of Python scripts to process, analyse, and enrich my personal running data leading up to the 2025 Manchester Marathon exported from Strava, Apple Health, and Samsung Health. The primary goal is to make my own analysis on all of the data created from my training routine up to the marathon as a vessel to explore different insights.

## Read the Full Analysis

Check out my detailed analysis and insights on Medium: [**Running on Bachelor's Time: A Marathon Analysis**](https://medium.com/@nab-iel/running-on-bachelors-time-a-marathon-analysis-23946d2f5b11)
## Project Structure

```
marathon-data/
│
├── combined_analysis.ipynb     # Main Jupyter notebook with comprehensive analysis
├── README.md                   # Project documentation
├── tester.py                   # Testing script
├── tester.fit                  # Sample FIT file for testing
├── RUNNING-01.03.2025 18.34.kml  # GPS tracking data
├── RUNNING-01.03.2025 18.34.tcx  # Training Center XML data
│
├── apple_health_export/
│   ├── organise.py             # Main script for Apple Health data processing
│   ├── export.xml              # Raw Apple Health export
│   ├── export_cda.xml          # Clinical Document Architecture export
│   └── workout-routes/         # GPX files from Apple Health workouts
│
├── output/                     # Processed CSVs generated from Apple Health XML
│   ├── workouts.csv            # Workout session data
│   ├── heart_rate.csv          # Heart rate measurements
│   ├── distance.csv            # Distance tracking data
│   ├── calories.csv            # Calorie burn data
│   └── routes.csv              # Route and GPS data
│
├── samsung_data/               # Samsung Health export data
│   ├── organise.py             # Samsung Health data processing script
│   ├── com.samsung.health.*.csv  # Various health metrics
│   ├── com.samsung.shealth.*.csv # Samsung Health app data
│   ├── files/                  # Additional Samsung data files  
│   └── jsons/                  # JSON formatted Samsung data
│
└── strava/
    ├── organise.py             # Main script for Strava data processing
    └── strava-data/            # Unzipped Strava export folder
        ├── activities.csv      # Strava activities summary
        └── activities/         # Individual activity GPX files
```

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd marathon-data
    ```

2.  **Prerequisites**: Ensure you have Python 3.8+ installed.

3.  **Install dependencies**: The scripts rely on `pandas`, `gpxpy`, and `numpy`.
    ```bash
    pip install pandas gpxpy numpy
    ```