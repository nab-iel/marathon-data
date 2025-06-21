# Marathon & Running Data Analysis

This project creates a suite of Python scripts to process, analyse, and enrich my personal running data leading up to the 2025 Manchester Marathon exported from Strava, Apple Health, and Samsung Health. The primary goal is to make my own analysis on all of the data created from my training routine up to the marathon as a vessel to explore different insights.

## Project Structure

```
marathon-data/
│
├── apple_health_export/
│   ├── organise.py             # Main script for Apple Health data
│   ├── export.xml              # Your raw Apple Health export
│   └── workout-routes/         # GPX files from Apple Health
│
├── output/                     # Intermediate CSVs generated from Apple Health XML
│   ├── workouts.csv
│   ├── heart_rate.csv
│   └── routes.csv
│
├── strava/
│   ├── organise.py             # Main script for Strava data
│   └── strava-data/            # Your unzipped Strava export folder
│       ├── activities.csv
│       └── activities/         # GPX files from Strava
│
└── README.md
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