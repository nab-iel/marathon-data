import os
import pandas as pd
import gpxpy
import numpy as np
import gpxpy.geo
import xml.etree.ElementTree as ET
from datetime import datetime
import re

USER_THRESHOLD_PACE = 5.5  # min/km
USER_MAX_HEART_RATE = 200   # bpm

PACE_ZONES = {
    'P1_Sprint': (0, USER_THRESHOLD_PACE * 0.8),
    'P2_Interval': (USER_THRESHOLD_PACE * 0.8, USER_THRESHOLD_PACE * 0.95),
    'P3_Tempo': (USER_THRESHOLD_PACE * 0.95, USER_THRESHOLD_PACE * 1.05),
    'P4_Aerobic': (USER_THRESHOLD_PACE * 1.05, USER_THRESHOLD_PACE * 1.25),
    'P5_Recovery': (USER_THRESHOLD_PACE * 1.25, 30),
}
PACE_ZONE_MULTIPLIERS = {'P1_Sprint': 5, 'P2_Interval': 4, 'P3_Tempo': 3, 'P4_Aerobic': 2, 'P5_Recovery': 1}

HEART_RATE_ZONES = {
    'Z1_VeryLight': (0, USER_MAX_HEART_RATE * 0.6),
    'Z2_Light': (USER_MAX_HEART_RATE * 0.6, USER_MAX_HEART_RATE * 0.7),
    'Z3_Moderate': (USER_MAX_HEART_RATE * 0.7, USER_MAX_HEART_RATE * 0.8),
    'Z4_Hard': (USER_MAX_HEART_RATE * 0.8, USER_MAX_HEART_RATE * 0.9),
    'Z5_Maximum': (USER_MAX_HEART_RATE * 0.9, USER_MAX_HEART_RATE * 2), # Cap at 2x max
}

HEART_RATE_ZONE_MULTIPLIERS = {'Z1_VeryLight': 1, 'Z2_Light': 2, 'Z3_Moderate': 3, 'Z4_Hard': 4, 'Z5_Maximum': 5}

def process_gpx_file(file_path):
    """Reads, parses, and cleans a single GPX file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as gpx_file:
            gpx = gpxpy.parse(gpx_file)
    except Exception as e:
        print(f"  - Could not parse {os.path.basename(file_path)}: {e}")
        return None

    data = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                point_data = {'timestamp': point.time, 'latitude': point.latitude, 'longitude': point.longitude, 'elevation_m': point.elevation}
                data.append(point_data)
    
    if not data: return None

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df[['latitude', 'longitude', 'elevation_m']] = df[['latitude', 'longitude', 'elevation_m']].interpolate()
    df['time_diff_s'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
    
    df['distance_km'] = 0.0
    for i in range(1, len(df)):
        distance = gpxpy.geo.haversine_distance(df.loc[i-1, 'latitude'], df.loc[i-1, 'longitude'], df.loc[i, 'latitude'], df.loc[i, 'longitude'])
        df.loc[i, 'distance_km'] = df.loc[i-1, 'distance_km'] + (distance / 1000)
    
    df['distance_diff_km'] = df['distance_km'].diff().fillna(0)
    df['pace_min_per_km'] = (df['time_diff_s'] / 60) / df['distance_diff_km']
    df.replace([np.inf, -np.inf], pd.NA, inplace=True)
    df['pace_min_per_km'] = df['pace_min_per_km'].ffill().bfill()

    return df.drop(columns=['distance_diff_km'])

def parse_apple_health_xml(xml_path):
    """Parses the Apple Health export.xml, filters the data, and saves it to CSV files in an 'output' directory."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    print("Parsing and filtering data from export.xml...")
    
    workouts = [elem.attrib for elem in root.iter('Workout')]
    workout_df = pd.DataFrame(workouts)
    initial_workout_count = len(workout_df)

    # Only running activities and from my Apple Watch
    workout_df = workout_df[
        (workout_df['workoutActivityType'] == 'HKWorkoutActivityTypeRunning') &
        (workout_df['sourceName'] != 'Strava')
    ].copy()

    all_records = root.findall('Record')
    hr_records = [r.attrib for r in all_records if r.attrib['type'] == 'HKQuantityTypeIdentifierHeartRate']
    distance_records = [r.attrib for r in all_records if r.attrib['type'] == 'HKQuantityTypeIdentifierDistanceWalkingRunning']
    calories_records = [r.attrib for r in all_records if r.attrib['type'] == 'HKQuantityTypeIdentifierActiveEnergyBurned']
    
    hr_df = pd.DataFrame(hr_records)
    distance_df = pd.DataFrame(distance_records)
    calories_df = pd.DataFrame(calories_records)
    
    routes = []
    for elem in root.iter('WorkoutRoute'):
        route_data = elem.attrib
        file_ref = elem.find('FileReference')
        if file_ref is not None:
            route_data['gpx_file_path'] = file_ref.attrib['path']
        routes.append(route_data)
    routes_df = pd.DataFrame(routes)

    print(f"Found {initial_workout_count} total workouts. After filtering, processing {len(workout_df)} activities.")
    print(f"Found {len(hr_df)} heart rate records, {len(distance_df)} distance records, and {len(calories_df)} active energy records.")
    print(f"Found {len(routes_df)} workout routes.")

    # Save to CSV
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    workout_df.to_csv(os.path.join(output_dir, 'workouts.csv'), index=False)
    hr_df.to_csv(os.path.join(output_dir, 'heart_rate.csv'), index=False)
    distance_df.to_csv(os.path.join(output_dir, 'distance.csv'), index=False)
    calories_df.to_csv(os.path.join(output_dir, 'calories.csv'), index=False)
    routes_df.to_csv(os.path.join(output_dir, 'routes.csv'), index=False)
    print(f"\nSaved filtered workouts and associated health records to '{output_dir}' directory.")

def organise_apple_health_data(export_base_path):
    """
    Organises Apple Health data by reading pre-processed CSVs and linking them
    with GPX track data.
    """
    processed_data_dir = 'output'
    routes_folder_path = os.path.join(export_base_path, 'workout-routes')
    
    workouts_csv = os.path.join(processed_data_dir, 'workouts.csv')
    hr_csv = os.path.join(processed_data_dir, 'heart_rate.csv')
    routes_csv = os.path.join(processed_data_dir, 'routes.csv')
    distance_csv = os.path.join(processed_data_dir, 'distance.csv')
    calories_csv = os.path.join(processed_data_dir, 'calories.csv')

    for f in [workouts_csv, hr_csv, routes_csv, distance_csv, calories_csv]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Processed data file not found: {f}. Please run the parsing step first.")

    print("\nReading pre-processed data from CSV files...")
    workouts_df = pd.read_csv(workouts_csv)
    hr_df = pd.read_csv(hr_csv)
    routes_df = pd.read_csv(routes_csv)
    distance_df = pd.read_csv(distance_csv)
    calories_df = pd.read_csv(calories_csv)
    print("Successfully loaded CSV data.")

    date_cols = ['startDate', 'endDate']
    for col in date_cols:
        if col in workouts_df.columns:
            workouts_df[col] = pd.to_datetime(workouts_df[col], utc=True)
        if col in routes_df.columns and not routes_df.empty:
            routes_df[col] = pd.to_datetime(routes_df[col], utc=True)
    
    for df in [hr_df, distance_df, calories_df]:
        if 'startDate' in df.columns:
            df['timestamp'] = pd.to_datetime(df['startDate'], utc=True)
            df['value'] = pd.to_numeric(df['value'])

    if 'timestamp' in hr_df.columns:
        hr_df['heart_rate_bpm'] = hr_df['value']
        hr_df = hr_df[['timestamp', 'heart_rate_bpm']].sort_values('timestamp').reset_index(drop=True)

    if not routes_df.empty and 'startDate' in routes_df.columns:
        print("\nMerging workouts with nearest workout routes...")
        workouts_df = workouts_df.sort_values('startDate').reset_index(drop=True)
        routes_df = routes_df.sort_values('startDate').reset_index(drop=True)

        # Find the route with the closest start time, within a 2-minute tolerance.
        workouts_with_routes = pd.merge_asof(
            left=workouts_df,
            right=routes_df.drop(columns=['creationDate', 'device', 'sourceName', 'sourceVersion'], errors='ignore'),
            on='startDate',
            direction='nearest',
            tolerance=pd.Timedelta('2 minutes')
        )
        
        # Invalidate matches where the workout and route durations are very different.
        if 'endDate_y' in workouts_with_routes.columns:
            duration_diff = (workouts_with_routes['endDate_x'] - workouts_with_routes['endDate_y']).abs()
            workouts_with_routes.loc[duration_diff > pd.Timedelta('2 minutes'), 'gpx_file_path'] = np.nan
            workouts_with_routes.rename(columns={'endDate_x': 'endDate'}, inplace=True)
            workouts_with_routes.drop(columns=['endDate_y'], inplace=True, errors='ignore')
        else:
             workouts_with_routes.rename(columns={'endDate_x': 'endDate'}, inplace=True)
    else:
        print("\nNo workout routes found to merge.")
        workouts_with_routes = workouts_df.copy()
        workouts_with_routes['gpx_file_path'] = np.nan

    apple_health_data_structure = {}
    print(f"\nProcessing {len(workouts_with_routes)} workouts...")

    for index, activity in workouts_with_routes.iterrows():
        activity_id = activity['startDate'].strftime('%Y%m%d_%H%M%S')
        start_time = activity['startDate']
        end_time = activity['endDate']
        print(f"\nProcessing Activity ID: {activity_id} ({activity['workoutActivityType']})")

        activity_dist_records = distance_df[
            (distance_df['timestamp'] >= start_time) & (distance_df['timestamp'] < end_time)
        ]
        total_distance_from_records = activity_dist_records['value'].sum()

        activity_cal_records = calories_df[
            (calories_df['timestamp'] >= start_time) & (calories_df['timestamp'] < end_time)
        ]
        total_calories_from_records = activity_cal_records['value'].sum()

        final_distance = activity.get('totalDistance', 0)
        if not pd.notna(final_distance) or final_distance == 0:
            final_distance = total_distance_from_records

        final_calories = activity.get('totalEnergyBurned', 0)
        if not pd.notna(final_calories) or final_calories == 0:
            final_calories = total_calories_from_records

        gpx_filename = os.path.basename(activity['gpx_file_path']) if pd.notna(activity['gpx_file_path']) else None

        metadata = {
            'Activity ID': activity_id,
            'Activity Date': activity['startDate'],
            'Activity Type': re.sub(r'HKWorkoutActivityType', '', activity['workoutActivityType']),
            'Elapsed Time': float(activity['duration']) * 60, # Convert mins to secs
            'Distance': final_distance,
            'Calories': final_calories,
            'Filename': gpx_filename
        }

        track_df = None
        if gpx_filename:
            gpx_full_path = os.path.join(routes_folder_path, gpx_filename)
            if os.path.exists(gpx_full_path):
                print(f"  - Found GPX file. Reading and cleaning...")
                track_df = process_gpx_file(gpx_full_path)
            else:
                print(f"  - GPX file not found at: {gpx_full_path}")
        else:
            print("  - No route/GPX file associated with this workout.")

        if track_df is not None and not track_df.empty:
            activity_hr = hr_df[(hr_df['timestamp'] >= activity['startDate']) & (hr_df['timestamp'] <= activity['endDate'])]
            if not activity_hr.empty:
                print(f"  - Merging {len(activity_hr)} heart rate points into track data.")
                track_df = pd.merge_asof(track_df.sort_values('timestamp'), activity_hr, on='timestamp', direction='nearest')
            else:
                print("  - No heart rate data found for this activity's timeframe.")
                track_df['heart_rate_bpm'] = np.nan

        apple_health_data_structure[activity_id] = {
            'metadata': metadata,
            'track_data': track_df
        }

    print("\nProcessing complete.")
    return apple_health_data_structure

def engineer_activity_features(track_df):
    """Calculates all new features for a single activity's track data."""
    if track_df is None or track_df.empty or 'time_diff_s' not in track_df.columns:
        return {}

    features = {}

    pace_bins = [v[0] for v in PACE_ZONES.values()] + [PACE_ZONES['P5_Recovery'][1]]
    track_df['pace_zone'] = pd.cut(track_df['pace_min_per_km'], bins=pace_bins, labels=PACE_ZONES.keys(), right=False)
    time_in_pace_zones = track_df.groupby('pace_zone', observed=False)['time_diff_s'].sum().to_dict()
    features['time_in_pace_zones_s'] = {k: v for k, v in time_in_pace_zones.items() if pd.notna(k)}
    pace_training_load = sum((duration_s / 60) * PACE_ZONE_MULTIPLIERS.get(zone, 0) for zone, duration_s in features['time_in_pace_zones_s'].items())
    features['training_load_pace'] = round(pace_training_load, 2)
    features['pace_variability_std'] = round(track_df['pace_min_per_km'].std(), 2)

    if 'heart_rate_bpm' in track_df.columns and track_df['heart_rate_bpm'].notna().any():
        hr_bins = [v[0] for v in HEART_RATE_ZONES.values()] + [HEART_RATE_ZONES['Z5_Maximum'][1]]
        track_df['hr_zone'] = pd.cut(track_df['heart_rate_bpm'], bins=hr_bins, labels=HEART_RATE_ZONES.keys(), right=False)
        time_in_hr_zones = track_df.groupby('hr_zone', observed=False)['time_diff_s'].sum().to_dict()
        features['time_in_hr_zones_s'] = {k: v for k, v in time_in_hr_zones.items() if pd.notna(k)}
        hr_trimp = sum((duration_s / 60) * HEART_RATE_ZONE_MULTIPLIERS.get(zone, 0) for zone, duration_s in features['time_in_hr_zones_s'].items())
        features['training_load_hr'] = round(hr_trimp, 2)
        features['avg_heart_rate_bpm'] = round(track_df['heart_rate_bpm'].mean(), 1)

    return features

def create_aggregate_summaries(apple_health_data_structure):
    """Creates weekly and monthly summaries from the enriched metadata."""
    if not apple_health_data_structure:
        return pd.DataFrame(), pd.DataFrame()

    metadata_list = [activity['metadata'] for activity in apple_health_data_structure.values()]
    summary_df = pd.DataFrame(metadata_list)
    
    summary_df['Activity Date'] = pd.to_datetime(summary_df['Activity Date'])
    summary_df.set_index('Activity Date', inplace=True)
    
    agg_ops = {'Activity ID': 'count'}
    potential_cols = {
        'Distance': 'sum', 'Elapsed Time': 'sum', 'Calories': 'sum',
        'training_load_pace': 'sum', 'training_load_hr': 'sum'
    }
    
    for col, op in potential_cols.items():
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce').fillna(0)
            agg_ops[col] = op
            
    weekly_summary = summary_df.resample('W-Mon', label='left', closed='left').agg(agg_ops).rename(columns={'Activity ID': 'Activity Count'})
    monthly_summary = summary_df.resample('ME').agg(agg_ops).rename(columns={'Activity ID': 'Activity Count'})
    
    for df in [weekly_summary, monthly_summary]:
        if 'Elapsed Time' in df.columns:
            df['Duration_hours'] = round(df['Elapsed Time'] / 3600, 2)
            df.drop(columns='Elapsed Time', inplace=True)
        
    return weekly_summary, monthly_summary

if __name__ == '__main__':
    apple_health_export_directory = './apple_health_export' 

    if not os.path.isdir(apple_health_export_directory):
        print("="*50)
        print(f"ERROR: The specified directory '{apple_health_export_directory}' does not exist.")
        print("Please update the variable with the correct path to your Apple Health export.")
        print("="*50)
    else:
        # Parse the raw XML and save to clean CSV files.
        xml_file_path = os.path.join(apple_health_export_directory, 'export.xml')
        if not os.path.exists(xml_file_path):
             raise FileNotFoundError(f"export.xml not found at: {xml_file_path}")
        parse_apple_health_xml(xml_file_path)

        # Organise the data from the clean CSVs into the main data structure.
        my_apple_health_data = organise_apple_health_data(apple_health_export_directory)
        print(f"\nSuccessfully organised {len(my_apple_health_data)} activities into the data structure.")

        # Feature engineering etc
        print("\nStarting feature engineering process...")
        for activity_id, activity_data in my_apple_health_data.items():
            if activity_data.get('track_data') is not None and not activity_data['track_data'].empty:
                print(f"  - Engineering features for Activity ID: {activity_id}")
                new_features = engineer_activity_features(activity_data['track_data'])
                activity_data['metadata'].update(new_features)
            else:
                print(f"  - No track data for Activity ID: {activity_id}. Setting defaults.")
                activity_data['metadata']['training_load_pace'] = 0
                activity_data['metadata']['pace_variability_std'] = 0
                activity_data['metadata']['training_load_hr'] = 0

        print("\nFeature engineering complete.")
        print("\nGenerating weekly and monthly summaries...")
        weekly_summary, monthly_summary = create_aggregate_summaries(my_apple_health_data)

        print("\n--- Enriched Activity Metadata Example ---")

        example_id = None
        for key in reversed(list(my_apple_health_data.keys())):
            if 'training_load_pace' in my_apple_health_data[key]['metadata']:
                example_id = key
                break
        
        if example_id:
            for key, val in my_apple_health_data[example_id]['metadata'].items():
                print(f"  {key}: {val}")
        else:
            print("  No enriched activities found to display as an example.")

        print("\n--- Weekly Summary ---")
        print(weekly_summary.tail())

        print("\n--- Monthly Summary ---")
        print(monthly_summary.tail())