import os
import pandas as pd
import gpxpy
import numpy as np
import gpxpy.geo

USER_THRESHOLD_PACE = 5.5  

PACE_ZONES = {
    'P1_Sprint': (0, USER_THRESHOLD_PACE * 0.8),
    'P2_Interval': (USER_THRESHOLD_PACE * 0.8, USER_THRESHOLD_PACE * 0.95),
    'P3_Tempo': (USER_THRESHOLD_PACE * 0.95, USER_THRESHOLD_PACE * 1.05),
    'P4_Aerobic': (USER_THRESHOLD_PACE * 1.05, USER_THRESHOLD_PACE * 1.25),
    'P5_Recovery': (USER_THRESHOLD_PACE * 1.25, 30), # Cap at 30 min/km
}

PACE_ZONE_MULTIPLIERS = {'P1_Sprint': 5, 'P2_Interval': 4, 'P3_Tempo': 3, 'P4_Aerobic': 2, 'P5_Recovery': 1}

def process_gpx_file(file_path):
    """
    Reads, parses, and cleans a single GPX file.

    Args:
        file_path (str): The full path to the GPX file.

    Returns:
        pandas.DataFrame: A cleaned DataFrame with detailed track data,
                          or None if the file cannot be processed.
    """
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
                point_data = {
                    'timestamp': point.time,
                    'latitude': point.latitude,
                    'longitude': point.longitude,
                    'elevation_m': point.elevation,
                    'cadence_spm': None
                }
                # Cadence is often in extensions
                for ext in point.extensions:
                    for child in ext.getchildren():
                        if 'cad' in child.tag.lower():
                            point_data['cadence_spm'] = int(child.text)
                data.append(point_data)
    
    if not data: return None

    df = pd.DataFrame(data)

    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['cadence_spm'] = pd.to_numeric(df['cadence_spm'], errors='coerce').ffill().bfill()
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

def organise_strava_data(export_base_path):
    """
    Organises Strava export data by linking activities.csv with GPX files.

    Args:
        export_base_path (str): The path to the root of the unzipped Strava export folder.

    Returns:
        dict: A dictionary where keys are Activity IDs and values contain
              activity metadata and the corresponding track data DataFrame.
    """
    activities_csv_path = os.path.join(export_base_path, 'activities.csv')
    activities_folder_path = export_base_path

    if not os.path.exists(activities_csv_path):
        raise FileNotFoundError(f"activities.csv not found at: {activities_csv_path}")

    master_df = pd.read_csv(activities_csv_path)
    strava_data_structure = {}
    
    print(f"Found {len(master_df)} activities in activities.csv. Processing...")

    # Iterate through each activity in the CSV
    for index, activity in master_df.iterrows():
        activity_id = activity['Activity ID']
        gpx_filename = activity['Filename']
        
        print(f"\nProcessing Activity ID: {activity_id} ({activity['Activity Type']})")

        # Skip if there's no associated file
        if pd.isna(gpx_filename):
            print("  - No filename associated. Storing metadata only.")
            strava_data_structure[activity_id] = {
                'metadata': activity.to_dict(),
                'track_data': None
            }
            continue

        # Construct the full path to the GPX file
        gpx_full_path = os.path.join(activities_folder_path, gpx_filename)
        print(f"  - Expected GPX file path: {gpx_full_path}")

        if os.path.exists(gpx_full_path):
            print(f"  - Found file: {gpx_filename}. Reading and cleaning...")
            track_df = process_gpx_file(gpx_full_path)
            
            strava_data_structure[activity_id] = {
                'metadata': activity.to_dict(),
                'track_data': track_df
            }
        else:
            print(f"  - File not found: {gpx_filename}. Storing metadata only.")
            strava_data_structure[activity_id] = {
                'metadata': activity.to_dict(),
                'track_data': None
            }
            
    print("\nProcessing complete.")
    return strava_data_structure

def engineer_activity_features(track_df):
    """
    Calculates all new features for a single activity's track data.
    """
    if track_df is None or track_df.empty or 'time_diff_s' not in track_df.columns:
        return {}

    features = {}

    # 1. Calculate Time in Pace Zones
    pace_bins = [v[0] for v in PACE_ZONES.values()] + [PACE_ZONES['P5_Recovery'][1]]
    track_df['pace_zone'] = pd.cut(track_df['pace_min_per_km'], bins=pace_bins, labels=PACE_ZONES.keys(), right=False)
    
    time_in_pace_zones = track_df.groupby('pace_zone', observed=False)['time_diff_s'].sum().to_dict()
    features['time_in_pace_zones_s'] = {k: v for k, v in time_in_pace_zones.items() if pd.notna(k)}

    # 2. Calculate Pace-Based Training Load
    training_load = 0
    for zone, duration_s in features['time_in_pace_zones_s'].items():
        multiplier = PACE_ZONE_MULTIPLIERS.get(zone, 0)
        training_load += (duration_s / 60) * multiplier # Duration in minutes * multiplier
    features['training_load'] = round(training_load, 2)
    
    # 3. Calculate Pace Variability
    features['pace_variability_std'] = round(track_df['pace_min_per_km'].std(), 2)

    # 4. Calculate Cadence Metrics (if data exists)
    if 'cadence_spm' in track_df.columns and track_df['cadence_spm'].notna().any():
        moving_df = track_df[track_df['pace_min_per_km'] < PACE_ZONES['P5_Recovery'][1]]
        if not moving_df.empty:
            # Cadence is often logged per foot, so multiply by 2 for steps per minute
            features['avg_cadence_spm'] = round(moving_df['cadence_spm'].mean() * 2, 1)

    return features

def create_aggregate_summaries(strava_data_structure):
    """
    Creates weekly and monthly summaries from the enriched metadata.
    This version is hardened to handle missing columns gracefully.
    """
    if not strava_data_structure:
        return pd.DataFrame(), pd.DataFrame()

    metadata_list = [activity['metadata'] for activity in strava_data_structure.values()]
    summary_df = pd.DataFrame(metadata_list)
    
    summary_df['Activity Date'] = pd.to_datetime(summary_df['Activity Date'])
    summary_df.set_index('Activity Date', inplace=True)
    
    agg_ops = {'Activity ID': 'count'}
    
    potential_cols = {
        'Distance': 'sum',
        'Elapsed Time': 'sum',
        'Elevation Gain': 'sum',
        'training_load': 'sum'
    }
    
    for col, op in potential_cols.items():
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce').fillna(0)
            agg_ops[col] = op
            
    weekly_summary = summary_df.resample('W-Mon', label='left', closed='left').agg(agg_ops).rename(columns={'Activity ID': 'Activity Count'})
    monthly_summary = summary_df.resample('ME').agg(agg_ops).rename(columns={'Activity ID': 'Activity Count'})
    
    if 'Elapsed Time' in weekly_summary.columns:
        weekly_summary['Duration_hours'] = weekly_summary['Elapsed Time'] / 3600
        weekly_summary = weekly_summary.drop(columns='Elapsed Time')
    
    if 'Elapsed Time' in monthly_summary.columns:
        monthly_summary['Duration_hours'] = monthly_summary['Elapsed Time'] / 3600
        monthly_summary = monthly_summary.drop(columns='Elapsed Time')
        
    return weekly_summary, monthly_summary


if __name__ == '__main__':
    strava_export_directory = './strava/strava-data'  # I mean technically

    if not os.path.isdir(strava_export_directory):
        print("="*50)
        print(f"ERROR: The specified directory '{strava_export_directory}' does not exist.")
        print("Please create it or update the 'strava_export_directory' variable")
        print("with the correct path to your unzipped Strava data.")
        print("="*50)
    else:
        my_strava_data = organise_strava_data(strava_export_directory)
        print(f"\nSuccessfully organised {len(my_strava_data)} activities into the data structure.")

        print("Starting feature engineering process...")
        for activity_id, activity_data in my_strava_data.items():
            if activity_data.get('track_data') is not None and not activity_data['track_data'].empty:
                print(f"  - Engineering features for Activity ID: {activity_id}")
                new_features = engineer_activity_features(activity_data['track_data'])
                activity_data['metadata'].update(new_features)
            else:
                print(f"  - No track data for Activity ID: {activity_id}. Setting defaults.")
                activity_data['metadata']['training_load'] = 0
                activity_data['metadata']['pace_variability_std'] = 0

        print("\nFeature engineering complete.")
        print("\nGenerating weekly and monthly summaries...")
        weekly_summary, monthly_summary = create_aggregate_summaries(my_strava_data)

        print("\n--- Enriched Activity Metadata Example ---")
        example_id = list(my_strava_data.keys())[-1]
        for key, val in my_strava_data[example_id]['metadata'].items():
            print(f"  {key}: {val}")

        print("\n--- Weekly Summary ---")
        print(weekly_summary.tail())

        print("\n--- Monthly Summary ---")
        print(monthly_summary.tail())