import os
import pandas as pd
import numpy as np
from datetime import timedelta

USER_THRESHOLD_PACE = 5.5  # min/km
USER_MAX_HEART_RATE = 200   # bpm

PACE_ZONES = {
    'P1_Sprint': (0, USER_THRESHOLD_PACE * 0.8),
    'P2_Interval': (USER_THRESHOLD_PACE * 0.8, USER_THRESHOLD_PACE * 0.95),
    'P3_Tempo': (USER_THRESHOLD_PACE * 0.95, USER_THRESHOLD_PACE * 1.05),
    'P4_Aerobic': (USER_THRESHOLD_PACE * 1.05, USER_THRESHOLD_PACE * 1.25),
    'P5_Recovery': (USER_THRESHOLD_PACE * 1.25, 30), # Cap at 30 min/km
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

def load_samsung_heart_rate_csv(export_base_path):
    """Reads all heart rate data from the main Samsung Health CSV file."""
    hr_csv_path = None
    for file_name in os.listdir(export_base_path):
        if file_name.startswith('com.samsung.health.heart_rate') and file_name.endswith('.csv'):
            hr_csv_path = os.path.join(export_base_path, file_name)
            break
    
    if not hr_csv_path:
        print("  - Warning: Heart rate CSV file not found. Heart rate features will not be calculated.")
        return None

    print(f"Reading heart rate data from: {os.path.basename(hr_csv_path)}")
    try:
        hr_df = pd.read_csv(hr_csv_path, skiprows=1, index_col=False)
        hr_df['timestamp'] = pd.to_datetime(hr_df['end_time'], utc=True, dayfirst=True)
        hr_df.rename(columns={'heart_rate': 'heart_rate_bpm'}, inplace=True)
        return hr_df[['timestamp', 'heart_rate_bpm']].sort_values('timestamp').reset_index(drop=True)
    except Exception as e:
        print(f"  - Warning: Could not parse heart rate file {os.path.basename(hr_csv_path)}: {e}")
        return None

def organise_samsung_data(export_base_path):
    """
    Organises Samsung Health data by parsing the main exercise CSV and linking it
    with heart rate data from its corresponding CSV.
    """
    exercise_csv_path = None
    for file_name in os.listdir(export_base_path):
        print(f"Checking file: {file_name}")
        if file_name.startswith('com.samsung.health.exercise') and file_name.endswith('.csv'):
            exercise_csv_path = os.path.join(export_base_path, file_name)
            print(f"Found exercise CSV: {os.path.basename(exercise_csv_path)}")
            break
    
    if not exercise_csv_path:
        raise FileNotFoundError(f"Could not find 'com.samsung.health.exercise.*.csv' in '{export_base_path}'. Please check your export structure.")

    all_hr_df = load_samsung_heart_rate_csv(export_base_path)

    print(f"Reading master exercise list from: {os.path.basename(exercise_csv_path)}")
    master_df = pd.read_csv(exercise_csv_path, skiprows=1, index_col=False)
    master_df = master_df[master_df['exercise_type'] == 1002].copy() # 1002 is 'Running'
    
    samsung_data_structure = {}
    print(f"\nProcessing {len(master_df)} running activities...")

    for index, activity in master_df.iterrows():
        start_time = pd.to_datetime(activity['start_time'], utc=True, dayfirst=True)
        end_time = start_time + timedelta(seconds=activity['duration'])
        activity_id = start_time.strftime('%Y%m%d_%H%M%S')
        
        print(f"\nProcessing Activity ID: {activity_id} (Running)")

        activity_hr_df = None
        if all_hr_df is not None:
            activity_hr_df = all_hr_df[
                (all_hr_df['timestamp'] >= start_time) & (all_hr_df['timestamp'] <= end_time)
            ].copy()
            if activity_hr_df.empty:
                print("  - No heart rate data found for this activity's time window.")
                activity_hr_df = None
            else:
                print(f"  - Found {len(activity_hr_df)} heart rate records for this activity.")

        metadata = {
            'Activity ID': activity_id,
            'Activity Date': start_time,
            'Activity Type': 'Run',
            'Elapsed Time': activity['duration'], # Already in seconds
            'Distance': activity['distance'] / 1000, # Convert m to km
            'Calories': activity['calorie'],
            'Filename': None # No GPX file
        }

        samsung_data_structure[activity_id] = {
            'metadata': metadata,
            'track_data': None, # No GPX data
            'heart_rate_data': activity_hr_df # Store HR data here
        }

    print("\nProcessing complete.")
    return samsung_data_structure

def engineer_activity_features(hr_df):
    """Calculates features from heart rate data for a single activity."""
    if hr_df is None or hr_df.empty:
        return {}

    features = {}
    
    hr_df['time_diff_s'] = hr_df['timestamp'].diff().dt.total_seconds().fillna(0)

    if 'heart_rate_bpm' in hr_df.columns and hr_df['heart_rate_bpm'].notna().any():
        hr_bins = [v[0] for v in HEART_RATE_ZONES.values()] + [HEART_RATE_ZONES['Z5_Maximum'][1]]
        hr_df['hr_zone'] = pd.cut(hr_df['heart_rate_bpm'], bins=hr_bins, labels=HEART_RATE_ZONES.keys(), right=False)
        time_in_hr_zones = hr_df.groupby('hr_zone', observed=False)['time_diff_s'].sum().to_dict()
        features['time_in_hr_zones_s'] = {k: v for k, v in time_in_hr_zones.items() if pd.notna(k)}

        hr_trimp = sum((duration_s / 60) * HEART_RATE_ZONE_MULTIPLIERS.get(zone, 0) for zone, duration_s in features['time_in_hr_zones_s'].items())
        features['training_load_hr'] = round(hr_trimp, 2)
        features['avg_heart_rate_bpm'] = round(hr_df['heart_rate_bpm'].mean(), 1)
    
    features['training_load_pace'] = 0
    features['pace_variability_std'] = 0
    features['time_in_pace_zones_s'] = {}

    return features

def create_aggregate_summaries(data_structure):
    """Creates weekly and monthly summaries from the enriched metadata."""
    if not data_structure:
        return pd.DataFrame(), pd.DataFrame()

    metadata_list = [activity['metadata'] for activity in data_structure.values()]
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
    samsung_health_export_directory = './samsung-data' 

    if not os.path.isdir(samsung_health_export_directory):
        print("="*50)
        print(f"ERROR: The specified directory '{samsung_health_export_directory}' does not exist.")
        print("Please update the variable with the correct path to your unzipped Samsung Health data.")
        print("="*50)
    else:
        my_samsung_data = organise_samsung_data(samsung_health_export_directory)
        print(f"\nSuccessfully organised {len(my_samsung_data)} activities into the data structure.")

        print("\nStarting feature engineering process...")
        for activity_id, activity_data in my_samsung_data.items():
            hr_data = activity_data.get('heart_rate_data')
            if hr_data is not None and not hr_data.empty:
                print(f"  - Engineering features for Activity ID: {activity_id}")
                new_features = engineer_activity_features(hr_data)
                activity_data['metadata'].update(new_features)
            else:
                print(f"  - No detailed heart rate data for Activity ID: {activity_id}. Setting defaults.")
                activity_data['metadata']['training_load_pace'] = 0
                activity_data['metadata']['pace_variability_std'] = 0
                activity_data['metadata']['training_load_hr'] = 0

        print("\nFeature engineering complete.")
        print("\nGenerating weekly and monthly summaries...")
        weekly_summary, monthly_summary = create_aggregate_summaries(my_samsung_data)

        print("\n--- Enriched Activity Metadata Example ---")
        if my_samsung_data:
            example_id = None
            for act_id in reversed(list(my_samsung_data.keys())):
                if my_samsung_data[act_id]['metadata'].get('training_load_hr', 0) > 0:
                    example_id = act_id
                    break
            if example_id is None: 
                example_id = list(my_samsung_data.keys())[-1]

            for key, val in my_samsung_data[example_id]['metadata'].items():
                print(f"  {key}: {val}")
        else:
            print("  No activities found to display as an example.")

        print("\n--- Weekly Summary ---")
        print(weekly_summary.tail())

        print("\n--- Monthly Summary ---")
        print(monthly_summary.tail())