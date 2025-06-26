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
            
        hr_df['timestamp'] = pd.to_datetime(hr_df['end_time'], utc=True)
        hr_df.rename(columns={'heart_rate': 'heart_rate_bpm'}, inplace=True)

        return hr_df[['timestamp', 'heart_rate_bpm']].sort_values('timestamp').reset_index(drop=True)
    except Exception as e:
        print(f"  - Warning: Could not parse heart rate file {os.path.basename(hr_csv_path)}: {e}")
        return None

def organise_samsung_data(export_base_path):
    """
    Organises Samsung Health data by parsing the main exercise CSV and extracting
    activity data, including average and max heart rate.
    """
    exercise_csv_path = None
    for file_name in os.listdir(export_base_path):
        if file_name.startswith('com.samsung.health.exercise') and file_name.endswith('.csv'):
            exercise_csv_path = os.path.join(export_base_path, file_name)
            break
    
    if not exercise_csv_path:
        raise FileNotFoundError(f"Could not find 'com.samsung.health.exercise.*.csv' in '{export_base_path}'. Please check your export structure.")

    print(f"Reading master exercise list from: {os.path.basename(exercise_csv_path)}")
    master_df = pd.read_csv(exercise_csv_path, skiprows=1, index_col=False)
    master_df = master_df[master_df['exercise_type'] == 1002].copy() # 1002 is 'Running'
    
    samsung_data_structure = {}
    print(f"\nProcessing {len(master_df)} running activities...")

    for index, activity in master_df.iterrows():
        start_time = pd.to_datetime(activity['start_time'], utc=True, dayfirst=True)
        activity_id = start_time.strftime('%Y%m%d_%H%M%S')
        
        print(f"Processing Activity ID: {activity_id} (Running)")

        metadata = {
            'Activity ID': activity_id,
            'Activity Date': start_time,
            'Activity Type': 'Run',
            'Elapsed Time': activity['duration'] / 1000, # Data is in milliseconds
            'Distance': activity['distance'] / 1000, # Convert m to km
            'Calories': activity['calorie'],
            'Filename': None # No GPX file
        }

        if 'mean_heart_rate' in activity and pd.notna(activity['mean_heart_rate']):
            metadata['avg_heart_rate_bpm'] = round(activity['mean_heart_rate'])
        
        if 'max_heart_rate' in activity and pd.notna(activity['max_heart_rate']):
            metadata['max_heart_rate_bpm'] = round(activity['max_heart_rate'])

        samsung_data_structure[activity_id] = {
            'metadata': metadata,
            'track_data': None, # No GPX data
            'heart_rate_data': None # No longer storing detailed HR data
        }

    print("\nProcessing complete.")
    return samsung_data_structure

def engineer_activity_features(metadata):
    """
    Calculates training load and other features from an activity's metadata,
    using average heart rate.
    """
    features = {}
    duration_min = metadata.get('Elapsed Time', 0) / 60
    avg_hr = metadata.get('avg_heart_rate_bpm')

    # Calculate Training Load (TRIMP) based on average heart rate
    if avg_hr and duration_min > 0:
        hr_zone = None
        for zone, (lower, upper) in HEART_RATE_ZONES.items():
            if lower <= avg_hr < upper:
                hr_zone = zone
                break
        
        if hr_zone:
            hr_trimp = duration_min * HEART_RATE_ZONE_MULTIPLIERS.get(hr_zone, 0)
            features['training_load_hr'] = round(hr_trimp, 2)
            features['avg_hr_zone'] = hr_zone
        else:
            features['training_load_hr'] = 0
            features['avg_hr_zone'] = None
    else:
        features['training_load_hr'] = 0
        features['avg_hr_zone'] = None

    # Add avg_heart_rate_bpm to features if not already there, for consistency
    if 'avg_heart_rate_bpm' not in features and avg_hr:
        features['avg_heart_rate_bpm'] = avg_hr

    # Other features are not applicable without detailed track data
    features['training_load_pace'] = 0
    features['pace_variability_std'] = 0
    features['time_in_pace_zones_s'] = {}
    features['time_in_hr_zones_s'] = {} # Cannot be calculated from average

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
    samsung_health_export_directory = './samsung_data' 

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
            print(f"  - Engineering features for Activity ID: {activity_id}")
            new_features = engineer_activity_features(activity_data['metadata'])
            activity_data['metadata'].update(new_features)

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