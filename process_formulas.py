import pandas as pd
import os
import dotenv
import numpy as np
from data_utils import load_persona, ChiaDataFrame

# Add query method to ChiaDataFrame to ensure it returns ChiaDataFrame
def chia_query(self, expr, **kwargs):
    result = super(ChiaDataFrame, self).query(expr, **kwargs)
    return ChiaDataFrame(result)

ChiaDataFrame.query = chia_query

dotenv.load_dotenv()

user_ids = [465, 171, 333, 41]

queries_df = pd.read_csv('data/auto_eval/improved_qualitative.csv', header=None, names=['Query', 'Formula'])

for user_id in user_ids:
    # Load data using load_persona
    summary_df, exercise_df, _ = load_persona(
        f'synthetic_wearable_users/summary_df_{user_id}.csv',
        f'synthetic_wearable_users/exercise_df_{user_id}.csv',
        enforce_schema=True,
        temporally_localize=False
    )

    # Ensure numeric columns are properly converted
    numeric_cols_summary = ['resting_heart_rate', 'heart_rate_variability', 'fatburn_active_zone_minutes',
                           'cardio_active_zone_minutes', 'peak_active_zone_minutes', 'active_zone_minutes',
                           'steps', 'rem_sleep_minutes', 'deep_sleep_minutes', 'awake_minutes',
                           'light_sleep_minutes', 'sleep_minutes', 'stress_management_score',
                           'deep_sleep_percent', 'rem_sleep_percent', 'awake_percent', 'light_sleep_percent']
    for col in numeric_cols_summary:
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce')

    numeric_cols_exercise = ['distance', 'duration', 'elevationGain', 'averageHeartRate',
                           'calories', 'steps', 'activeZoneMinutes', 'speed']
    for col in numeric_cols_exercise:
        if col in exercise_df.columns:
            exercise_df[col] = pd.to_numeric(exercise_df[col], errors='coerce')

    # Rename columns to match formula expectations
    summary_df['user_id'] = user_id
    exercise_df['user_id'] = user_id
    
    # Make dataframes available for eval
    daily_metrics = summary_df
    exercise_entries = exercise_df
    
    for idx, row in queries_df.iterrows():
        formula = row['Formula']
        # Fix incorrect .count() usage to get number of rows instead of per-column counts
        formula = formula.replace('.count()', '.shape[0]')
        try:
            result = eval(formula)
            # Handle pandas objects
            if isinstance(result, pd.Series):
                if len(result) == 1:
                    result = result.iloc[0]
                else:
                    result = result.tolist()
            elif isinstance(result, pd.DataFrame):
                result = result.to_dict('records')
        except Exception as e:
            result = np.nan
        queries_df.at[idx, f'answer_key_{user_id}'] = result

# Save the updated DataFrame
queries_df.to_csv('data/auto_eval/improved_qualitative_with_answers.csv', index=False)