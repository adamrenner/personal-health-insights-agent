import argparse
import os
import json
import glob
import pandas as pd
import numpy as np
import random

# List of realistic male names
MALE_NAMES = [
    "John Smith", "Michael Johnson", "David Williams", "James Brown", "Robert Jones", "William Davis", "Christopher Miller", "Daniel Wilson", "Matthew Moore", "Joseph Taylor",
    "Andrew Anderson", "Edward Thomas", "Joshua Jackson", "Anthony White", "Charles Harris", "Ryan Martin", "Nicholas Thompson", "Alexander Garcia", "Tyler Martinez", "Jacob Robinson",
    "Ethan Clark", "Benjamin Rodriguez", "Samuel Lewis", "Jonathan Lee", "Logan Walker", "Lucas Hall", "Jack Allen", "Henry Young", "Owen Hernandez", "Aiden King",
    "Mason Wright", "Liam Lopez", "Noah Hill", "William Green", "James Adams", "Benjamin Baker", "Lucas Gonzalez", "Henry Nelson", "Alexander Carter", "Mason Mitchell",
    "Ethan Perez", "Daniel Roberts", "Matthew Turner", "Aiden Phillips", "Jackson Campbell", "David Parker", "Joseph Evans", "Carter Edwards", "Owen Collins", "Wyatt Stewart",
    "Jack Sanchez", "Luke Morris", "Grayson Rogers", "Julian Reed", "Levi Cook", "Hunter Morgan", "Jayden Bell", "Mason Murphy", "Lincoln Bailey", "Asher Rivera"
]

# List of realistic female names
FEMALE_NAMES = [
    "Mary Johnson", "Linda Williams", "Patricia Brown", "Susan Jones", "Deborah Davis", "Barbara Miller", "Debra Wilson", "Karen Moore", "Nancy Taylor", "Donna Anderson",
    "Cynthia Thomas", "Sandra Jackson", "Lisa White", "Betty Harris", "Helen Martin", "Sharon Thompson", "Dorothy Garcia", "Carol Martinez", "Ruth Robinson", "Sharon Clark",
    "Michelle Rodriguez", "Laura Lewis", "Sarah Lee", "Kimberly Walker", "Deborah Hall", "Jessica Allen", "Shirley Young", "Angela Hernandez", "Melissa King", "Rebecca Wright",
    "Virginia Lopez", "Kathleen Hill", "Amy Green", "Anna Adams", "Margaret Baker", "Dorothy Gonzalez", "Lisa Nelson", "Nancy Carter", "Betty Mitchell", "Helen Perez",
    "Sandra Roberts", "Donna Turner", "Carol Phillips", "Ruth Campbell", "Sharon Parker", "Michelle Evans", "Laura Edwards", "Sarah Collins", "Kimberly Stewart", "Deborah Sanchez",
    "Jessica Morris", "Shirley Rogers", "Angela Reed", "Melissa Cook", "Rebecca Morgan", "Virginia Bell", "Kathleen Murphy", "Amy Bailey", "Anna Rivera", "Margaret Peterson"
]

def generate_hybrid_name(gender):
    """Generates a unique name by combining parts of existing funny names."""
    pool = MALE_NAMES
    if gender and isinstance(gender, str) and gender.lower().strip() == 'female':
        pool = FEMALE_NAMES
    
    first_names = [n.split()[0] for n in pool if " " in n]
    last_names = [n.split()[-1] for n in pool if " " in n]
    
    # Avoid generating a name that already exists in the base list
    while True:
        new_name = f"{random.choice(first_names)} {random.choice(last_names)}"
        if new_name not in pool:
             return new_name

def get_funny_name(gender, used_names):
    """
    Generates a unique funny name based on gender using predefined lists.
    Fallback to hybrid names if lists are exhausted.
    """
    # Create a neutral pool as fallback
    neutral_names = list(set(MALE_NAMES + FEMALE_NAMES)) # remove duplicates
    
    available_names = []
    
    # Determine candidate pool based on gender
    if gender and isinstance(gender, str):
        if gender.lower().strip() == 'female':
            available_names = [n for n in FEMALE_NAMES if n not in used_names]
            # Fallback to male names if female specific ones run out
            if not available_names:
                 available_names = [n for n in MALE_NAMES if n not in used_names]
        elif gender.lower().strip() == 'male':
            available_names = [n for n in MALE_NAMES if n not in used_names]
            # Fallback to female names (some are unisex enough) if male run out
            if not available_names:
                 available_names = [n for n in FEMALE_NAMES if n not in used_names]
        else:
             available_names = [n for n in neutral_names if n not in used_names]
    else:
        available_names = [n for n in neutral_names if n not in used_names]
    
    # Select a name
    if available_names:
        return random.choice(available_names)
    
    # If we run out of unique names from the list, generate a hybrid name
    while True:
        hybrid_name = generate_hybrid_name(gender)
        if hybrid_name not in used_names:
            return hybrid_name

def process_data(input_path, output_path):
    """
    Reads CSV data, transforms it into JSON with funny names, and saves it.
    """
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Identify unique IDs from summary filenames
    # Pattern: summary_df_{ID}.csv
    summary_files = glob.glob(os.path.join(input_path, "summary_df_*.csv"))
    
    if not summary_files:
        print(f"No summary files found in {input_path}")
        return

    mapping_data = []
    used_names = set()
    
    print(f"Found {len(summary_files)} summary files. Starting processing...")

    for summary_file in summary_files:
        filename = os.path.basename(summary_file)
        
        # Extract ID
        try:
            # removing prefix and suffix
            subject_id = filename.replace("summary_df_", "").replace(".csv", "")
        except ValueError:
            print(f"Skipping file with unexpected name format: {filename}")
            continue
            
        # Read summary dataframe
        try:
            summary_df = pd.read_csv(summary_file)
        except Exception as e:
            print(f"Error reading {summary_file}: {e}")
            continue
            
        # Extract gender for name generation (assuming 'gender' column exists)
        gender = None
        if 'gender' in summary_df.columns and not summary_df['gender'].empty:
            gender = summary_df['gender'].iloc[0]
            
        # Generate full name
        name = get_funny_name(gender, used_names)
        used_names.add(name)
        
        # Prepare Exercise Data
        exercise_file = os.path.join(input_path, f"exercise_df_{subject_id}.csv")
        exercise_data = []
        
        if os.path.exists(exercise_file):
            try:
                exercise_df = pd.read_csv(exercise_file)
                # Rename pseudo_id2 to subject_id
                if 'pseudo_id2' in exercise_df.columns:
                    exercise_df.rename(columns={'pseudo_id2': 'subject_id'}, inplace=True)
                # Replace NaN with None for JSON validity
                exercise_data = exercise_df.where(pd.notnull(exercise_df), None).to_dict(orient='records')
            except Exception as e:
                print(f"Error reading exercise file {exercise_file}: {e}")
        else:
            # Handle missing exercise file gracefully (empty list)
            exercise_data = []
        
        # Process Summary Data
        # Rename pseudo_id2 to subject_id
        if 'pseudo_id2' in summary_df.columns:
            summary_df.rename(columns={'pseudo_id2': 'subject_id'}, inplace=True)
        # Replace NaN with None for JSON validity
        summary_data = summary_df.where(pd.notnull(summary_df), None).to_dict(orient='records')
        
        # Construct the final dictionary
        subject_record = {
            "subject_id": subject_id,
            "name": name,
            "gender": gender,
            "summary_data": summary_data,
            "exercise_data": exercise_data
        }
        
        # Define output filename
        output_filename = f"subject_{subject_id}.json"
        output_file_path = os.path.join(output_path, output_filename)
        
        # Write to JSON file
        try:
            with open(output_file_path, 'w') as f:
                # default=str handles serialization of dates and other non-standard types
                json.dump(subject_record, f, indent=4, default=str)
        except Exception as e:
            print(f"Error writing output file {output_file_path}: {e}")
            continue
            
        # Add to mapping list
        mapping_data.append({
            "name": name,
            "subject_id": subject_id,
            "file": output_filename
        })
        
        # print(f"Processed Subject ID: {subject_id} -> Name: {name}")

    # Create subject_mapping.json
    mapping_file_path = os.path.join(output_path, "subject_mapping.json")
    try:
        with open(mapping_file_path, 'w') as f:
            json.dump(mapping_data, f, indent=4)
        print(f"Processing complete. Mapping file created at {mapping_file_path}")
    except Exception as e:
        print(f"Error creating mapping file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reformat synthetic wearable data into JSON with funny names.")
    parser.add_argument("input_path", help="Path to the directory containing synthetic data CSVs.")
    parser.add_argument("output_path", help="Path to the directory where output JSON files will be stored.")
    
    args = parser.parse_args()
    
    process_data(args.input_path, args.output_path)