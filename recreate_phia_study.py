#!/usr/bin/env python3
"""
Recreate PHIA Study Quantitative Results

This script serves as the main entry point for recreating the PHIA study's
quantitative results. It provides a framework for loading data, running models,
and generating outputs with configurable user IDs and question ranges.
"""

import argparse
import pandas as pd
import os
import sys
import importlib
import glob
import json
import google.generativeai as genai
from dotenv import load_dotenv
from onetwo import ot
from onetwo.backends import gemini_api
import data_utils
import colab_utils
import prompt_templates
import phia_agent

# Load environment variables
load_dotenv()

# Add current directory to sys.path
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Reload local modules
importlib.reload(data_utils)
importlib.reload(colab_utils)
importlib.reload(prompt_templates)
importlib.reload(phia_agent)

from data_utils import load_persona
from phia_agent import get_react_agent, QUESTION_PREFIX


def load_data(user_ids, question_range):
    """
    Load questions and answers for the specified user IDs from the corresponding CSV files.

    Args:
        user_ids (list): List of user IDs to process
        question_range (str): Range of questions to test (e.g., "1-100")

    Returns:
        list: List of dictionaries containing user_id, question, and correct answer
    """
    # Mapping of user IDs to CSV filenames
    user_files = {
        465: "health_behavior_final_v2.csv",
        171: "inactive_insomniacs_final_v2.csv",
        333: "sedentary_sleeper_final_v2.csv",
        41: "active_achiver_final_v2.csv"
    }

    # Parse question range
    start, end = map(int, question_range.split('-'))

    data = []
    for user_id in user_ids:
        if user_id not in user_files:
            raise ValueError(f"Unknown user ID: {user_id}")

        file_path = f"data/auto_eval/{user_files[user_id]}"
        df = pd.read_csv(file_path)

        # Select rows for the question range (1-based to 0-based indexing)
        selected_df = df.iloc[start-1:end]

        for _, row in selected_df.iterrows():
            data.append({
                'user_id': user_id,
                'question': row['question'],
                'answer': row['label']
            })

    return data


def run_model(data):
    """
    Run the Gemini 2.5 Flash Lite model on the loaded data.

    Args:
        data (list): List of dictionaries containing user_id, question, and answer

    Returns:
        list: List of dictionaries with original data, model answers, and reasoning steps
    """
    print(f"Running Gemini 2.5 Flash Lite model on loaded data with {len(data)} items...")

    # Set API Keys
    google_api_key = os.getenv("GOOGLE_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    genai.configure(api_key=google_api_key)

    # Model to use
    model_name = "models/gemini-2.5-flash-lite"

    # Load exemplars
    exemplar_dir = "few_shots"
    exemplar_pattern = os.path.join(exemplar_dir, "*.ipynb")
    final_exemplar_paths = glob.glob(exemplar_pattern)

    # Group data by user_id
    user_data = {}
    for item in data:
        user_id = item['user_id']
        if user_id not in user_data:
            user_data[user_id] = []
        user_data[user_id].append(item)

    results = []

    for user_id, questions in user_data.items():
        print(f"Processing user {user_id}")

        # Load user data
        summary_path = os.path.join("synthetic_wearable_users", f"summary_df_{user_id}.csv")
        activities_path = os.path.join("synthetic_wearable_users", f"exercise_df_{user_id}.csv")
        summary_df, activities_df, profile_df = load_persona(
            summary_path=summary_path,
            activities_path=activities_path,
            enforce_schema=True,
            temporally_localize="today"
        )

        # Setup LLM Backend
        llm_engine = gemini_api.GeminiAPI(
            generate_model_name=model_name,
            api_key=google_api_key,
            temperature=0.0,
        )
        llm_engine.register()

        # Create Agent
        agent = get_react_agent(
            summary_df=summary_df,
            activities_df=activities_df,
            profile_df=profile_df,
            example_files=final_exemplar_paths,
            tavily_api_key=tavily_api_key,
            use_mock_search=False
        )

        # Process each question for this user
        for item in questions:
            question = item['question']
            full_question = "Please provide only the direct answer to the following question, without any additional explanation, conversation, or introductory text." + QUESTION_PREFIX + question
            print(f"  - Running question: {question}")

            try:
                final_answer, final_state = ot.run(
                    agent(inputs=full_question, return_final_state=True)
                )
                results.append({
                    'user_id': user_id,
                    'question': question,
                    'correct_answer': item['answer'],
                    'model_answer': final_answer,
                    'reasoning_steps': str(final_state)
                })
            except Exception as e:
                print(f"    - Error running question: {e}")
                results.append({
                    'user_id': user_id,
                    'question': question,
                    'correct_answer': item['answer'],
                    'model_answer': f"Error: {e}",
                    'reasoning_steps': ""
                })

    return results


def generate_output(results):
    """
    Generate output from model results.

    Args:
        results (list): List of result dictionaries from run_model
    """
    print("Generating output from model results...")

    if not results:
        print("No results to generate output from.")
        return

    # Create output directory
    output_dir = "phia_recreation_output"
    os.makedirs(output_dir, exist_ok=True)

    # Create DataFrame with specified columns
    results_df = pd.DataFrame({
        'subject_number': [r['user_id'] for r in results],
        'question': [r['question'] for r in results],
        'correct_answer': [r['correct_answer'] for r in results],
        'calculated_result': [r['model_answer'] for r in results]
    })

    # Save CSV
    csv_path = os.path.join(output_dir, "phia_recreation_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Create reasoning file
    reasoning_data = [
        {
            'question': r['question'],
            'subject_number': r['user_id'],
            'reasoning_steps': r['reasoning_steps']
        } for r in results
    ]
    reasoning_path = os.path.join(output_dir, "phia_recreation_reasoning.json")
    with open(reasoning_path, 'w') as f:
        json.dump(reasoning_data, f, indent=2)
    print(f"Reasoning steps saved to {reasoning_path}")

    # Summary statistics
    total_questions = len(results)
    users = set(r['user_id'] for r in results)
    questions_per_user = {uid: sum(1 for r in results if r['user_id'] == uid) for uid in users}
    print(f"Total questions processed: {total_questions}")
    print(f"Users processed: {len(users)}")
    for uid in sorted(users):
        print(f"  User {uid}: {questions_per_user[uid]} questions")

    # Count errors
    error_count = sum(1 for r in results if r['model_answer'].startswith('Error:'))
    print(f"Errors encountered: {error_count}")


def main():
    """
    Main function that orchestrates the entire process.
    """
    parser = argparse.ArgumentParser(
        description="Recreate PHIA study quantitative results"
    )
    parser.add_argument(
        '--user_ids',
        nargs='+',
        type=int,
        required=True,
        help='List of user IDs to process (space-separated)'
    )
    parser.add_argument(
        '--question_range',
        type=str,
        required=True,
        help='Question range to test (e.g., "1-100")'
    )

    args = parser.parse_args()

    # Load data
    data = load_data(args.user_ids, args.question_range)

    # Run model
    results = run_model(data)

    # Generate output
    generate_output(results)


if __name__ == "__main__":
    main()