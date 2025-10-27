#!/usr/bin/env python3
"""
Recreate PHIA Study Quantitative Results - Parallel Version

This script serves as the main entry point for recreating the PHIA study's
quantitative results using parallel processing to speed up data processing.
It provides a framework for loading data, running models in parallel,
and generating outputs with configurable user IDs, question ranges, and workers.
"""

import argparse
import pandas as pd
import os
import sys
import importlib
import glob
import json
import time
import re
import multiprocessing
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from onetwo import ot
from onetwo.backends import gemini_api
from onetwo.backends import openai_api
import data_utils
import colab_utils
import prompt_templates
import post_processor
import phia_agent

# Add current directory to sys.path
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Reload local modules
importlib.reload(data_utils)
importlib.reload(colab_utils)
importlib.reload(prompt_templates)
importlib.reload(phia_agent)

from post_processor import process_answer
from data_utils import load_persona
from phia_agent import get_react_agent, QUESTION_PREFIX


def extract_react_trace(final_state):
    """
    Extracts the detailed ReAct trace (THOUGHT, ACT, OBSERVATION steps) from the final_state.
    
    Args:
        final_state: The final state object returned by ot.run()
    
    Returns:
        str: Formatted string containing the agent's reasoning steps
    """
    if hasattr(final_state, 'messages') and final_state.messages:
        trace = []
        for i, msg in enumerate(final_state.messages):
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                role = msg.role
                content = msg.content
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    content += f"\nTool Calls: {msg.tool_calls}"
                trace.append(f"Step {i+1} ({role}): {content}")
        return '\n\n---\n\n'.join(trace)
    elif hasattr(final_state, 'trace'):
        # Alternative structure if trace is directly available
        return str(final_state.trace)
    else:
        # Fallback to string representation
        return str(final_state)


def load_and_prepare_data(user_ids, question_range):
    """
    Load questions and answers for the specified user IDs and prepare them for distribution to workers.

    Args:
        user_ids (list): List of user IDs to process
        question_range (str): Range of questions to test (e.g., "1-100")

    Returns:
        dict: Dictionary mapping user_id to list of question dictionaries
    """
    # Read the CSV file
    df = pd.read_csv('data/auto_eval/improved100_qualitative_with_answers.csv')

    # Parse question range
    start, end = map(int, question_range.split('-'))

    user_questions = {}
    for user_id in user_ids:
        answer_col = f'answer_key_{user_id}'
        if answer_col not in df.columns:
            raise ValueError(f"No answer column for user {user_id}")

        # Select rows for the question range (1-based to 0-based indexing)
        selected_df = df.iloc[start-1:end]

        questions = []
        for idx, row in enumerate(selected_df.iterrows()):
            questions.append({
                'question': row[1]['Query'],
                'answer': row[1][answer_col],
                'question_index': start + idx
            })

        user_questions[user_id] = questions

    return user_questions


def run_with_retry(agent, full_question, item, user_id):
    """
    Run the agent with retry logic for 429 errors in both exceptions and results.

    Args:
        agent: The agent to run
        full_question: The full question string
        item: Dict with question details
        user_id: The user ID

    Returns:
        dict: Result dictionary
    """
    max_retries = 5
    question = item['question']


    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                retry_prefix = f"(Retry attempt {attempt + 1} due to previous code error) "
            else:
                retry_prefix = ""
            full_question = retry_prefix + (
                f"Please provide only the direct answer to the following question. It is critically important to be concise and obey the following rules for your final answer or your response will be considered invalid:"
                f"1. Minimize your final response to numbers, dates, or just one to three words and do not provide additional explanation, conversation, or introductory text. "
                f"2. Never format your answer with markdown"
                f"3. Provide just a number if the answer is quantitative. "
                f"4. If the answer is zero, return '0' or '0.0'. "
                f"5. If the question is about an activity for which there is no data, assume the activity was performed zero times and answer 'NA', '0', or '0.0'. "
                f"6. Final answers must be extremely brief, preferably no longer than two words. "
            ) + QUESTION_PREFIX + question
            print(question)
            print(f"Attempt {attempt + 1}: Running agent for question {item['question_index']}")
            final_answer, final_state = ot.run(
                agent(inputs=full_question, return_final_state=True)
            )
            # Check for 429 error indicators in the result
            fa_str = str(final_answer)
            fs_str = str(final_state)
            if ("429" in fa_str or "429" in fs_str or
                "quota exceeded" in fa_str.lower() or "quota exceeded" in fs_str.lower()):
                if attempt < max_retries:
                    # Retry with delay
                    retry_delay = 15.0
                    sleep_time = retry_delay + 3 + attempt * 10
                    print(f"quota exceeded on query {item['question_index']}, retry in {sleep_time} seconds")
                    print(f"Retrying attempt {attempt + 1} for query {item['question_index']}")
                    time.sleep(sleep_time)
                    continue
                else:
                    print(f"Retry failed after {max_retries} retries for query {item['question_index']}")
                    return {
                        'user_id': user_id,
                        'question': question,
                        'correct_answer': item['answer'],
                        'model_answer': f"Error: 429 in result after {max_retries} retries",
                        'reasoning_steps': str(final_state) if 'final_state' in locals() else "",
                        'question_index': item['question_index']
                    }
            else:
                if attempt > 0:
                    print(f"Retry succeeded for query {item['question_index']}")
                try:
                    cleaned_answer = process_answer(question, final_answer)
                    if cleaned_answer == "RETRY_AGENT":
                        if attempt < max_retries:
                            print(f"Code error detected in agent answer, retrying agent for query {item['question_index']}")
                            continue  # Retry the agent query
                        else:
                            print(f"Agent retry failed after {max_retries} retries due to code errors for query {item['question_index']}")
                            return {
                                'user_id': user_id,
                                'question': question,
                                'correct_answer': item['answer'],
                                'model_answer': f"Error: Agent retry failed after {max_retries} retries due to code errors",
                                'reasoning_steps': str(final_state) if 'final_state' in locals() else "",
                                'question_index': item['question_index']
                            }
                except Exception as e:
                    print(f"Post-processing failed: {e}, using original answer")
                    cleaned_answer = final_answer
                reasoning_trace = extract_react_trace(final_state)
                return {
                    'user_id': user_id,
                    'question': question,
                    'correct_answer': item['answer'],
                    'model_answer': cleaned_answer,
                    'reasoning_steps': reasoning_trace,
                    'question_index': item['question_index']
                }
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota exceeded" in error_msg.lower():
                if attempt < max_retries:
                    # Extract retry delay from error message
                    match = re.search(r"retry in (\d+\.?\d*)s", error_msg)
                    if match:
                        retry_delay = float(match.group(1))
                    else:
                        match = re.search(r"retry_delay \{ seconds: (\d+) \}", error_msg)
                        if match:
                            retry_delay = float(match.group(1))
                        else:
                            retry_delay = 15.0
                    sleep_time = retry_delay + 3 + attempt * 10
                    print(f"quota exceeded on query {item['question_index']}, retry in {sleep_time} seconds")
                    print(f"Retrying attempt {attempt + 1} for query {item['question_index']}")
                    time.sleep(sleep_time)
                    continue
                else:
                    print(f"Retry failed after {max_retries} retries for query {item['question_index']}")
                    return {
                        'user_id': user_id,
                        'question': question,
                        'correct_answer': item['answer'],
                        'model_answer': f"Error after {max_retries} retries: {error_msg}",
                        'reasoning_steps': str(final_state) if 'final_state' in locals() else "",
                        'question_index': item['question_index']
                    }
            else:
                return {
                    'user_id': user_id,
                    'question': question,
                    'correct_answer': item['answer'],
                    'model_answer': f"Error: {e}",
                    'reasoning_steps': str(final_state) if 'final_state' in locals() else "",
                    'question_index': item['question_index']
                }

    # If we exit the loop without returning, it means max retries exceeded for RETRY_AGENT
    return {
        'user_id': user_id,
        'question': question,
        'correct_answer': item['answer'],
        'model_answer': f"Error: Agent retry failed after {max_retries} retries due to code errors",
        'reasoning_steps': str(final_state) if 'final_state' in locals() else "",
        'question_index': item['question_index']
    }


def run_worker_process(user_data):
    """
    Function executed by each parallel worker process.

    Args:
        user_data (tuple): Tuple of (user_id, questions_list)

    Returns:
        str: Path to the intermediate results file
    """
    user_id, questions = user_data

    # Define intermediate file path
    intermediate_path = f"temp_results_worker_{user_id}.csv"

    # Set API Keys
    google_api_key = os.getenv("GOOGLE_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    api_choice = os.getenv("API_CHOICE")
    gemini_model_name = os.getenv("GEMINI_MODEL_NAME")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model_name = os.getenv("OPENAI_MODEL_NAME")
    openai_api_base = os.getenv("OPENAI_BASE_URL")

    # Modify __post_init__ to use base_url
    original_post_init = openai_api.OpenAIAPI.__post_init__

    def patched_post_init(self):
        # Create cache first (from original __post_init__)
        self._cache_handler = openai_api.caching.SimpleFunctionCache(
            cache_filename=self.cache_filename,
        )

        # Create client with base_url if provided
        base_url = getattr(self, '_base_url', None)
        if base_url:
            self._client = openai_api.openai.OpenAI(api_key=self.api_key, base_url=base_url)
        elif self.api_key:
            self._client = openai_api.openai.OpenAI(api_key=self.api_key)
        else:
            self._client = openai_api.openai.OpenAI()

    # Replace __post_init__
    openai_api.OpenAIAPI.__post_init__ = patched_post_init

    if api_choice == 'gemini':
        genai.configure(api_key=google_api_key)
        model_name = gemini_model_name
        llm_engine = gemini_api.GeminiAPI(
            generate_model_name=model_name,
            api_key=google_api_key,
            temperature=0.0,
        )
    elif api_choice == 'openai':
        model_name = openai_model_name
        llm_engine = openai_api.OpenAIAPI(
            model_name=model_name,
            api_key=openai_api_key,
            temperature=0.0,
        )
        # Set base_url after instantiation
        llm_engine._base_url = openai_api_base
    else:
        raise ValueError(f"Unsupported API_CHOICE: {api_choice}")

    llm_engine.register()

    # Load exemplars
    exemplar_dir = "few_shots"
    exemplar_pattern = os.path.join(exemplar_dir, "*.ipynb")
    final_exemplar_paths = glob.glob(exemplar_pattern)

    # Load user data
    summary_path = os.path.join("synthetic_wearable_users", f"summary_df_{user_id}.csv")
    activities_path = os.path.join("synthetic_wearable_users", f"exercise_df_{user_id}.csv")
    summary_df, activities_df, profile_df = load_persona(
        summary_path=summary_path,
        activities_path=activities_path,
        enforce_schema=True,
        temporally_localize="today"
    )

    # Create Agent
    agent = get_react_agent(
        summary_df=summary_df,
        activities_df=activities_df,
        profile_df=profile_df,
        example_files=final_exemplar_paths,
        tavily_api_key=tavily_api_key,
        use_mock_search=False
    )

    results = []

    # Process each question for this user
    for i, item in enumerate(questions):
        question = item['question']
        full_question = (
            f"Please provide only the direct answer to the following question. It is critically important to be concise and obey the following rules for your final answer or your response will be considered invalid:"
            f"1. Minimize your final response to numbers, dates, or just one to three words and do not provide additional explanation, conversation, or introductory text. "
            f"2. Never format your answer with markdown"
            f"3. Provide just a number if the answer is quantitative. "
            f"4. If the answer is zero, return '0' or '0.0'. "
            f"5. If the question is about an activity for which there is no data, assume the activity was performed zero times and answer 'NA', '0', or '0.0'. "
            f"6. Final answers must be extremely brief, preferably no longer than two words. "
        ) + QUESTION_PREFIX + question

        results.append(run_with_retry(agent, full_question, item, user_id))

        # Periodic saving every 10 questions
        if (i + 1) % 10 == 0:
            # Save intermediate results
            results_df = pd.DataFrame({
                'subject_number': [r['user_id'] for r in results],
                'question': [r['question'] for r in results],
                'correct_answer': [r['correct_answer'] for r in results],
                'calculated_result': [r['model_answer'] for r in results],
                'reasoning_steps': [r['reasoning_steps'] for r in results],
                'question_index': [r['question_index'] for r in results]
            })
            results_df.to_csv(intermediate_path, index=False)

    # Save final results
    results_df = pd.DataFrame({
        'subject_number': [r['user_id'] for r in results],
        'question': [r['question'] for r in results],
        'correct_answer': [r['correct_answer'] for r in results],
        'calculated_result': [r['model_answer'] for r in results],
        'reasoning_steps': [r['reasoning_steps'] for r in results],
        'question_index': [r['question_index'] for r in results]
    })
    results_df.to_csv(intermediate_path, index=False)

    return intermediate_path


def aggregate_results(worker_file_paths, output_dir="phia_recreation_output", append=False):
    """
    Aggregate results from all worker processes by reading their intermediate files,
    combining into a single DataFrame, sorting, and generating final outputs.

    Args:
        worker_file_paths (list): List of file paths to intermediate result files

    Returns:
        pd.DataFrame: The final aggregated and sorted DataFrame
    """
    dfs = []
    for file_path in worker_file_paths:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            dfs.append(df)
        else:
            print(f"Warning: Intermediate file {file_path} not found.")

    if not dfs:
        return pd.DataFrame()

    # Combine all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)

    if append:
        csv_path = os.path.join(output_dir, "phia_recreation_parallel_results.csv")
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            combined_df = pd.concat([existing_df, combined_df], ignore_index=True)

    # Ensure required columns are present
    required_cols = ['question_index', 'subject_number', 'question', 'correct_answer', 'calculated_result', 'reasoning_steps']
    if not all(col in combined_df.columns for col in required_cols):
        raise ValueError(f"Missing required columns in intermediate files. Required: {required_cols}")

    # Reorder columns for final output
    final_columns = ['question_index', 'subject_number', 'question', 'correct_answer', 'calculated_result']
    combined_df = combined_df[final_columns + ['reasoning_steps']]  # Keep reasoning_steps for JSON

    # Sort by subject_number then question_index
    combined_df = combined_df.sort_values(['subject_number', 'question_index']).reset_index(drop=True)

    # Remove duplicates based on subject_number and question_index
    combined_df = combined_df.drop_duplicates(subset=['subject_number', 'question_index'], keep='first')

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save final CSV
    csv_path = os.path.join(output_dir, "phia_recreation_parallel_results.csv")
    combined_df[final_columns].to_csv(csv_path, index=False)
    print(f"Final results saved to {csv_path}")

    # Generate final JSON for reasoning steps
    reasoning_data = combined_df[['question_index', 'subject_number', 'question', 'reasoning_steps']].to_dict('records')
    json_path = os.path.join(output_dir, "phia_recreation_reasoning.json")
    if append and os.path.exists(json_path):
        with open(json_path, 'r') as f:
            existing_reasoning = json.load(f)
        reasoning_data = existing_reasoning + reasoning_data

    # Sort and dedupe reasoning data
    reasoning_df = pd.DataFrame(reasoning_data)
    reasoning_df = reasoning_df.sort_values(['subject_number', 'question_index']).reset_index(drop=True)
    reasoning_df = reasoning_df.drop_duplicates(subset=['subject_number', 'question_index'], keep='first')
    reasoning_data = reasoning_df.to_dict('records')

    with open(json_path, 'w') as f:
        json.dump(reasoning_data, f, indent=2)
    print(f"Reasoning steps saved to {json_path}")

    # Cleanup: Delete intermediate files
    for file_path in worker_file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted intermediate file: {file_path}")

    return combined_df


def normalize_boolean(answer):
    """
    Normalize boolean-like answers to canonical forms.
    Treats "None", "No", 0, "0", "0.0" as "no"
    Treats "Yes", 1, "1", "1.0" as "yes"
    Returns original string for other values.
    """
    answer = str(answer).strip().lower()
    if answer in ['none', 'no', '0', '0.0', 'false']:
        return 'no'
    elif answer in ['yes', '1', '1.0', 'true']:
        return 'yes'
    else:
        return str(answer).strip()


def generate_output(results_df):
    """
    Generate summary statistics from the aggregated results DataFrame.

    Args:
        results_df (pd.DataFrame): The final aggregated DataFrame
    """
    print("Generating summary statistics...")

    if results_df.empty:
        print("No results to generate output from.")
        return

    # Summary statistics
    total_questions = len(results_df)
    users = results_df['subject_number'].unique()
    questions_per_user = results_df.groupby('subject_number').size()
    print(f"Total questions processed: {total_questions}")
    print(f"Users processed: {len(users)}")
    for uid in sorted(users):
        print(f"  User {uid}: {questions_per_user[uid]} questions")

    # Count errors
    error_count = sum(1 for result in results_df['calculated_result'] if str(result).startswith('Error:'))
    print(f"Errors encountered: {error_count}")

    # Evaluate accuracy
    correct_count = 0
    valid_count = 0
    for _, row in results_df.iterrows():
        correct_answer = str(row['correct_answer']).strip()
        model_answer = str(row['calculated_result']).strip()
        if model_answer.startswith('Error:'):
            continue  # Skip errors
        # Special case: if final answer is zero and correct answer is missing or blank, consider correct
        if (model_answer in ['0', '0.0']) and correct_answer == '':
            correct_count += 1
            valid_count += 1
            continue
        valid_count += 1

        # Normalize boolean equivalents
        normalized_correct = normalize_boolean(correct_answer)
        normalized_model = normalize_boolean(model_answer)

        if normalized_correct == normalized_model:
            correct_count += 1
            continue

        # Try to parse as number if not boolean equivalents
        try:
            correct_num = float(normalized_correct)
            model_num = float(normalized_model)
            if abs(correct_num - model_num) <= 0.1:
                correct_count += 1
        except ValueError:
            # Treat as text
            if normalized_correct.lower() in normalized_model.lower():
                correct_count += 1

    if valid_count > 0:
        percent_correct = (correct_count / valid_count) * 100
        print(f"Accuracy: {correct_count}/{valid_count} ({percent_correct:.2f}%)")
    else:
        print("No valid answers to evaluate.")


def main():
    """
    Main function that orchestrates the parallel processing.
    """
    parser = argparse.ArgumentParser(
        description="Recreate PHIA study quantitative results with parallel processing"
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
    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='Number of parallel workers to use (default: number of CPU cores)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='phia_recreation_output',
        help='Output directory for results (default: phia_recreation_output)'
    )
    parser.add_argument(
        '--append',
        action='store_true',
        default=False,
        help='Append to existing results files instead of overwriting'
    )

    args = parser.parse_args()

    # Load and prepare data
    user_questions = load_and_prepare_data(args.user_ids, args.question_range)

    # Run parallel processing
    print(f"Running parallel processing with {args.num_workers} workers...")
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        worker_results = pool.map(run_worker_process, user_questions.items())

    # Aggregate results
    all_results = aggregate_results(worker_results, args.output_dir, args.append)

    # Generate output
    generate_output(all_results)


if __name__ == "__main__":
    main()