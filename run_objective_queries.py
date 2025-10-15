import os
import sys
import importlib
import glob
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from onetwo import ot
from onetwo.backends import gemini_api
import data_utils
import colab_utils
import prompt_templates
import phia_agent
import pprint

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

# Configuration
START_QUERY_INDEX = 3101  # Starting index of queries to run (0-based)
END_QUERY_INDEX = 3102    # Ending index of queries to run (exclusive, set to None to run to end)
USER_IDS = [465, 171, 333, 41]

# 465 - healthy behavior - data/auto_eval/health_behavior_final_v2.csv
# 171 - inactive insomniacs - data/auto_eval/inactive_insomniacs_final_v2.csv
# 333 - sedentary sleeper - data/auto_eval/sedentary_sleeper_final_v2.csv
# 41 - active archiver - data/auto_eval/active_achiver_final_v2.csv



def main():
    """
    Main function to run objective queries with different Gemini models.
    """
    # Set API Keys
    google_api_key = os.getenv("GOOGLE_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    genai.configure(api_key=google_api_key)

    # Models to test
    models_to_test = ["models/gemini-2.5-flash-lite"]
    # models_to_test = ["models/gemini-2.5-flash-lite","models/gemini-2.0-flash"]  # Temporarily test one model

    # Load objective queries
    objective_queries_df = pd.read_excel("Objective Query - PHIA.xlsx")

    # Load exemplars
    exemplar_dir = "few_shots"
    exemplar_pattern = os.path.join(exemplar_dir, "*.ipynb")
    final_exemplar_paths = glob.glob(exemplar_pattern)

    results = []

    # Calculate the range of queries to run
    end_index = END_QUERY_INDEX if END_QUERY_INDEX is not None else len(objective_queries_df)

    for user_id in USER_IDS:
        print(f"Processing user {user_id}")

        # Load data
        summary_path = os.path.join("synthetic_wearable_users", f"summary_df_{user_id}.csv")
        activities_path = os.path.join("synthetic_wearable_users", f"exercise_df_{user_id}.csv")
        summary_df, activities_df, profile_df = load_persona(
            summary_path=summary_path,
            activities_path=activities_path,
            enforce_schema=True,
            temporally_localize="today"
        )

        # print(f"Data loaded - Summary: {summary_df.shape}, Activities: {activities_df.shape}, Profile: {profile_df.shape}")
        print(f"Activities columns: {list(activities_df.columns)}")
        # print(f"Sample activities data:\n{activities_df.head(2)}")
        # print(f"Summary columns: {list(summary_df.columns)}")
        print(f"Sample summary data:\n{summary_df.head(2)}")

        for model_name in models_to_test:
            print(f"Running queries {START_QUERY_INDEX} to {end_index-1} for model: {model_name} and user: {user_id}")

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

            # Select queries in the specified range
            queries_to_run = objective_queries_df.iloc[START_QUERY_INDEX:end_index]
            for index, row in queries_to_run.iterrows():
                question = row["Question"]
                full_question = QUESTION_PREFIX + question
                print(f"  - Running question: {question}")
                # print(f"  - Full question: {full_question}")
                # print(f"  - QUESTION_PREFIX: {QUESTION_PREFIX}")

                try:
                    final_answer, final_state = ot.run(
                        agent(inputs=full_question, return_final_state=True)
                    )
                    # print(f"    - Final answer: {final_answer}")
                    # print(f"    - Final state type: {type(final_state)}")
                    print("Final state:")
                    pprint.pprint(final_state)
                    results.append({
                        "user_id": user_id,
                        "model": model_name,
                        "question": question,
                        "answer": final_answer,
                        "final_state": str(final_state)  # Convert to string for CSV
                    })
                except Exception as e:
                    print(f"    - Error running question: {e}")
                    results.append({
                        "user_id": user_id,
                        "model": model_name,
                        "question": question,
                        "answer": f"Error: {e}"
                    })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("objective_query_results.csv", index=False)
    print("Finished running all queries. Results saved to objective_query_results.csv")

if __name__ == "__main__":
    main()