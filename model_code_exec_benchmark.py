#!/usr/bin/env python3
"""
Benchmark Script for LLM Code Execution on PHIA Qualitative QA Tasks

This script benchmarks alternative LLM approaches using internal code execution
against the existing phia_agent for qualitative QA tasks. It simulates user
interactions by providing structured exercise and summary data along with specific
questions, enabling code-assisted analysis to evaluate accuracy improvements.
"""

import argparse
import pandas as pd
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import time
import re

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import model-specific clients (handle conditionally)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI client not available. Install with 'pip install openai'.")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Gemini client not available. Install with 'pip install google-generativeai'.")

try:
    from xai_sdk import Client as XAIClient
    from xai_sdk.chat import user
    from xai_sdk.tools import code_execution
    GROK_AVAILABLE = True
except ImportError:
    GROK_AVAILABLE = False
    logger.warning("Grok client not available. Install with 'pip install xai-sdk'.")

# Post-processor import
try:
    from post_processor import process_answer
    POST_PROCESSOR_AVAILABLE = True
except ImportError:
    POST_PROCESSOR_AVAILABLE = False
    logger.warning("Post-processor not available. Will use raw responses.")
    def process_answer(question: str, answer: str) -> str:
        return str(answer).strip()

def parse_question_range(range_str: str) -> List[int]:
    """
    Parse question range string into a list of 1-based indices.
    
    Args:
        range_str (str): Range like "1-10" or "5,10-15,20"
    
    Returns:
        List[int]: Sorted list of question indices
    """
    indices = []
    parts = range_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    return sorted(list(set(indices)))  # Remove duplicates and sort

def data_load(user_id: int, format_type: str = "markdown") -> str:
    """
    Load and format user-specific summary and exercise data.
    
    Args:
        user_id (int): User ID (465, 333, 171, 41)
        format_type (str): Format for data ("markdown", "csv", "json")
    
    Returns:
        str: Formatted data string
    """
    summary_path = f"synthetic_wearable_users/summary_df_{user_id}.csv"
    exercise_path = f"synthetic_wearable_users/exercise_df_{user_id}.csv"
    
    if not os.path.exists(summary_path) or not os.path.exists(exercise_path):
        raise FileNotFoundError(f"Data files for user {user_id} not found.")
    
    summary_df = pd.read_csv(summary_path)
    exercise_df = pd.read_csv(exercise_path)
    
    # Format data
    if format_type == "markdown":
        summary_formatted = summary_df.to_markdown(index=False)
        exercise_formatted = exercise_df.to_markdown(index=False)
    elif format_type == "csv":
        summary_formatted = summary_df.to_csv(index=False)
        exercise_formatted = exercise_df.to_csv(index=False)
    elif format_type == "json":
        summary_formatted = summary_df.to_json(orient='records')
        exercise_formatted = exercise_df.to_json(orient='records')
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    data_block = f"""
User Summary Data:
{summary_formatted}

User Exercise Data:
{exercise_formatted}
"""
    return data_block

def prompt_build(user_data: str, question: str, model_type: str) -> str:
    """
    Build the prompt for the LLM, adapted for code execution per model docs.
    
    Args:
        user_data (str): Formatted user data
        question (str): Specific question
        model_type (str): Model provider ("openai", "gemini", "grok")
    
    Returns:
        str: Complete prompt
    """
    base_instructions = """
Analyze the provided data using Python code execution to derive insights and answer the question step-by-step. 
Use code to manipulate data, compute statistics, or visualize patterns as needed. 
Output your reasoning, any code used, execution results, and final answer clearly.
"""
    
    if model_type == "openai":
        # Adapt for OpenAI code interpreter (via tools in chat completions or assistants)
        instructions = base_instructions + "\nUse the code interpreter tool to run Python code."
    elif model_type == "gemini":
        # For Gemini, enable code execution tool
        instructions = base_instructions + "\nGenerate and execute Python code using the code execution tool."
    elif model_type == "grok":
        # For Grok, use code_execution tool
        instructions = base_instructions + "\nWrite and execute Python code using the code execution tool."
    else:
        raise ValueError(f"Unsupported model: {model_type}")
    
    prompt = f"{instructions}\n\n{user_data}\n\nQuestion: {question}\n\nFinal Answer:"
    return prompt

def api_call_openai(prompt: str, max_retries: int = 3) -> Optional[str]:
    """
    Make API call to OpenAI with code execution enabled (using Assistants API simulation via tools).
    
    Args:
        prompt (str): The prompt
        max_retries (int): Number of retries for rate limits
    
    Returns:
        str: Full response text or None on error
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI not available.")
    
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in .env")
    
    client = openai.OpenAI(api_key=api_key)
    
    # For code execution, use chat completions with tools (simulate interpreter)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "code_interpreter",
                "description": "Run Python code in a sandboxed environment",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to execute"}
                    },
                    "required": ["code"]
                }
            }
        }
    ]
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                tools=tools,
                tool_choice="auto",
                temperature=0.0
            )
            full_response = response.choices[0].message.content
            if full_response:
                return full_response
        except openai.RateLimitError as e:
            wait_time = 2 ** attempt + 1  # Exponential backoff
            logger.warning(f"Rate limit hit, retrying in {wait_time}s: {e}")
            time.sleep(wait_time)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(1)
    return None

def api_call_gemini(prompt: str, max_retries: int = 3) -> Optional[str]:
    """
    Make API call to Gemini with code execution enabled.
    
    Args:
        prompt (str): The prompt
        max_retries (int): Number of retries for rate limits
    
    Returns:
        str: Full response text or None on error
    """
    if not GEMINI_AVAILABLE:
        raise ImportError("Gemini not available.")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    model = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set in .env")
    
    genai.configure(api_key=api_key)
    model_obj = genai.GenerativeModel(model)
    
    # Enable code execution tool
    from google.genai import types
    tools = [types.Tool(code_execution=types.ToolCodeExecution)]
    
    for attempt in range(max_retries):
        try:
            response = model_obj.generate_content(
                prompt,
                tools=tools,
                generation_config=types.GenerationConfig(temperature=0.0)
            )
            full_response = ""
            for part in response.candidates[0].content.parts:
                if part.text:
                    full_response += part.text
                if hasattr(part, 'executable_code') and part.executable_code:
                    full_response += f"\nCode: {part.executable_code.code}"
                if hasattr(part, 'code_execution_result') and part.code_execution_result:
                    full_response += f"\nExecution Result: {part.code_execution_result.output}"
            if full_response:
                return full_response
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(2 ** attempt)
    return None

def api_call_grok(prompt: str, max_retries: int = 3) -> Optional[str]:
    """
    Make API call to Grok with code execution enabled.
    
    Args:
        prompt (str): The prompt
        max_retries (int): Number of retries for rate limits
    
    Returns:
        str: Full response text or None on error
    """
    if not GROK_AVAILABLE:
        raise ImportError("Grok not available.")
    
    api_key = os.getenv("XAI_API_KEY")  # Assuming XAI_API_KEY in .env
    model = os.getenv("XAI_MODEL_NAME", "grok-4-fast")
    if not api_key:
        raise ValueError("XAI_API_KEY not set in .env")
    
    from xai_sdk import Client
    from xai_sdk.chat import user
    from xai_sdk.tools import code_execution

    client = Client(api_key=api_key)
    chat = client.chat.create(
        model=model,
        tools=[code_execution()],
    )
    
    for attempt in range(max_retries):
        try:
            chat.append(user(prompt))
            
            full_response = ""
            is_thinking = True
            for response, chunk in chat.stream():
                # Log tool calls
                for tool_call in chunk.tool_calls:
                    # logger.info(f"Tool call: {tool_call.function.name} with args: {tool_call.function.arguments}")
                    logger.info(f"Tool call: {tool_call.function.name}")
                if response.usage.reasoning_tokens and is_thinking:
                    pass
                    # logger.info(f"Thinking... ({response.usage.reasoning_tokens} tokens)")
                if chunk.content and is_thinking:
                    logger.info("Final Response started")
                    is_thinking = False
                if chunk.content and not is_thinking:
                    full_response += chunk.content
            
            # Optionally append citations or usage if needed
            if response.citations:
                full_response += f"\nCitations: {response.citations}"
            
            if full_response:
                return full_response.strip()
        except Exception as e:
            logger.error(f"Grok API error: {e}", exc_info=True)
            if attempt == max_retries - 1:
                return None
            time.sleep(2 ** attempt)
    return None

def api_call(prompt: str, model_type: str, max_retries: int = 3) -> Optional[str]:
    """
    Generic API call dispatcher based on model type.
    
    Args:
        prompt (str): The prompt
        model_type (str): "openai", "gemini", "grok"
        max_retries (int): Retries
    
    Returns:
        Optional[str]: Response or None
    """
    if model_type == "openai":
        return api_call_openai(prompt, max_retries)
    elif model_type == "gemini":
        return api_call_gemini(prompt, max_retries)
    elif model_type == "grok":
        return api_call_grok(prompt, max_retries)
    else:
        raise ValueError(f"Unsupported model: {model_type}")

def post_process(question: str, response: str) -> str:
    """
    Post-process the LLM response to extract concise answer.
    
    Args:
        question (str): Original question
        response (str): Full response
    
    Returns:
        str: Processed answer
    """
    if POST_PROCESSOR_AVAILABLE:
        return process_answer(question, response)
    else:
        # Simple extraction: look for "Final Answer:" or last sentence
        if "Final Answer:" in response:
            return response.split("Final Answer:")[-1].strip().split('\n')[0].strip()
        return response.strip()[-100:].strip()  # Last 100 chars as fallback

def evaluate_accuracy(pred: str, ground_truth: str) -> Dict[str, Any]:
    """
    Evaluate qualitative match (simple normalized string comparison).
    
    Args:
        pred (str): Predicted answer
        ground_truth (str): Ground truth
    
    Returns:
        Dict: Accuracy score (1 if match, 0 else), flag
    """
    pred_norm = re.sub(r'[^\w\s]', '', pred.lower().strip())
    gt_norm = re.sub(r'[^\w\s]', '', str(ground_truth).lower().strip())
    match = 1 if pred_norm == gt_norm else 0
    return {"accuracy_score": match, "match_flag": "match" if match else "discrepancy"}

def save_results(results: List[Dict], model: str, user: int, format_type: str):
    """
    Save benchmark results to CSV and full responses to JSONL in subfolder.
    
    Args:
        results (List[Dict]): List of result dicts
        model (str): Model name
        user (int): User ID
        format_type (str): Data format
    """
    subfolder = "model_code_exec_benchmark_results"
    os.makedirs(subfolder, exist_ok=True)
    
    # Final CSV with specified columns
    csv_df = pd.DataFrame([
        {
            "user_number": r["user"],
            "question_number": r["question_id"],
            "question": r.get("question", ""),
            "correct_answer": r["ground_truth"],
            "agent_final_answer": r["post_processed_answer"]
        }
        for r in results
    ])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(subfolder, f"final_benchmark_results_{model}_{user}_{format_type}_{timestamp}.csv")
    csv_df.to_csv(csv_filename, index=False)
    logger.info(f"Final results saved to {csv_filename}")
    
    # # Detailed CSV (legacy format with raw_prompt etc.)
    # detailed_df = pd.DataFrame([
    #     {
    #         "user": r["user"],
    #         "question_id": r["question_id"],
    #         "raw_prompt": r["raw_prompt"],
    #         "post_processed_answer": r["post_processed_answer"],
    #         "ground_truth": r["ground_truth"],
    #         "accuracy_score": r["accuracy_score"]
    #     }
    #     for r in results
    # ])
    # detailed_filename = os.path.join(subfolder, f"detailed_benchmark_results_{model}_{user}_{format_type}_{timestamp}.csv")
    # detailed_df.to_csv(detailed_filename, index=False, mode='a', header=not os.path.exists(detailed_filename))
    # logger.info(f"Detailed results saved to {detailed_filename}")
    
    # JSONL full responses
    jsonl_filename = os.path.join(subfolder, f"full_responses_{model}_{user}_{format_type}_{timestamp}.jsonl")
    with open(jsonl_filename, 'a') as f:
        for r in results:
            metadata = {
                "user": r["user"],
                "question_id": r["question_id"],
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "format": format_type
            }
            entry = {
                "response_text": r.get("full_response", ""),
                "post_processed_answer": r["post_processed_answer"],
                "metadata": metadata
            }
            f.write(json.dumps(entry) + '\n')
    logger.info(f"Full responses saved to {jsonl_filename}")
    
    # Summary report
    if results:
        avg_accuracy = sum(r["accuracy_score"] for r in results) / len(results)
        avg_length = sum(len(r.get("full_response", "")) for r in results) / len(results)
        logger.info(f"Summary: Overall Accuracy: {avg_accuracy:.2f}, Avg Response Length: {avg_length:.0f} chars")

def run_benchmark(user_id: int, question_indices: List[int], model_type: str, format_type: str):
    """
    Run the full benchmark for given parameters.
    
    Args:
        user_id (int): User ID
        question_indices (List[int]): Question indices
        model_type (str): Model provider
        format_type (str): Data format
    """
    # Load dataset
    dataset_path = "data/auto_eval/improved100_qualitative_with_answers.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    answer_col = f"answer_key_{user_id}"
    if answer_col not in df.columns:
        raise ValueError(f"No answer column for user {user_id}")
    
    # Load user data once
    user_data = data_load(user_id, format_type)
    
    results = []
    for q_idx in question_indices:
        if q_idx < 1 or q_idx > len(df):
            logger.warning(f"Invalid question index: {q_idx}")
            continue
        
        row = df.iloc[q_idx - 1]  # 0-based
        question = row["Query"]
        ground_truth = row[answer_col]
        
        logger.info(f"Processing question {q_idx}: {question[:50]}...")
        
        max_post_retries = 5
        post_attempt = 0
        full_response = None
        processed_answer = None
        base_prompt = prompt_build(user_data, question, model_type)
        prompt = base_prompt
        
        while post_attempt <= max_post_retries:
            # API call
            full_response = api_call(prompt, model_type)
            if full_response is None:
                logger.error(f"Failed to get response for question {q_idx} on attempt {post_attempt + 1}")
                if post_attempt == max_post_retries:
                    results.append({
                        "user": user_id,
                        "question_id": q_idx,
                        "question": question,
                        "raw_prompt": prompt,
                        "full_response": "API Error",
                        "post_processed_answer": "Error",
                        "ground_truth": ground_truth,
                        "accuracy_score": 0
                    })
                    break
                post_attempt += 1
                prompt = base_prompt + f" (retry {post_attempt})"
                continue
            
            # Post-process
            processed_answer = post_process(question, full_response)
            
            if processed_answer != "RETRY_AGENT":
                break  # Success
            
            post_attempt += 1
            logger.warning(f"Post-process returned RETRY_AGENT for question {q_idx}, retry {post_attempt}")
            if post_attempt <= max_post_retries:
                prompt = base_prompt + f" (retry {post_attempt} - please provide a clear final answer)"
        
        if processed_answer is None or processed_answer == "Error":
            processed_answer = "Processing Error"
        
        # Evaluate
        eval_result = evaluate_accuracy(processed_answer, ground_truth)
        
        result = {
            "user": user_id,
            "question_id": q_idx,
            "question": question,
            "raw_prompt": prompt,
            "full_response": full_response,
            "post_processed_answer": processed_answer,
            "ground_truth": ground_truth,
            **eval_result
        }
        results.append(result)
        logger.info(f"Question {q_idx} completed. Accuracy: {eval_result['accuracy_score']}")
    
    # Save results
    save_results(results, model_type, user_id, format_type)

def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM code execution on PHIA QA tasks")
    parser.add_argument("--user", type=int, required=True, choices=[465, 333, 171, 41],
                        help="User ID to select data for")
    parser.add_argument("--question-range", type=str, required=True,
                        help="Question range, e.g., '1-10' or '5,10-15,20'")
    parser.add_argument("--format", type=str, default="markdown", choices=["markdown", "csv", "json"],
                        help="Data formatting")
    parser.add_argument("--model", type=str, default="openai", choices=["openai", "gemini", "grok"],
                        help="LLM provider")
    
    args = parser.parse_args()
    
    try:
        indices = parse_question_range(args.question_range)
        run_benchmark(args.user, indices, args.model, args.format)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    # Self-test: Run with small range
    logger.info("Running self-test: user 465, questions 1-2, grok, markdown")
    test_indices = [1, 2]
    run_benchmark(465, test_indices, "grok", "markdown")
    logger.info("Self-test completed. Use CLI args for full runs.")