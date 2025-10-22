# -*- coding: utf-8 -*-

"""
Post-Processing Module for PHIA Agent Output

This module provides functionality to clean and standardize the output from the PHIA agent
using the OpenAI API with function calling.
"""

import os
import json
from openai import OpenAI
from typing import Optional


# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ai_model = os.getenv("OPENAI_MODEL_NAME")


# Define the function schemas for OpenAI API
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "extract_numerical_value",
            "description": "Extracts a single numerical value from the agent's answer. If the answer indicates no activity or zero, return 0.0.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "The numerical value extracted from the answer."
                    }
                },
                "required": ["value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_yes_no",
            "description": "Extracts a Yes or No answer from the agent's response for yes/no questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The yes/no answer, either 'Yes' or 'No'.",
                        "enum": ["Yes", "No"]
                    }
                },
                "required": ["answer"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_date",
            "description": "Extracts a date from the agent's answer and formats it as YYYY-MM-DD.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "The date extracted from the answer, in YYYY-MM-DD format."
                    }
                },
                "required": ["date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "normalize_activity_name",
            "description": "Normalizes an activity name to a value from a predefined list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "activity": {
                        "type": "string",
                        "description": "The normalized activity name.",
                        "enum": ["Outdoor Bike", "Bike", "Run", "Swim", "Elliptical", "Aerobic Workout", "Spinning", "Yoga", "Weights", "Treadmill" "Other"]
                    }
                },
                "required": ["activity"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "report_code_error",
            "description": "Reports a code-related error found in the agent's output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "error_type": {
                        "type": "string",
                        "description": "The type of code error identified (e.g., 'IndentationError')."
                    },
                    "error_message": {
                        "type": "string",
                        "description": "A brief description of the error."
                    }
                },
                "required": ["error_type", "error_message"]
            }
        }
    }
]


def process_answer(question: str, answer: str) -> str:
    """
    Processes the agent's answer using the OpenAI API to clean and standardize it.

    Args:
        question (str): The original question posed to the agent.
        answer (str): The raw answer from the agent.

    Returns:
        str: The cleaned and standardized answer, or "RETRY_AGENT" if a code error is detected.
    """
    print(f"Post-processing: Agent's original answer: {answer}")

    prompt = f"""
    You are a post-processing assistant for the PHIA agent. Your task is to clean and standardize the agent's output based on the original question.

    Original Question: {question}
    Agent's Raw Answer: {answer}

    Analyze the agent's answer and use the appropriate function to extract or standardize the information.
    If the answer contains code or processing errors, function calls (like <function_call or tool_code), agent reasoning logs (with [THOUGHT] or [ACT]), or other nonsensical outputs that don't make sense as an answer to the question, use report_code_error to trigger a retry.
    For yes/no questions, use extract_yes_no to return just 'Yes' or 'No'.
    For numerical questions, if the answer indicates no activity or zero, use extract_numerical_value with 0.0.
    For questions asking "when" or "which day" or other indicators that the objective is a date, use extract_date to format as YYYY-MM-DD.
    """

    try:
        response = client.chat.completions.create(
            model=ai_model,  
            messages=[
                {"role": "system", "content": "You are a helpful assistant for post-processing agent outputs."},
                {"role": "user", "content": prompt}
            ],
            tools=TOOLS,
            tool_choice="auto"
        )

        message = response.choices[0].message

        if message.tool_calls:
            tool_call = message.tool_calls[0]
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            print(f"Post-processing: Function called: {function_name}")

            if function_name == "report_code_error":
                print(f"Code error detected in agent answer: {arguments['error_type']} - {arguments['error_message']}")
                final_response = "RETRY_AGENT"
            elif function_name == "extract_numerical_value":
                final_response = str(arguments["value"])
            elif function_name == "extract_yes_no":
                final_response = arguments["answer"]
            elif function_name == "extract_date":
                final_response = arguments["date"]
            elif function_name == "normalize_activity_name":
                final_response = arguments["activity"]
            elif function_name == "handle_miscellaneous_output":
                final_response = arguments["output"]
        else:
            # No function call, return raw answer as fallback
            final_response = answer

        print(f"Post-processing: Final response: {final_response}")
        return final_response

    except Exception as e:
        print(f"Error in OpenAI API call during post-processing: {e}")
        final_response = answer  # Return original answer on error
        print(f"Post-processing: Final response (on error): {final_response}")
        return final_response

    return answer