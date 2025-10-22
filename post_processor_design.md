# Post-Processor Module Design

This document outlines the design for a new post-processing module (`post_processor.py`) for the PHIA agent. The module will use the OpenAI API with function calling to standardize and clean the agent's output.

## 1. OpenAI Function-Calling Tools

The following function schemas will be defined for the OpenAI API to handle various output types from the PHIA agent.

### a. `extract_numerical_value`

-   **Description**: Extracts a single numerical value (integer or float) from the agent's answer.
-   **Schema**:
    ```json
    {
      "name": "extract_numerical_value",
      "description": "Extracts a single numerical value from the agent's answer.",
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
    ```

### b. `extract_date`

-   **Description**: Extracts a date from the agent's answer and formats it as YYYY-MM-DD.
-   **Schema**:
    ```json
    {
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
    ```

### c. `normalize_activity_name`

-   **Description**: Normalizes an activity name to a value from a predefined list.
-   **Schema**:
    ```json
    {
      "name": "normalize_activity_name",
      "description": "Normalizes an activity name to a value from a predefined list.",
      "parameters": {
        "type": "object",
        "properties": {
          "activity": {
            "type": "string",
            "description": "The normalized activity name.",
            "enum": ["Outdoor Bike", "Running", "Swimming", "Other"]
          }
        },
        "required": ["activity"]
      }
    }
    ```

### d. `report_code_error`

-   **Description**: Identifies when the agent has returned a code-related error (e.g., "IndentationError", "ValueError"). This triggers a retry mechanism.
-   **Schema**:
    ```json
    {
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
    ```

### e. `handle_miscellaneous_output`

-   **Description**: Handles any other miscellaneous or unclassifiable outputs that do not fit the other function schemas.
-   **Schema**:
    ```json
    {
      "name": "handle_miscellaneous_output",
      "description": "Handles miscellaneous or unclassifiable outputs.",
      "parameters": {
        "type": "object",
        "properties": {
          "output": {
            "type": "string",
            "description": "The original, unclassified output from the agent."
          }
        },
        "required": ["output"]
      }
    }
    ```

## 2. Main Function Outline

The main function in `post_processor.py` will be `process_answer`.

```python
def process_answer(question: str, answer: str, max_retries: int = 4) -> str:
    """
    Orchestrates the post-processing of the agent's answer using the OpenAI API.

    Args:
        question (str): The original question posed to the agent.
        answer (str): The raw answer from the agent.
        max_retries (int): The maximum number of retries for code-related errors.

    Returns:
        str: The cleaned and standardized answer.
    """
    # 1. Initialize the OpenAI client.
    # 2. Define the list of tools (function schemas) to be used.
    # 3. Construct the prompt for the OpenAI API, including the question and the raw answer.
    # 4. Loop for retries (up to max_retries):
    #    a. Call the OpenAI API with the prompt and tools.
    #    b. If the API returns a function call for 'report_code_error':
    #       - Log the error.
    #       - Increment the retry counter and continue the loop.
    #    c. If the API returns any other function call:
    #       - Extract the standardized value from the function call's arguments.
    #       - Return the extracted value.
    #    d. If the API does not return a function call:
    #       - Return the raw answer as a fallback.
    # 5. If the loop completes without a successful result, return an error message.

```

## 3. Integration Plan

The new post-processing module will be integrated into the `run_with_retry` function in `recreate_phia_study_parallel.py`.

-   **Location**: Inside the `run_with_retry` function, after a successful response is received from the agent (i.e., after the `ot.run` call).
-   **Logic**:
    1.  Import the `process_answer` function from `post_processor.py` at the top of the script.
    2.  After getting the `final_answer` from the agent, call `process_answer(question, final_answer)`.
    3.  The result of this call will be the new, cleaned `model_answer`.
    4.  This cleaned `model_answer` will then be stored in the results dictionary.

## 4. Architectural Diagram

The following Mermaid diagram illustrates the data flow:

```mermaid
graph TD
    A[Agent's Raw Output] --> B{Post-Processing Module};
    B --> C{OpenAI API w/ Function Calling};
    C -->|Function Call| D[Standardized Output];
    D --> E[Final Cleaned Output];
    C -->|Code Error| B;
    B -->|Retry| C;