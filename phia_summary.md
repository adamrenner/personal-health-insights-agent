
## **Personal Health Insights Agent (PHIA): Operational Summary**

PHIA is an agent system designed to analyze and interpret complex, multi-dimensional, continuous behavioral health data derived from wearables. Its primary function is to transform this raw data into **personalized, actionable insights** in response to thousands of health queries.

### 1. Core Architecture and Reasoning

PHIA's superior performance stems from its use of the **ReAct agent framework** (Reasoning and Acting). This framework enables multi-step iterative reasoning, which is essential for tackling the complex, open-ended analysis required for personal health questions.

The agent continuously cycles through three sequential stages:

1. **Thought:** The model integrates its current context and prior outputs to formulate a plan (e.g., deciding which tool to use next). This strategic planning is critical; its absence in baselines led to twice the error rate and zero recovery from fatal errors.
2. **Act:** The model executes its strategy by dispatching commands to one of its auxiliary tools (Code Generation or Search).
3. **Observe:** The outputs from the tool execution (e.g., data analysis results, web search snippets, or error messages) are incorporated back into the model's context for the next planning step. This allows PHIA to **recover from fatal errors** in 11.4% of cases.

### 2. PHIA's Essential Tools

To overcome the inherent limitations of Large Language Models (LLMs) in numerical precision and accessing up-to-date information, PHIA is augmented with two critical external tools:

#### A. Wearable Data Analysis with Code Generation (Python/Pandas)

- **Function:** Used during the `Act` stage to engage with wearable tabular data.
- **Implementation:** Leverages the **Pandas Python library** within a customized sandbox runtime environment.
- **Necessity:** **Numerical results derived from code generation are factual and reliably maintain arithmetic precision**. This tool is mandatory for objective numerical analysis because previous LLM-only baselines showed "poor mathematical and tabular reasoning abilities," achieving only **22% accuracy** on objective queries.
- **Data Structure:** The agent analyzes structured synthetic wearable user data, which includes two main tables:
    - **Daily Summary Table:** Contains metrics like `steps`, `sleep_minutes`, `resting_heart_rate`, and various sleep percentages for each day.
    - **Activities Table:** Details specific activity events, including `startTime`, `activityName` (e.g., 'Run', 'Yoga'), `distance` (in meters), and `calories` burned.

#### B. Integration of Additional Health Knowledge (Web Search)

- **Function:** Used during the `Act` stage to retrieve the latest and relevant health information.
- **Necessity:** This custom search capability addresses the **inherent limitations of the language model's training on historical data**. It is essential for generating comprehensive and personalized answers to open-ended queries (e.g., "How do I reduce stress?"), especially those requiring **Domain Knowledge** or comparisons to cohort/population norms.
- **Benefits:** The search mechanism allows for direct attribution of information, bolstering credibility in a sensitive domain.

### 3. Key Artifacts and Directory Structure

The repository structure houses the essential components for replicating and testing PHIA:

| Folder/File Name               | Purpose and Explanation                                                                                                                                                                                                 |
| :----------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `phia_agent.py`                | Contains the **core agent logic** for PHIA. This file manages the implementation of the ReAct loop and dictates how the LLM interacts with its external tools (Code and Search).                                        |
| `prompt_templates.py`          | Contains key prompt templates, such as the **agent preamble**, which define the instructions, constraints, and tool definitions provided to the underlying LLM. Few-shot examples are also provided to guide the model. |
| `synthetic_wearable_users/`    | Contains the synthetic, high-fidelity wearable user data used for evaluation. Subjects **465, 333, 171, and 41** were specifically utilized for the original evaluation.                                                |
| `data/`                        | Contains model outputs and human annotations from the extensive evaluation process.                                                                                                                                     |
| `figs/`                        | Contains all the code necessary to reproduce the figures and graphical results presented in the paper.                                                                                                                  |
| `few_shots/`                   | Contains the **few-shot examples** (ReAct trajectories) used to augment PHIA's performance, providing high-quality demonstrations of iterative planning, code generation, and web search use.                           |
| `Objective Query - PHIA.xlsx`  | Contains the benchmark set of **4000 objective queries** (e.g., "What was the duration of my last run?") used for automatic numerical evaluation.                                                                       |
| `Open-Ended Query - PHIA.xlsx` | Contains the benchmark set of **172 open-ended queries** (e.g., "How can I improve my sleep?") used for detailed human and expert evaluation of reasoning quality.                                                      |
| `phia_demo.ipynb`              | A primary runnable notebook that provides code to try out PHIA. API keys (for Gemini and search via Tavily) must be configured within this notebook.                                                                    |
| `setup.sh`                     | A shell script used to fully set up the required `phia` conda environment, which requires Python 3.11 or higher.                                                                                                        |