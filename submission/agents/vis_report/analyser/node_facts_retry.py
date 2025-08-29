from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import json

from helpers import get_llm, get_dataset_info, query_by_sql
from agents.vis_report.analyser.state import State, Model
from agents.vis_report.analyser.memory import memory
from agents.vis_report.analyser.sandbox import run_in_sandbox, run_in_sandbox_with_venv

from agents.vis_report.load_config import config


def extract_facts(state: State, max_retries=3):
    """
    Generate python code to get the facts about the dataset with retry logic.
    """

    if config["dev"]:
        if "knowledge" in state and state["knowledge"]:
            return state

    print(f"▶️ Extracting facts")
    fact_term = "facts"

    llm = get_llm(temperature=0, max_tokens=8192)

    # Get path of the main program being executed
    dataset_info = get_dataset_info(config['dataset']) # TODO: use transformed data

    base_sys_prompt = f"""
        You are a data analyst who write python script to analyse the dataset.

        The dataset is as follows:
        {dataset_info}

        Please generate a python code to analyse the dataset, the goal is to get {fact_term} about the dataset that can answer the given question.

        Background information:
        General topic: {config['topic']}

        You should read the dataset from the following path:
        The original dataset path is: {config['dataset']}
        The global filtered dataset path is: {state['global_filter_state']['dataset_path']}
        You can use the global filtered dataset path if it is available and needed, otherwise use the original dataset path.

        Analysis requirement:
        {state['analysis_schema']['information_needed']}

        Requirements:
        1. only get the most relevant {fact_term}, don't generate too many.
        2. make the code concise
        3. example {fact_term}: statistics, top k papers, trends, most cited authors, etc. you are free to choose the {fact_term} that are most relevant to the question. just keep in mind that you are an expert in data analysis.
        4. the {fact_term} should be relevant to the question
        5. irrelevant or meaningless facts should be avoided
        6. each {fact_term} should be printed as:
            ### Begin of {fact_term}
            <{fact_term}>
            ### End of {fact_term}
        7. feel free to use python libraries to help you analyse the dataset. supported libraries: pandas, numpy, matplotlib, seaborn, networkx.
        8. make sure the code is executable.
        9. if the question is too complex and can not be solved by a short code, just ignore it and do the most basic and simple analysis.
        10. the code should be short and concise.

        Code requirements:
        - The code must be fully valid and executable in Python 3, with correct indentation and no syntax errors. 
        - Prioritize clarity and robustness over brevity: avoid overly complex one-liners; instead, write explicit, step-by-step code that is easy to follow and less error-prone.
        - Always handle missing or null values (NaN/None) safely before performing string concatenation, arithmetic, or statistical calculations. 
        - For text fields: replace missing values with an empty string ("").
        - For numeric fields: replace missing values with 0 or use functions that gracefully handle NaN.
        - Never assume column data types. Explicitly cast values to the required type (e.g., str(), int(), float()) before use.
        - If the analysis request is too complex to implement reliably, simplify it into smaller, robust steps and return only the most basic and directly relevant facts.

        You may refer to the following domain knowledge, if needed:
        {config['domain_knowledge']}
    """

    class ResponseFormatter(BaseModel):
        model: Model

    last_error = None
    last_code = None

    for attempt in range(max_retries):
        print(f"🔄 Attempt {attempt + 1}/{max_retries}")
        
        # Build system prompt with error context if this is a retry
        if attempt == 0:
            sys_prompt = base_sys_prompt
        else:
            sys_prompt = base_sys_prompt + f"""

            IMPORTANT: This is attempt {attempt + 1}. The previous code failed with the following error:
            {last_error}

            Previous code that failed:
            {last_code}

            Please fix the code based on the error message. Common issues to check:
            1. Python syntax errors
            2. Pandas method usage (especially .replace() method - use .apply() or .map() instead of .replace() with lambda)
            3. Library import issues
            4. Column name mismatches
            5. Data type conversion errors
            """

        human_prompt = f"""
        Please generate the python code.
        """

        response = llm.with_structured_output(ResponseFormatter).invoke(
            [SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)]
        )

        new_state = state.copy()
        new_state["model"] = response.model
        memory.add_state(new_state)

        # Run the code in sandbox
        try:
            result = run_in_sandbox(response.model["python_script"])
            last_code = response.model["python_script"]
        except Exception as e:
            result = {
                "stdout": "",
                "stderr": str(e),
                "exit_code": 1
            }
            last_error = str(e)
            last_code = response.model["python_script"]
        
        print(f"🔍 Facts extraction result: {result}")

        if result["exit_code"] == 0:
            new_state["knowledge"] = {
                "facts": result["stdout"],
            }
            # Save the state to memory
            memory.add_state(new_state)
            return new_state
        else:
            last_error = result["stderr"]
            last_code = response.model["python_script"]
            print(f"❌ Attempt {attempt + 1} failed, retrying...")

    # All attempts failed
    print(f"❌ All {max_retries} attempts failed")
    new_state = state.copy()
    new_state["knowledge"] = {
        "facts": f"Failed to extract facts after {max_retries} attempts. Last error: {last_error}",
    }
    memory.add_state(new_state)
    return new_state
