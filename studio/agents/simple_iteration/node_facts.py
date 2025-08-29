from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import json

from helpers import get_llm, get_dataset_info, query_by_sql
from agents.simple_iteration.state import State
from agents.simple_iteration.memory import shared_memory
from agents.simple_iteration.sandbox import run_in_sandbox, run_in_sandbox_with_venv

class ResponseFormatter(BaseModel):
    code: str = Field(description="The python code to get the facts about the dataset")


def get_facts(state: State):
    """
    Generate SQL query to select the data (based on the topic)
    """

    fact_term = "facts"

    # Get path of the main program being executed
    dataset_info = get_dataset_info(state['select_data_state']['dataset_path'])

    sys_prompt = f"""
        You are a data analyst who write python script to analyse the dataset.

        The dataset is as follows:
        {dataset_info}

        Please generate a python code to analyse the dataset, the goal is to get {fact_term} about the dataset that can answer the given question.

        You should read the dataset from the following path:
        {state['select_data_state']['dataset_path']}

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
        10. keep the code short and concise.
    """

    human_prompt = f"""
    I would like to explore the dataset about the topic of {state['topic']}.
    The current analysis question is: {state['question']['question']}.
    Please generate the python code.
    """

    llm = get_llm(temperature=0, max_tokens=8192)

    response = llm.with_structured_output(ResponseFormatter).invoke(
        [SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)]
    )

    # Run the code in sandbox
    try:
        result = run_in_sandbox_with_venv(response.code)
    except Exception as e:
        result = {
            "stdout": "",
            "stderr": str(e),
            "exit_code": 1
        }
    print(result)

    new_state = state.copy()
    new_state["facts"] = {
        "code": response.code,
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "exit_code": result["exit_code"]
    }

    # Save the state to memory
    shared_memory.save_state(new_state)
    print(f"state saved to memory for thread {shared_memory.thread_id}")

    return new_state
