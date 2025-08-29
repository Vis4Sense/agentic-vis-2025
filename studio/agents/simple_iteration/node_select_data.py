from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from helpers import get_llm, get_dataset_info, query_by_sql
from agents.simple_iteration.state import State
from agents.simple_iteration.memory import shared_memory

class ResponseFormatter(BaseModel):
    description: str = Field(description="The description of the SQL query")
    sql_query: str = Field(description="The SQL query to select the data from the dataset")


def select_data(state: State):
    """
    Generate SQL query to select the data (based on the topic)
    """

    # Get path of the main program being executed
    dataset_info = get_dataset_info("dataset.csv")

    sys_prompt = f"""
        You are a helpful assistant that generates SQL queries to select the data from the dataset.
        The dataset is as follows:
        {dataset_info}

        Please generate a SQL query to select the data from the dataset to support the analysis of the topic given by the user.

        Rules:
        1. Always use 'FROM Papers' (not FROM dataset or any other table name)
        2. Use standard SQL syntax compatible with pandasql
        3. Make sure column names match exactly with the dataset headers
        4. Use double quotes for column names that contain special characters
    """

    human_prompt = f"I would like to explore the dataset about the topic of {state['topic']}. Please generate a SQL query to select the data."

    llm = get_llm(temperature=0, max_tokens=4096)

    response = llm.with_structured_output(ResponseFormatter).invoke(
        [SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)]
    )

    # test sql query
    dataset = query_by_sql(response.sql_query)
    thread_dir = shared_memory._get_thread_dir()
    dataset_path = f"{thread_dir}/dataset_selected.csv"
    dataset.to_csv(dataset_path, index=False)

    new_state = state.copy()
    new_state["select_data_state"] = {
        "description": response.description,
        "sql_query": response.sql_query,
        "dataset_path": dataset_path
    }

    # Save the state to memory
    shared_memory.save_state(new_state)
    print(f"state saved to memory for thread {shared_memory.thread_id}")

    return new_state
