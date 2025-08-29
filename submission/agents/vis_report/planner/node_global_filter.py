from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from helpers import get_llm, get_dataset_info, query_by_sql
from agents.vis_report.planner.state import State
from agents.vis_report.memory import memory

from agents.vis_report.load_config import config


class ResponseFormatter(BaseModel):
    description: str = Field(description="The description of the SQL query")
    sql_query: str = Field(description="The SQL query to select the data from the dataset")


def global_filter(state: State):
    """
    Generate SQL query to select the data (based on the topic)
    """

    if config["dev"]:
        if "global_filter_state" in state and state["global_filter_state"]:
            return state
    print(f"▶️ Global filtering")
    
    # Get path of the main program being executed
    dataset_info = get_dataset_info(config['dataset'])

    sys_prompt = f"""
        You are a helpful assistant that generates SQL queries to select the data from the dataset.
        The dataset is as follows:
        {dataset_info}

        You may refer to the following domain knowledge:
        {config["domain_knowledge"]}

        Please generate a SQL query to select the data from the dataset to support the analysis of the topic given by the user and return all the columns.

        Rules:
        1. Always use 'FROM Papers' (not FROM dataset or any other table name)
        2. Use standard SQL syntax compatible with pandasql
        3. Make sure column names match exactly with the dataset headers
        4. Use double quotes for column names that contain special characters
    """

    human_prompt = f"I would like to explore the dataset about the topic of {config['topic']}. Please generate a SQL query to select the data."

    llm = get_llm(temperature=0, max_tokens=4096)

    response = llm.with_structured_output(ResponseFormatter).invoke(
        [SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)]
    )

    # test sql query
    print("LLM generatedsql_query: ", response.sql_query)
    dataset = query_by_sql(response.sql_query)
    print("Filtered dataset size: ", dataset.shape)
    if memory.latest_state is not None:
        thread_dir = memory._get_thread_dir()
    else:
        thread_dir = "outputs_sync/vis_report"

    dataset_path = f"{thread_dir}/dataset_global_filtered.csv"
    dataset.to_csv(dataset_path, index=False)

    new_state = state.copy()
    new_state["global_filter_state"] = {
        "description": response.description,
        "sql_query": response.sql_query,
        "dataset_path": dataset_path
    }

    # Save the state to memory
    memory.save_state(new_state)
    print(f"✅ Global filtering state saved to memory for thread {memory.thread_id}")

    return new_state