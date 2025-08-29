from typing_extensions import TypedDict
from typing import List, Union, Literal, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from helpers import get_llm, get_dataset_info

from agents.vis_report.analyser.state import State, Visualisation
from agents.vis_report.analyser.memory import memory

from agents.vis_report.load_config import config

def visualise(state: State):
    """
    Visualise the data based on the information need.
    """

    if config["dev"]:
        if "visualisation" in state and state["visualisation"]:
            return state

    new_state = state.copy()
    visualisation = get_vega_lite_spec(state)
    new_state["visualisation"] = visualisation.visualisation
    memory.add_state(new_state)
    return new_state


def get_vega_lite_spec(state: State):
    """
    Get the vega-lite specification for the visualisation.
    """
    llm = get_llm()
    dataset_info = get_dataset_info(config["dataset"])

    system_message = SystemMessage(content=f"""
    You are an expert in creating vega-lite specifications for visualisations.

    Use this dataset: {config["dataset_url"]}

    The dataset information is as follows:
    {dataset_info}

    Please following the information need when you designing and generating the visualisation:
    {state["analysis_schema"]["information_needed"]}

    You may refer to the following domain knowledge:
    {config["domain_knowledge"]}

    Requirements:
    1. Generate valid vega-lite specification.
    2. Robustness is prioritised over complexity.
    3. If the information need is too complex, you don't need to fulfil the complete need. You may generate a visualisation that is relevant to the core need.
    """
    )

    human_message = HumanMessage(content=f"""
    Please generate the vega-lite specification for the visualisation.
    """
    )

    class ResponseFormatter(BaseModel):
        visualisation: Visualisation

    response = llm.with_structured_output(ResponseFormatter).invoke([system_message, human_message])
    return response
