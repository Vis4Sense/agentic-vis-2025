from typing_extensions import TypedDict
from typing import List, Union, Literal, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from helpers import get_llm, get_dataset_info

from agents.vis_report.analyser.agent import Agent as AnalyserAgent
from agents.vis_report.planner.state import State
from agents.vis_report.memory import memory

from agents.vis_report.load_config import config


def execute(state: State):
    print(f"🔍 Executing report plan")

    new_state = state.copy()
    sections = new_state["report_outline"]

    for section_index, section in enumerate(sections):
        print(f">>> Executing section {section['section_number']}: {section['section_name']}")

        analyses = section["analyses"]
        for analysis_index, analysis in enumerate(analyses):
            analysis_schema = analysis["analysis_schema"]

            print(f"    - Executing analysis {analysis_schema['action']}")
            # if analysis["action"] == "present":
            #     visualisation = analysis["visualisation"]
            #     knowledge = analysis["knowledge"]
            # elif analysis["action"] == "explore":
            #     pass
            # else:
            #     raise ValueError(f"Invalid action: {analysis['action']}")

            analyser_agent = AnalyserAgent()
            analyser_agent.initialize(analysis)

            # Change made:add global filter to the analyser agent
            global_filter_state = new_state["global_filter_state"]
            analyser_agent.state["global_filter_state"] = global_filter_state

            analyse_result = analyser_agent.process()
            # print(f"    - Analysis result: {analyse_result}")

            new_state["report_outline"][section_index]["analyses"][analysis_index] = analyse_result

            memory.save_state(new_state)

    return new_state


