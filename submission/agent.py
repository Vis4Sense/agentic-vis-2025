from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from datetime import datetime

import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
print("Agent current_dir", current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
print("Agent current_dir", sys.path)

from agents.vis_report.memory import memory

from agents.vis_report.planner.state import State
from agents.vis_report.planner.agent import create_workflow as create_planner_workflow
from agents.vis_report.planner.agent import Agent as PlannerAgent


def create_workflow():
    return create_planner_workflow()


class Agent(PlannerAgent):
    def __init__(self) -> None:
        super().__init__()
        self.workflow = create_workflow()
