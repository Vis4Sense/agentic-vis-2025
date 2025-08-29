import traceback

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from datetime import datetime

from agents.vis_report.analyser.state import State
from agents.vis_report.load_config import config

from agents.vis_report.analyser.node_vis import visualise
from agents.vis_report.analyser.memory import memory
from agents.vis_report.analyser.node_facts import extract_facts

def create_workflow():
    builder = StateGraph(State)
    # builder.add_node("vis", visualise)
    from agents.vis_report.analyser.node_vis_withnetwork import visualise
    builder.add_node("vis", visualise)

    # builder.add_node("facts", extract_facts)
    from agents.vis_report.analyser.node_facts_retry import extract_facts
    builder.add_node("facts", extract_facts)

    builder.add_edge(START, "vis")
    builder.add_edge("vis", "facts")
    builder.add_edge("facts", END)

    

    return builder.compile()


class Agent:
    def __init__(self) -> None:
        self.workflow = None

    def initialize(self, init_state: State):
        self.workflow = create_workflow()
        self.state = init_state

    # def initialize_state(self) -> dict:
    #     if self.state is None:
    #         state = {
    #             "config": config,
    #             "analysis_schema": analysis_schema
    #         }
    #     return state
    
    def process(self, thread_id: str = None):
        if self.workflow is None:
            raise RuntimeError("Agent not initialised. Call initialize() first.")
        
        # state = self.initialize_state()
        state = self.state
        memory.add_state(state)
        output_state = None

        try:
            output_state = self.workflow.invoke(state, config={"configurable": {"thread_id": thread_id}})
        except Exception as e:
            print(f"❌ Analyser Error: {e}")
            traceback.print_exc()
            output_state = memory.get_latest_state()

        return output_state
