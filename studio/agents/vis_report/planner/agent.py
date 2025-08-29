import traceback

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from datetime import datetime
import shutil

from agents.vis_report.planner.node_plan import plan
from agents.vis_report.planner.node_execute import execute
from agents.vis_report.planner.state import State
from agents.vis_report.planner.node_write import write_content
from agents.vis_report.memory import memory

from agents.vis_report.load_config import config
from agents.vis_report.planner.report_html import generate_html_report

def create_workflow():
    builder = StateGraph(State)
    # builder.add_node("plan", plan)

    # TODO: use the async version
    from agents.vis_report.planner.node_plan_async import plan
    builder.add_node("plan", plan)

    # TODO: add global filter node
    from agents.vis_report.planner.node_global_filter import global_filter
    builder.add_node("global_filter", global_filter)

    builder.add_node("execute", execute)
    builder.add_node("write", write_content)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "global_filter")
    builder.add_edge("global_filter", "execute")
    builder.add_edge("execute", "write")
    builder.add_edge("write", END)

    # builder.add_edge(START, "plan")
    # builder.add_edge("plan", "execute")
    # builder.add_edge("execute", "write")
    # builder.add_edge("write", END)

    # builder.add_node("write", write_content)
    # builder.add_edge(START, "write")
    # builder.add_edge("write", END)

    return builder.compile()

class Agent:
    def __init__(self) -> None:
        self.workflow = None

    def initialize(self):
        self.workflow = create_workflow()
        png_data = self.workflow.get_graph().draw_mermaid_png()

    def initialize_state(self) -> dict:

        if config["dev"]:
            try:
                new_state = memory.load_state_from_thread(config["thread_to_load"])
                print(f"✅ Loaded state from thread {config['thread_to_load']}")
                memory.save_state(new_state)
                return new_state
            except Exception as e:
                print(f"❌ Error: {e}")
                print(f"❌ Starting from scratch")

        state = {
            "config": config,
            "report_outline": []
        }
        return state
    
    def process(self):
        if self.workflow is None:
            raise RuntimeError("Agent not initialised. Call initialize() first.")
        
        state = self.initialize_state()

        output_state = None

        try:
            output_state = self.workflow.invoke(state, config={"configurable": {"thread_id": memory.thread_id}})
        except Exception as e:
            print(f"❌ Error: {e}")
            traceback.print_exc()
            output_state = memory.latest_state
        
        generate_html_report(output_state, f"outputs_sync/vis_report/{memory.thread_id}/report.html")
        
        # copy to output.html
        shutil.copy(f"outputs_sync/vis_report/{memory.thread_id}/report.html", "output.html")

        return
