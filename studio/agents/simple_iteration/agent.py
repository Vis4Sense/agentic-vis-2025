from typing import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from datetime import datetime

from agents.simple_iteration.report_html import generate_html_report
from report_pdf import generate_pdf_report

from agents.simple_iteration.memory import shared_memory

from agents.simple_iteration.node_select_data import select_data
from agents.simple_iteration.node_question import question
from agents.simple_iteration.node_facts import get_facts
from agents.simple_iteration.node_insights import get_insights
from agents.simple_iteration.node_storyline import generate_storyline
from agents.simple_iteration.node_assign_insights import assign_insights_to_story_nodes
from agents.simple_iteration.node_story_vis import generate_story_visualizations
from agents.simple_iteration.node_follow_up_decision import follow_up_decision
from agents.simple_iteration.node_vis import generate_visualizations
from agents.simple_iteration.state import State


def create_workflow():
    # create the agentic workflow using LangGraph
    builder = StateGraph(State)
    builder.add_node("select_data", select_data)
    builder.add_node("question", question)
    builder.add_node("facts", get_facts)
    builder.add_node("insights", get_insights)
    builder.add_node("storyline", generate_storyline)
    builder.add_node("assign_insights", assign_insights_to_story_nodes)
    builder.add_node("story_visualizations", generate_story_visualizations)
    builder.add_node("follow_up_decision", follow_up_decision)
    builder.add_node("visualizations", generate_visualizations)
    
    builder.add_edge(START, "select_data")
    builder.add_edge("select_data", "question")
    builder.add_edge("question", "facts")
    builder.add_edge("facts", "insights")
    
    # Add conditional edge from insights to either follow_up_decision or storyline
    def route_after_insights(state):
        if state.get("should_continue", False):
            return "follow_up_decision"
        else:
            return "storyline"

    builder.add_conditional_edges("insights", route_after_insights, {
        "follow_up_decision": "follow_up_decision",
        "storyline": "storyline"
    })
    
    # Add edge from storyline to assign_insights
    builder.add_edge("storyline", "assign_insights")
    
    # Add edge from assign_insights to story_visualizations
    builder.add_edge("assign_insights", "story_visualizations")
    
    # Add edge from story_visualizations to visualizations
    builder.add_edge("story_visualizations", "visualizations")
    
    # Add edge from visualizations to END
    builder.add_edge("visualizations", END)
    
    # Add conditional edge from follow_up_decision to either select_data or facts
    def route_after_follow_up_decision(state):
        if state.get("follow_up_decision", {}).get("should_reselect_data", False):
            return "select_data"
        else:
            return "facts"

    builder.add_conditional_edges("follow_up_decision", route_after_follow_up_decision, {
        "select_data": "select_data",
        "facts": "facts"
    })
    
    # Comment out the old direct edge from insights to question
    # builder.add_edge("insights", "question")
    
    # builder.add_edge("insights", "visualizations")
    # builder.add_edge("visualizations", END)
    # builder.add_edge("insights", END)  # Remove this line
    return builder.compile()

class Agent:
    def __init__(self):
        self.workflow = None

    def initialize(self):
        self.workflow = create_workflow()

    def initialize_state(self) -> dict:
        state = {
            # "topic": "evolution of research on sensemaking",
            # "topic": "find a phd on visualisation for digital humanities",
            "topic": "Jefferey Heer's research and influences",
            "iteration_count": 0,      # Initialize iteration counter
            "max_iterations": 2,       # Set maximum iterations (adjust as needed)
            "should_continue": True,   # Initialize to continue
            "iteration_history": [],   # yuhan: this is not used
            "storyline": {             # Initialize storyline
                "theme": "",
                "nodes": []
            }
        }
        return state

    def decode_output(self, output: dict):
        # if the final output contains Vega-Lite codes, then use generate_html_report
        # if the final output contains Python codes, then use generate_pdf_report

        # generate_pdf_report(output, "output.pdf")
        output_path = f"outputs/simple_iteration/{shared_memory.thread_id}/output.html"
        generate_html_report(output, output_path, shared_memory)
        print(f"Visualization report generated: {output_path}")
        # generate_html_report(output, "output.html", shared_memory)
        # print(f"Visualization report generated: output.html")
    
    def generate_thread_id(self) -> str:
        """Generate a unique thread ID"""
        return f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def process(self, thread_id: str = None):
        if self.workflow is None:
            raise RuntimeError("Agent not initialised. Call initialize() first.")
        
        # Generate thread_id if not provided
        if thread_id is None:
            thread_id = self.generate_thread_id()
            shared_memory.set_thread_id(thread_id)
            print(f"Generated new thread_id: {thread_id}")
        
        # initialize the state & save to memory
        state = self.initialize_state()
        shared_memory.save_state(state)

        # invoke the workflow with generated thread_id
        try:
            output_state = self.workflow.invoke(state, config={"configurable": {"thread_id": thread_id}})
        except Exception as e:
            # output_state = self.workflow.get_latest_state()
            # Get the last state from memory history
            history = shared_memory.get_history()
            output_state = history[-1] if history else state
        print(output_state)

        # flatten the output
        def _flatten(value):
            return getattr(value, "content", value)
        result = {k: _flatten(v) for k, v in output_state.items()}

        # decode the output
        self.decode_output(result)

        # return the result
        return result
