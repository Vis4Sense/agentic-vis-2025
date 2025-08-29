from typing_extensions import List
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import copy

from helpers import get_llm, get_dataset_info
from agents.simple_iteration.state import State
from agents.simple_iteration.memory import shared_memory

class ResponseFormatter(BaseModel):
    storyNodes: List[str]

def generate_storyline(state: State):
    """
    Generate a storyline based on dataset info, topic, and insights
    """
    
    # Get dataset information
    dataset_info = get_dataset_info("dataset.csv")
    
    # Get topic and insights from state
    topic = state.get("topic", "")
    
    # Collect all insights from all iterations
    all_insights = []
    
    # Add current insights
    current_insights = state.get("insights", [])
    all_insights.extend(current_insights)
    
    # Add insights from iteration history
    iteration_history = state.get("iteration_history", [])
    for iteration in iteration_history:
        if "insights" in iteration:
            all_insights.extend(iteration["insights"])
    
    # Remove duplicates while preserving order
    seen_insights = set()
    unique_insights = []
    for insight in all_insights:
        if insight not in seen_insights:
            seen_insights.add(insight)
            unique_insights.append(insight)
    
    # Create insights text for the prompt
    insights_text = ""
    if unique_insights:
        insights_text = "\n".join([f"- {insight}" for insight in unique_insights])
    
    sys_prompt = f"""
        You are a helpful assistant that generates a storyline for data exploration based on the dataset information, topic, and insights.

        The topic to explore is: {topic}

        The information about the dataset is as follows:
        {dataset_info}

        The insights discovered so far are:
        {insights_text if insights_text else "No insights discovered yet."}

        Please generate a storyline that organizes the exploration into logical story nodes.

        Rules:
        1. The storyline should be a list of story nodes.
        2. Each story node should be a string describing what aspect of the topic to explore.
        3. The storyline should be coherent and relevant to the topic.
        4. Limit the number of story nodes to no more than 5.
        5. The story should be interesting, engaging, and logical for data exploration.
        6. The story should be within the scope of the dataset.
        7. Focus on the most important aspects, the content of each node should be concise.
        8. Each story node should represent a logical step in understanding the topic.
        9. Consider the insights when creating story nodes - they should help guide the storyline.
    """

    human_prompt = f"I would like to explore the dataset about the topic of {topic}. Please generate a storyline that organizes this exploration into logical story nodes."

    llm = get_llm(temperature=0, max_tokens=4096)

    response = llm.with_structured_output(ResponseFormatter).invoke(
        [SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)]
    )

    # Use deepcopy to properly copy nested dictionaries
    new_state = copy.deepcopy(state)
    
    # Create storyline structure
    storyline = {
        "theme": topic,
        "nodes": [
            {
                "description": node,
                "insights": [],  # Will be populated later
                "visualizations": []  # Will be populated later
            }
            for node in response.storyNodes
        ]
    }
    
    new_state["storyline"] = storyline

    # Save the state to memory
    shared_memory.save_state(new_state)
    print(f"Storyline generated with {len(response.storyNodes)} nodes")
    print(f"State saved to memory for thread {shared_memory.thread_id}")

    return new_state
