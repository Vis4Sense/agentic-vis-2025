from typing_extensions import List
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import copy

from helpers import get_llm
from agents.simple_iteration.state import State
from agents.simple_iteration.memory import shared_memory

class InsightAssignment(BaseModel):
    story_node_index: int = Field(description="Index of the story node (0-based) to assign this insight to")
    reasoning: str = Field(description="Brief explanation of why this insight belongs to this story node")

class InsightAssignments(BaseModel):
    assignments: List[InsightAssignment] = Field(description="List of insight assignments to story nodes")

def assign_insights_to_story_nodes(state: State):
    """
    Assign insights to appropriate story nodes based on their content and relevance
    """
    
    # Get storyline from state
    storyline = state.get("storyline", {})
    
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
    
    insights = unique_insights
    
    if not insights:
        print("No insights available for assignment")
        return state
    
    if not storyline or "nodes" not in storyline:
        print("No storyline available for insight assignment")
        return state
    
    story_nodes = storyline["nodes"]
    if not story_nodes:
        print("No story nodes available for insight assignment")
        return state
    
    print(f"Assigning {len(insights)} insights to {len(story_nodes)} story nodes")
    
    # Create system prompt for insight assignment
    story_nodes_text = "\n".join([f"{i+1}. {node['description']}" for i, node in enumerate(story_nodes)])
    insights_text = "\n".join([f"{i+1}. {insight}" for i, insight in enumerate(insights)])
    
    sys_prompt = f"""
    You are a data analysis expert who assigns insights to appropriate story nodes in a data exploration narrative.
    
    Your task is to assign each insight to the most relevant story node based on the content and theme of both the insight and the story node.
    
    Story nodes:
    {story_nodes_text}
    
    Available insights:
    {insights_text}
    
    Rules for assignment:
    1. Each insight should be assigned to exactly one story node
    2. Choose the story node that best represents the theme or topic of the insight
    3. Consider the logical flow of the narrative
    4. If an insight could fit multiple nodes, choose the one that provides the best narrative progression
    5. Provide a brief reasoning for each assignment
    
    IMPORTANT: 
    - Use 0-based indexing for story node indices (0 for the first story node, 1 for the second, etc.)
    - Each insight must be assigned to a valid story node index
    - The reasoning should explain why this insight belongs to this particular story node
    """
    
    human_prompt = f"""
    Please assign each of the {len(insights)} insights to the most appropriate story node from the {len(story_nodes)} available story nodes.
    
    For each insight, provide:
    1. The story node index (0-based) it should be assigned to
    2. A brief reasoning for the assignment
    
    Focus on creating a logical narrative flow where insights support and build upon each story node's theme.
    """
    
    llm = get_llm(temperature=0, max_tokens=4096)
    
    try:
        response = llm.with_structured_output(InsightAssignments).invoke(
            [SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)]
        )
        
        print(f"Generated {len(response.assignments)} insight assignments")
        
        # Convert to state format and update state
        new_state = copy.deepcopy(state)
        
        # Update storyline with assigned insights
        updated_storyline = storyline.copy()
        updated_nodes = []
        
        # Initialize all nodes with empty insights lists
        for story_node in story_nodes:
            updated_node = story_node.copy()
            updated_node["insights"] = []
            updated_nodes.append(updated_node)
        
        # Assign insights to nodes based on the response
        for i, assignment in enumerate(response.assignments):
            if i < len(insights) and assignment.story_node_index < len(updated_nodes):
                insight = insights[i]
                node_index = assignment.story_node_index
                updated_nodes[node_index]["insights"].append(insight)
                print(f"Assigned insight {i+1} to story node {node_index+1}: {assignment.reasoning}")
        
        updated_storyline["nodes"] = updated_nodes
        new_state["storyline"] = updated_storyline
        
        # Save state to memory
        shared_memory.save_state(new_state)
        print(f"Insight assignments saved to memory for thread {shared_memory.thread_id}")
        
        # Print summary
        total_assigned = sum(len(node.get("insights", [])) for node in updated_nodes)
        print(f"Insight assignment summary: {total_assigned}/{len(insights)} insights assigned to {len(updated_nodes)} story nodes")
        
        # Debug: Print details of each story node
        for i, node in enumerate(updated_nodes):
            print(f"Story node {i+1}: {node.get('description', 'No description')}")
            print(f"  - Insights: {len(node.get('insights', []))}")
            if node.get('insights'):
                for j, insight in enumerate(node['insights']):
                    print(f"    - Insight {j+1}: {insight[:100]}...")
        
        return new_state
        
    except Exception as e:
        print(f"Error assigning insights to story nodes: {e}")
        # Return state without assignments if there's an error
        return state
