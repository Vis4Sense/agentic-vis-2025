from typing_extensions import TypedDict, List
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import json
import pandas as pd
from pathlib import Path

from helpers import get_llm, get_dataset_info
from agents.simple_iteration.state import State, Visualization
from agents.simple_iteration.memory import shared_memory

class StoryVisualizationDesign(BaseModel):
    insight: str = Field(description="The insight this visualization represents")
    chart_type: str = Field(description="Type of chart (bar, line, scatter, etc.)")
    description: str = Field(description="Brief description of what the visualization shows")
    spec: str = Field(description="Vega-Lite specification as a JSON string")
    is_appropriate: bool = Field(description="Whether this insight can be effectively visualized")

class StoryVisualizationResponse(BaseModel):
    visualizations: List[StoryVisualizationDesign] = Field(description="List of visualization designs for the story node")

def generate_story_visualizations(state: State):
    """
    Generate visualizations for each story node in the storyline.
    For each story node, generate visualizations based on its description and related insights.
    """
    
    # Get dataset information
    dataset_info = get_dataset_info("dataset.csv")
    
    # Get storyline from state
    storyline = state.get("storyline", {})
    if not storyline or "nodes" not in storyline:
        print("No storyline available for visualization generation")
        return state
    
    story_nodes = storyline["nodes"]
    if not story_nodes:
        print("No story nodes available for visualization generation")
        return state
    
    # Collect facts from all iterations
    all_facts = []
    
    # Add current facts
    current_facts = state.get("facts", {})
    if current_facts and current_facts.get("stdout"):
        all_facts.append(current_facts["stdout"])
    
    # Add facts from iteration history
    iteration_history = state.get("iteration_history", [])
    for iteration in iteration_history:
        if "facts" in iteration and iteration["facts"] and iteration["facts"].get("stdout"):
            all_facts.append(iteration["facts"]["stdout"])
    
    # Combine all facts
    facts_text = "\n\n".join(all_facts) if all_facts else "No facts available"
    
    print(f"Generating visualizations for {len(story_nodes)} story nodes")
    print(f"Using facts from {len(all_facts)} iterations")
    
    # Create system prompt for story visualization design
    sys_prompt = f"""
    You are a data visualization expert who designs effective visualizations for story nodes in a data exploration narrative.
    
    The dataset information is as follows:
    {dataset_info}
    
    The facts from data analysis are as follows:
    {facts_text}
    
    Your task is to design visualizations for each story node. For each story node:
    1. Analyze the story node description to understand what aspect of the data it explores
    2. Review the facts to understand what data is available and what patterns were found
    3. Determine what type of visualization would best represent this aspect using the actual data
    4. Design an appropriate chart type and Vega-Lite specification based on the facts
    5. Focus on visualizations that support the narrative flow
    
    Rules for story visualization design:
    1. Use simple, effective chart types: bar charts, line charts, scatter plots, heatmaps
    2. Focus on the most important aspects of the story node
    3. Use Vega-Lite specification format (JSON)
    4. Make sure the visualization is relevant to the story node description AND the facts
    5. Keep the specification simple and readable
    6. Use appropriate encodings for the data types
    7. Include proper titles and labels
    8. Consider the narrative flow - each visualization should build on the previous ones
    9. Base your visualizations on the actual data patterns found in the facts
    
    Available chart types:
    - Bar charts: for categorical comparisons
    - Line charts: for temporal trends
    - Scatter plots: for correlations
    - Heatmaps: for matrix data
    - Pie charts: for proportions (use sparingly)
    
    IMPORTANT: 
    - Create visualizations that will help understand the story node.
    - The spec field should contain a valid Vega-Lite JSON specification.
    - data source file name: file_name = "https://raw.githubusercontent.com/demoPlz/mini-template/main/studio/dataset.csv"
    - Each story node should have at least one visualization that represents its main concept.
    - Use the facts to ensure your visualizations are based on real data patterns, not assumptions.
    """
    
    # Create human prompt with story nodes
    story_nodes_text = "\n".join([f"{i+1}. {node['description']}" for i, node in enumerate(story_nodes)])
    
    human_prompt = f"""
    Please design visualizations for the following story nodes:
    
    {story_nodes_text}
    
    For each story node, provide:
    1. The story node description
    2. Chart type that best represents this aspect based on the facts
    3. Vega-Lite specification as JSON string that uses the actual data patterns
    4. Description of what the visualization shows
    
    Focus on creating visualizations that support the narrative and help readers understand each aspect of the data exploration.
    The Vega-Lite specification should be a valid JSON string that can be parsed and used directly.
    
    IMPORTANT: Use the facts provided to ensure your visualizations are based on real data patterns and findings, not assumptions.
    """
    
    llm = get_llm(temperature=0, max_tokens=8192)
    
    try:
        response = llm.with_structured_output(StoryVisualizationResponse).invoke(
            [SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)]
        )
        
        print(f"Generated {len(response.visualizations)} story visualization designs")
        
        # Convert to state format and update state
        new_state = state.copy()
        
        # Update storyline with visualizations
        updated_storyline = storyline.copy()
        updated_nodes = []
        
        for i, story_node in enumerate(story_nodes):
            # Find corresponding visualization design
            if i < len(response.visualizations):
                vis_design = response.visualizations[i]
                
                visualization = {
                    "insight": story_node["description"],  # Use story node description as insight
                    "chart_type": vis_design.chart_type,
                    "description": vis_design.description,
                    "is_appropriate": vis_design.is_appropriate,
                    "image_path": "",  # Will be populated if visualization is generated
                    "spec": vis_design.spec  # Vega-Lite specification
                }
                
                updated_node = story_node.copy()
                updated_node["visualizations"] = [visualization]
                updated_nodes.append(updated_node)
            else:
                # If no visualization design available, keep original node
                updated_nodes.append(story_node)
        
        updated_storyline["nodes"] = updated_nodes
        new_state["storyline"] = updated_storyline
        
        # Save state to memory
        shared_memory.save_state(new_state)
        print(f"Story visualizations saved to memory for thread {shared_memory.thread_id}")
        
        # Print summary
        total_visualizations = sum(len(node.get("visualizations", [])) for node in updated_nodes)
        print(f"Story visualization summary: {total_visualizations} visualizations generated for {len(updated_nodes)} story nodes")
        
        # Debug: Print details of each story node
        for i, node in enumerate(updated_nodes):
            print(f"Story node {i+1}: {node.get('description', 'No description')}")
            print(f"  - Insights: {len(node.get('insights', []))}")
            print(f"  - Visualizations: {len(node.get('visualizations', []))}")
            if node.get('visualizations'):
                for j, vis in enumerate(node['visualizations']):
                    print(f"    - Vis {j+1}: {vis.get('chart_type', 'Unknown')} - {vis.get('is_appropriate', False)}")
        
        return new_state
        
    except Exception as e:
        print(f"Error generating story visualizations: {e}")
        # Return state without story visualizations if there's an error
        return state
