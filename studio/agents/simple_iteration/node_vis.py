from typing_extensions import TypedDict, List
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import json
import pandas as pd
from pathlib import Path

from helpers import get_llm, get_dataset_info
from agents.simple_iteration.state import State, Visualization
from agents.simple_iteration.memory import shared_memory

class VisualizationDesign(BaseModel):
    insight: str = Field(description="The insight this visualization represents")
    chart_type: str = Field(description="Type of chart (bar, line, scatter, etc.)")
    description: str = Field(description="Brief description of what the visualization shows")
    spec: str = Field(description="Vega-Lite specification as a JSON string")
    is_appropriate: bool = Field(description="Whether this insight can be effectively visualized")

class VisualizationResponse(BaseModel):
    visualizations: List[VisualizationDesign] = Field(description="List of visualization designs for the insights")

def generate_visualizations(state: State):
    """
    Generate visualizations for each insight in the state.
    For each insight, attempt to design an appropriate visualization.
    """
    
    # Get dataset information
    dataset_info = get_dataset_info("dataset.csv")
    
    # Get insights from state
    insights = state.get("insights", [])
    
    if not insights:
        print("No insights available for visualization generation")
        return state
    
    print(f"Generating visualizations for {len(insights)} insights")
    
    # Create system prompt for visualization design
    sys_prompt = f"""
    You are a data visualization expert who designs effective visualizations for data insights.
    
    The dataset information is as follows:
    {dataset_info}
    
    Your task is to design visualizations for each insight provided. For each insight:
    1. Determine if the insight can be effectively visualized
    2. If yes, design an appropriate chart type and Vega-Lite specification
    3. If no, mark it as not appropriate
    
    Rules for visualization design:
    1. Use simple, effective chart types: bar charts, line charts, scatter plots, heatmaps
    2. Focus on the most important aspects of the insight
    3. Use Vega-Lite specification format (JSON)
    4. Make sure the visualization is relevant to the insight
    5. Keep the specification simple and readable
    6. Use appropriate encodings for the data types
    7. Include proper titles and labels
    
    Available chart types:
    - Bar charts: for categorical comparisons
    - Line charts: for temporal trends
    - Scatter plots: for correlations
    - Heatmaps: for matrix data
    - Pie charts: for proportions (use sparingly)
    
    IMPORTANT: 
    - Only create visualizations that will actually help understand the insight.
    - If an insight is too abstract, qualitative, or doesn't have clear visualizable data,
      mark it as not appropriate (is_appropriate: false).
    - The spec field should contain a valid Vega-Lite JSON specification.
    - data source file name: file_name = "https://raw.githubusercontent.com/demoPlz/mini-template/main/studio/dataset.csv"
    """
    
    # Create human prompt with insights
    insights_text = "\n".join([f"{i+1}. {insight}" for i, insight in enumerate(insights)])
    
    human_prompt = f"""
    Please design visualizations for the following insights:
    
    {insights_text}
    
    For each insight, determine if it can be effectively visualized and provide:
    1. The insight text
    2. Chart type (if appropriate)
    3. Vega-Lite specification as JSON string (if appropriate)
    4. Description of what the visualization shows
    5. Whether the insight is appropriate for visualization
    
    Focus on insights that have clear, quantifiable data that can be meaningfully represented visually.
    The Vega-Lite specification should be a valid JSON string that can be parsed and used directly.
    """
    
    llm = get_llm(temperature=0, max_tokens=8192)
    
    try:
        response = llm.with_structured_output(VisualizationResponse).invoke(
            [SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)]
        )
        
        print(f"Generated {len(response.visualizations)} visualization designs")
        
        # Convert to state format and update state
        new_state = state.copy()
        
        # Convert visualization designs to state format
        visualizations_list = []
        for vis_design in response.visualizations:
            visualization = {
                "insight": vis_design.insight,
                "chart_type": vis_design.chart_type,
                "description": vis_design.description,
                "is_appropriate": vis_design.is_appropriate,
                "image_path": "",  # Will be populated if visualization is generated
                "spec": vis_design.spec  # Vega-Lite specification
            }
            visualizations_list.append(visualization)
        
        # Add visualizations to state
        new_state["visualizations"] = visualizations_list
        
        # Save state to memory
        shared_memory.save_state(new_state)
        print(f"State saved to memory for thread {shared_memory.thread_id}")
        
        # Print summary
        appropriate_count = sum(1 for vis in visualizations_list if vis["is_appropriate"])
        print(f"Visualization summary: {appropriate_count}/{len(visualizations_list)} insights can be visualized")
        
        return new_state
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        # Return state without visualizations if there's an error
        return state


