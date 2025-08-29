from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import json

from helpers import get_llm, get_dataset_info, query_by_sql
from agents.simple_iteration.state import State
from agents.simple_iteration.memory import shared_memory

class ResponseFormatter(BaseModel):
    insights: list[str] = Field(description="The insights about the dataset")
    should_continue: bool = Field(description="Whether to continue with a follow-up analysis")


def get_insights(state: State):
    """
    Generate insights about the dataset and decide whether to continue
    """
    
    # Increment iteration counter
    current_iteration = state.get("iteration_count", 0) + 1
    max_iterations = state.get("max_iterations", 3)
    
    # Get path of the main program being executed
    dataset_info = get_dataset_info(state['select_data_state']['dataset_path'])


    # Get previous questions and insights from iteration history
    previous_questions = []
    previous_insights = []
    
    if state.get("iteration_history"):
        for iteration in state["iteration_history"]:
            if "question" in iteration and iteration["question"]:
                previous_questions.append(iteration["question"]["question"])
            if "insights" in iteration and iteration["insights"]:
                previous_insights.extend(iteration["insights"])

    questions_text = "\n".join([f"- {q}" for q in previous_questions])
    insights_text = "\n".join([f"- {i}" for i in previous_insights])

    sys_prompt = f"""
        You are a data analyst who get insights from the dataset.

        The dataset is as follows:
        {dataset_info}

        Please summarize the insights about the dataset regarding the analysis question and topic provided by the user.

        Here are some facts:
        {state['facts']['stdout']}

        Requirements:
        1. the insights should be faithful to the facts provided, but not directly the fact, an idea form is higher level summary combined with example facts or details.
        2. the insights should be concise and to the point.
        3. the insights should be relevant to the user's topic and question.
        4. don't include those are not insights.
        
        IMPORTANT: After generating insights, you need to decide whether to continue exploring with a follow-up analysis.
        Consider:
        - Current iteration: {current_iteration}/{max_iterations}
        - existing questions:
        {questions_text}
        - existing insights:
        {insights_text}
        - Whether there are still unexplored aspects of the topic
        - Whether the current insights suggest new directions for analysis
        - Whether you've reached a satisfactory level of understanding
    """

    human_prompt = f"""
    I would like to explore the dataset about the topic of {state['topic']}.
    The current analysis question is: {state['question']['question']}.
    This is iteration {current_iteration} out of {max_iterations}.
    Please generate the insights and decide whether to continue with a follow-up analysis.
    """

    llm = get_llm(temperature=0, max_tokens=8192)

    response = llm.with_structured_output(ResponseFormatter).invoke(
        [SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)]
    )

    # Decide whether to continue based on iteration count and insights
    # If this is the last iteration, always set should_continue to False to trigger storyline generation
    if current_iteration >= max_iterations:
        should_continue = False
    else:
        should_continue = response.should_continue and len(response.insights) > 0

    new_state = state.copy()
    new_state["insights"] = response.insights
    new_state["iteration_count"] = current_iteration
    new_state["should_continue"] = should_continue
    
    # Save current iteration data to history
    current_iteration_data = {
        "question": state["question"],
        "facts": state["facts"],
        "insights": response.insights
    }
    new_state["iteration_history"] = state.get("iteration_history", []) + [current_iteration_data]

    # Save the state to memory
    shared_memory.save_state(new_state)
    print(f"state saved to memory for thread {shared_memory.thread_id}")
    print(f"Iteration {current_iteration}/{max_iterations}, continuing: {should_continue}")

    return new_state
