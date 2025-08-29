from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import json

from helpers import get_llm, get_dataset_info, query_by_sql
from agents.simple_iteration.state import State
from agents.simple_iteration.memory import shared_memory

class ResponseFormatter(BaseModel):
    follow_up_question: str = Field(description="The follow-up question to explore based on the insights")
    should_reselect_data: bool = Field(description="Whether to re-select data for the new question")
    reasoning: str = Field(description="Reasoning for the decision")


def follow_up_decision(state: State):
    """
    Decide on follow-up question and whether to re-select data based on insights
    """
    
    # Get current iteration info
    current_iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    
    # Get dataset info
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
        You are a data analyst who decides on follow-up questions and data selection strategy based on insights.

        The dataset is as follows:
        {dataset_info}

        Current topic: {state['topic']}
        Current iteration: {current_iteration}/{max_iterations}

        Previous questions asked:
        {questions_text}

        Previous insights generated:
        {insights_text}

        Latest insights from current analysis:
        {state['insights']}

        Your task is to:
        1. Generate a follow-up question based on the insights that would provide deeper understanding
        2. Decide whether the current dataset is sufficient for the follow-up question or if new data selection is needed

        Rules for follow-up question:
        - Should be focused and operationalizable
        - Should build upon the insights from previous analysis
        - Should explore new aspects or deeper analysis of existing findings
        - Should be relevant to the overall topic
        - should not be too complex, keep it focused and concise
        - Considering the data set info, the analysis should be possible to be done with the current dataset.

        Rules for data selection decision:
        - If the follow-up question requires different data than what's currently selected, choose to re-select
        - If the current dataset can answer the follow-up question, don't re-select
        - Consider whether the current SQL query and dataset are sufficient for the new question

        IMPORTANT: Consider the iteration limit ({current_iteration}/{max_iterations}) when making decisions.
    """

    human_prompt = f"""
    Based on the insights from the current analysis, please decide on a follow-up question and whether to re-select data.
    The current analysis question was: {state['question']['question']}.
    """

    llm = get_llm(temperature=0, max_tokens=8192)

    response = llm.with_structured_output(ResponseFormatter).invoke(
        [SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)]
    )

    new_state = state.copy()
    
    # Update the question with the follow-up question
    new_state["question"] = {
        "question": response.follow_up_question,
        "handled": False,
        "spec": ""
    }
    
    # Add decision info to state
    new_state["follow_up_decision"] = {
        "should_reselect_data": response.should_reselect_data,
        "reasoning": response.reasoning
    }

    # Save the state to memory
    shared_memory.save_state(new_state)
    print(f"state saved to memory for thread {shared_memory.thread_id}")
    print(f"Follow-up question: {response.follow_up_question}")
    print(f"Should re-select data: {response.should_reselect_data}")
    print(f"Reasoning: {response.reasoning}")

    return new_state
