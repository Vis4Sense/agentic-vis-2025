from agents.simple_iteration.state import State
from helpers import get_dataset_info
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from helpers import get_llm
from agents.simple_iteration.memory import shared_memory

class ResponseFormatter(BaseModel):
    question: str

def question(state: State):
    """
    Generate a question based on the topic and selected dataset
    """

    selected_dataset_path = state["select_data_state"]["dataset_path"]
    dataset_info = get_dataset_info(selected_dataset_path)

    questions, _ = shared_memory.export_questions_and_insights()

    context = ""
    if state["iteration_count"] > 0:
        context = f"""
        Here are the previous questions:
        {questions}

        Here are the insights generated in the last iteration:
        {state["insights"]}

        Please pick a follow-up question based on the previous questions and insights.
        The follow-up question should be focused and operationalizable with not very complex code.
        For example, it can be a further analysis of specific insights that you find interesting.
    """

    sys_prompt = f"""
        You are a helpful assistant that generate analysis questions to explore the dataset.

        In previous analysis, the dataset has been selected based on the topic of {state["topic"]}.
        The query to select the dataset is to {state['select_data_state']['description']}.

        Here are the information of the selected dataset:
        {dataset_info}

        Please generate the most relevant question to explore the dataset.
        
        Rules:
        1. the question should be focused and relevant
        2. the question should be a subtask that is operationalizable

        {context}
    """

    human_prompt = f"Please generate the question."

    llm = get_llm(temperature=0, max_tokens=4096)

    response = llm.with_structured_output(ResponseFormatter).invoke(
        [SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)]
    )

    new_state = state.copy()
    
    new_state["question"] = {
        "question": response.question,
        "handled": False,
        "spec": ""
    }

    # save the state to memory
    shared_memory.save_state(new_state)
    print(f"state saved to memory for thread {shared_memory.thread_id}")

    return new_state
