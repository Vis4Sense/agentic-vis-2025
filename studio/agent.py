from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

import csv
from helpers import get_llm
from report_html import generate_html_report
from report_pdf import generate_pdf_report

class State(TypedDict):
    message: str
    dataset_info: str

def generate_msg(state: State):
    dataset_info = state["dataset_info"]
    # if the prompt is to generate Vega-Lite charts, then specify in sys_prompt and use generate_html_report()
    sys_prompt = f"Please generate Vega-Lite graphs to visualize insights from the dataset, output should be graphs and narrative: {dataset_info}"
   
    # if the prompt is to generate Python codes, then specify in sys_prompt and use generate_pdf_report()
    # sys_prompt = f"Please generate Python code to visualize insights from the dataset, output should be graphs and narrative: {dataset_info}"
    
    # get the LLM instance
    llm = get_llm(temperature=0, max_tokens=4096)

    # generate the response
    response = llm.invoke(
        [SystemMessage(content=sys_prompt), HumanMessage(content="Generate a response.")]
    )
    return {"message": response}


def create_workflow():
    # create the agentic workflow using LangGraph
    builder = StateGraph(State)
    builder.add_node("generate_msg", generate_msg)
    builder.add_edge(START, "generate_msg")
    builder.add_edge("generate_msg", END)
    return builder.compile()

class Agent:
    def __init__(self):
        self.workflow = None

    def initialize(self):
        self.workflow = create_workflow()

    def initialize_state_from_csv(self) -> dict:
        # The dataset should be first input to the agentic configuration, and it should be generalizable to any dataset
        path = "./dataset.csv"
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)
            first_row = next(reader)

        attributes = ", ".join(header)
        example_values = "\t".join(first_row)

        # if the final output contains Vega-Lite codes, then use the github hosted dataset rather than the local dataset
        file_name = "https://raw.githubusercontent.com/demoPlz/mini-template/main/studio/dataset.csv"
        # file_name = "dataset.csv"
        example_input = f"""
            There is a dataset, there are the following {len(header)} attributes:
            {attributes}
            Name of csv file is {file_name}
        """
        state = {
            "dataset_info": str(example_input)
        }
        return state
    def decode_output(self, output: dict):
        # if the final output contains Vega-Lite codes, then use generate_html_report
        # if the final output contains Python codes, then use generate_pdf_report

        # generate_pdf_report(output, "output.pdf")
        generate_html_report(output, "output.html")
    def process(self):

        if self.workflow is None:
            raise RuntimeError("Agent not initialised. Call initialize() first.")
        
        # initialize the state & read the dataset
        state = self.initialize_state_from_csv()

        # invoke the workflow
        output_state = self.workflow.invoke(state)
        print(output_state)

        # flatten the output
        def _flatten(value):
            return getattr(value, "content", value)
        result = {k: _flatten(v) for k, v in output_state.items()}

        # decode the output
        self.decode_output(result)

        # return the result
        return result