from typing_extensions import TypedDict
from typing import List, Union, Literal, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from helpers import get_llm, get_dataset_info

from agents.vis_report.load_config import Config
from agents.vis_report.planner.state import ReportSection, State
from agents.vis_report.memory import memory
from agents.vis_report.analyser.state import State as AnalysisState

from agents.vis_report.load_config import config
import time

def plan(state: State):
    print(f"▶️ Planning report")

    if config["dev"]:
        if "report_outline" in state and len(state["report_outline"]) > 0:
            return state

    new_state = state.copy()
    new_state = initiate(new_state, config)
    sections = new_state["report_outline"]

    # save the report outline to a file
    if config["dev"]:
        import json 
        with open(f"outputs_sync/vis_report/{memory.thread_id}/report_outline.json", "w", encoding="utf-8") as f:
            json.dump(new_state["report_outline"], f, indent=2, ensure_ascii=False)


    # TODO: can we do this in one go, i.e., plan all sections at once?
    start_time = time.time()
    for i, section in enumerate(sections):
        print(f"▶️ Planning section {i+1} of {len(sections)}")
        response = simple_action_plan(state, section, config)
        new_state["report_outline"][i]["analyses"] = response.analyses
        print(f"✅ Section {i+1} planned")

        memory.save_state(new_state)
    
    if config["dev"]:
        import json 
        with open(f"outputs_sync/vis_report/{memory.thread_id}/report_outline_planned.json", "w", encoding="utf-8") as f:
            json.dump(new_state["report_outline"], f, indent=2, ensure_ascii=False)

    print(f"✅ Report outline saved")
    total_time = time.time() - start_time
    print(f"⏱️ Total plan function time: {total_time:.2f} seconds")

    return new_state


def initiate(state: State, config: Config):
    """
    Initiate the report plan with the topic and dataset info
    """

    print(f"▶️ Initiating report plan for topic {config['topic']}")

    dataset_info = get_dataset_info(config["dataset"])
    llm = get_llm()

    system_message = SystemMessage(content=f"""
    You are an expert in data analysis and visualization.
    You are preparing a visualization report based on analysis of the vis publication dataset.

    The topic of the report is {config["topic"]}.

    The information about the dataset is as follows:
    {dataset_info}

    You need to plan a report for the topic using the dataset.

    The following knowledge may help you make the report:
    {config["domain_knowledge"]}

    Requirements:
    1. You will only have a limited amount of time to finish the report, so the report should be concise and to the point.
    2. Target audience is {config["target_audience"]}.
    3. The report should be interesting, coherent, insightful, and visually compelling.
    4. You can decide the number of sections, but no more than {config["max_section_number"]}.

    Response format:
    A list of report sections, each section contains:
    - section_number: the number of the section
    - section_name: the name of the section
    - section_size: the size of the section, either "short", "medium", or "long"
    - section_description: a short description of the section
    - analyses: SKIP this field for now, you will fill it in later
    - content: SKIP this field for now, you will fill it in later
    """
    )

    human_message = HumanMessage(content=f"""
    Please plan a report for the topic: {config["topic"]} using the dataset.
    """
    )

    class ResponseFormatter(BaseModel):
        report_sections: List[ReportSection]

    response = llm.with_structured_output(ResponseFormatter).invoke(
        [system_message, human_message]
    )

    new_state = state.copy()
    new_state["report_outline"] = response.report_sections

    print(f"✅ Initial report plan created")

    memory.save_state(new_state)

    return new_state


def simple_action_plan(state: State, section: ReportSection, config: Config):
    """
    Decide the action to perform for each section.

    Choices:
    1. Generate vis & insights
    2. Plan for exploratory analysis
    3. Pend
    """

    dataset_info = get_dataset_info(config["dataset"])
    llm = get_llm()

    report_outline = state["report_outline"]

    system_message = SystemMessage(content=f"""
    You are an expert in data analysis and visualization.
    You are preparing a visualization report based on analysis of the vis publication dataset.

    You have created an outline of the report and now you are working on a specific section.
    You need to design the analysis to perform for completing the section.

    Please follow the steps below to design the analysis:
    1. Decide whether to perform analysis and how many analysess
        1.1 if you decide not to perform analysis, return an empty list
        1.2 if you decide to perform analysis, continue to the next step
    2. For each analysis
        2.1. Decide the action (present or explore)
        2.2. Specify the information need based on the action you choose

    ### Instruction for deciding the number of analyses ###

    - 0 (pending analysis): you may choose pending the analysis if you can reuse the results from the other sections.
        - for example, if the section is an introduction, executive summary, synthesis, or conclusion
        - especially for the first and last sections
        - Return an empty list if you choose this
    - Otherwise, you can choose from 1 to {config["max_analyses_per_section"]} analyses, depending on the complexity of the section.
        - Each analysis should be focused on a specific question.


     ### Instructions for the action ###

    You can choose between three types of actions:
    1. Present: Generate visualisation and corresponding insights
        - If you are confident about which visualisation to use and that you can generate the vega-lite specification without further exploration, you can choose this action.
        - You don't need to know what the insights are, you will be able to analyse and generate the insights.
        - The question is whether you are confident about the visualisation to use.
    2. Explore: Plan for exploratory analysis
        - If you are not sure about the content of the section, you can choose this action.
        - You need to specify the information need (what you expect through the exploration and analysis)

    ### Instructions for the information need ###

    Specify the information need if you choose 'present' or 'explore'.
    - For 'present', basically you need to structure how you plan to design the visualisation.
        - It is a high level plan (such as the important attributes), you don't need to work on the visual design.
        - You can decide how many visualisations, but usually no more than 2.
        - There is no need to fully fulfil the planned description, you can do a trade-off and focus on the most important parts.
        - Note the visualisation will be generated by vega-lite, so it can not be too complex.
    - For 'explore', you need to specify the information need (what you expect through the exploration and analysis)
        - You can specify multiple information needs, but usually no more than 3.
        - You are encouraged to re-formulate the need as you perform the exploration. Therefore, you can start with simple need if you are not very sure.
        - Example situations that need exploration:
            - You are not sure about which filter or transformation to apply to the data.
            - Certain attributes need to be derived from the data (which can not be easily derived through vega-lite).
            - When you decide to present subset or examples but you are not sure about which subset or examples to use.
            - Others...
        - Similarly, focus on the most important parts.

    Form the information need for 'present' as follows:
    - question_text: the question you want to answer through the exploration and analysis. The question should be focused (just one specifc point).
    - primary_attributes: the primary attributes to use for visual encoding (visual channels), usually no more than 2.
    - secondary_attributes: the secondary attributes to use for visual encoding, usually no more than 2.
    - transformation: the transformation to apply to the data, can be none, usually no more than 3.
    - expected_insight_types: the expected insight types from the analysis, such as top, trend, distribution, outlier, etc.
    - parameters: the parameters for the analysis, such as the number of top items, the threshold for the trend, the threshold for the distribution, the threshold for the outlier, etc.

    Form the information need for 'explore' as follows:
    - question_text: the question you want to answer through the exploration and analysis. The question should be focused (just one specifc point).
    - key_uncertainty: if you choose 'explore', specify the key uncertainty you have about the data, which reflects the main goal of the exploration.
    - expected_outputs: the expected outputs from the exploration to resolve uncertainty.
        - Should be practical, such as parameter values for filtering, transformation; or important insights that help you decide the next step.
        - Consider the limit of tools that you can use: you may use pandas and simple networkx, but time-consuming tasks or machine learning tasks are not recommended.
        - Usually no more than 3.

    Please make the information need as simple and focused as possible.


    ------------------------------------------------------------
    Below is the information of the section and the overall outline of the report.
    
    Section information:
    {section}

    Overall outline:
    {';'.join([f"{section['section_number']}. {section['section_name']}" for section in report_outline])}

    The information of the dataset that you will analyse is as follows:
    {dataset_info}


   
    The following knowledge may help you make the decision:

    Domain knowledge:
    {config["domain_knowledge"]}

    Requirements:
    1. You don't need to fully follow the section description, there is a limit of time and space, so focus on the most important parts.
    2. Be simple and concise. If you feel confident about which visualisation to use, please choose present.
    3. leave 'visualisation' and 'knowledge' fields empty for now, you will fill them in later.
    """
    )

    human_message = HumanMessage(content=f"""
    Please design the analysis to perform for the section.
    """
    )

    class ResponseFormatter(BaseModel):
        analyses: List[AnalysisState]

    response = llm.with_structured_output(ResponseFormatter).invoke(
        [system_message, human_message]
    )

    return response
