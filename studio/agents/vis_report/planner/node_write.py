from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from typing import List
import json

from helpers import get_llm, get_dataset_info

from agents.vis_report.planner.state import State
from agents.vis_report.analyser.state import is_vis_valid, is_knowledge_valid
from agents.vis_report.memory import memory

from agents.vis_report.load_config import config


def write_content(state: State):
    """
    Write the content of the report.
    """

    print(f"📝 Writing content...")

    new_state = state.copy()
    sections = new_state["report_outline"]
    simplified_outline = []

    # generate captions for successful analysis
    for section in sections:
        print(f"🔎 Analyzing section {section['section_number']}...")
        analyses = section["analyses"]

        content = [{
            "id": 0,
            "type": "introduction",
        }]

        content_simplified = [{
            "id": 0,
            "type": "introduction",
        }]

        section["content"] = content
        section["content_simplified"] = content_simplified

        if len(analyses) == 0:
            continue

        for analysis in analyses:
            if not is_vis_valid(analysis):
                continue
            if not is_knowledge_valid(analysis):
                continue

            vis_data = analysis["visualisation"]["specification"]
            if isinstance(vis_data, str) and len(vis_data) > 1000:
                vis_data = vis_data[:1000] + "...[truncated]"
            print(vis_data)

            content_simplified.append({
                "id": len(content),
                "type": "visualisation",
                "visualisation": vis_data,
                "facts": analysis["knowledge"]["facts"],
            })

            content.append({
                "id": len(content),
                "type": "visualisation",
                "visualisation": analysis["visualisation"],
                "facts": analysis["knowledge"]["facts"],
            })
            section["content"] = content
            section["content_simplified"] = content_simplified

    for section in sections:
        simplified_outline.append({
            "section_number": section["section_number"],
            "section_name": section["section_name"],
            "section_size": section["section_size"],
            "section_description": section["section_description"],
            "content": section["content_simplified"],
            # "content": section["content"],
        })

    system_message = SystemMessage(content=f"""
    You are an expert in data analysis and visualization.
    You are preparing a visualization report based on analysis of the vis publication dataset.

    Please generate textual description for the report.

    You will be given an outline of the report, each section has a list of content.
    You need to generate textual description for each content.

    - if the content type is "introduction", you need to generate a textual description.
        - for introduction or summary sections, this also includes filling the section content by synthesising the results from the other sections
    - if the content type is "visualisation", you need to generate a textual description of the visualisation.
        - Your description should catch the key insights and be faithful to the data facts.

    Background information:
    - Topic of the report: {config["topic"]}
    - Target audience: {config["target_audience"]}

    Requirements:
    - Your report should be coherent and insightful.
    - Style of writing: easy to understand and engaging.
    - The content should be a natural paragraph, don't begin with phrases.
    """)

    human_message = HumanMessage(content=f"""
    {simplified_outline}
    """)

    class GeneratedContent(TypedDict):
        section_number: int
        content_id: int
        text: str

    class ResponseFormatter(BaseModel):
        generated_content: List[GeneratedContent]

    llm = get_llm()

    response = llm.with_structured_output(ResponseFormatter).invoke(
        [system_message, human_message]
    )

    print(f"✅ Response: {response}")

    for content in response.generated_content:
        try:
            section = next(s for s in sections if s["section_number"] == content["section_number"])
            content_index = next(i for i, c in enumerate(section["content"]) if c["id"] == content["content_id"])
            section["content"][content_index]["text"] = content["text"]
        except StopIteration:
            print(f"❌ Section {content['section_number']} not found")
            continue

    memory.save_state(new_state)

    return new_state


