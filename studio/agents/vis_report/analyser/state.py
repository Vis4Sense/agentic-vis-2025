from typing_extensions import TypedDict
from typing import Literal, List, Optional, Union
from agents.vis_report.load_config import Config

class InformationNeededPresent(TypedDict):
    question_text: str # your exploration or analysis question
    primary_attributes: List[str] # primary attributes to use for the analysis
    secondary_attributes: List[str] # secondary attributes to use for the analysis
    transformation: List[str] # possible transformations to apply to the data
    expected_insight_types: List[str] # expected insight types from the analysis, such as top, trend, distribution, outlier, etc.

class InformationNeededExplore(TypedDict):
    question_text: str # your exploration or analysis question
    key_uncertainty: str
    expected_outputs: List[str] # expected outputs from the exploration to resolve uncertainty

class Visualisation(TypedDict):
    library: Literal["vega-lite", "altair", "antv"]
    specification: str

class Model(TypedDict):
    method: str
    python_script: str

class Knowledge(TypedDict):
    # facts: List[
    facts: str
    insights: List[str]

class AnalysisSchema(TypedDict):
    action: Literal["present", "explore"]
    information_needed: Union[InformationNeededPresent, InformationNeededExplore]

class GlobalFilterState(TypedDict):
    description: str
    sql_query: str
    dataset_path: str

class State(TypedDict):
    analysis_schema: AnalysisSchema
    visualisation: Optional[Visualisation] = None
    knowledge: Optional[Knowledge] = None
    global_filter_state: Optional[GlobalFilterState] = None


def is_vis_valid(state: State):
    try:
        return state["visualisation"]["specification"] is not None
    except Exception as e:
        print(f"🚫 Visualisation is invalid: {e}")
        return False

def is_knowledge_valid(state: State):
    try:
        return state["knowledge"]["facts"] is not None
    except Exception as e:
        print(f"🚫 Knowledge is invalid: {e}")
        return False
