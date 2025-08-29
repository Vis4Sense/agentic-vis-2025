from typing_extensions import TypedDict
from typing import List, Union, Literal, Optional

from agents.vis_report.load_config import Config
from agents.vis_report.analyser.state import State as AnalysisState
from agents.vis_report.analyser.state import Visualisation


class Content(TypedDict):
    id: int
    type: Literal["introduction", "visualisation"]
    visualisation: Optional[Visualisation] = None
    facts: Optional[str] = None

class ReportSection(TypedDict):
    section_number: int
    section_name: str
    section_size: Literal["short", "medium", "long"]
    # section_goal: Literal["present", "exploratory analysis", "confirmatory analysis"]
    section_description: str
    analyses: Optional[List[AnalysisState]] = None
    content: Optional[List[Content]] = None

class GlobalFilterState(TypedDict):
    description: str
    sql_query: str
    dataset_path: str

class State(TypedDict):
    config: Config
    report_outline: List[ReportSection]
    global_filter_state: Optional[GlobalFilterState] = None
