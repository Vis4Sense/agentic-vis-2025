from typing_extensions import TypedDict
import yaml

class Config(TypedDict):
    dataset: str
    topic: str
    target_audience: str
    domain_knowledge: str

## TODO: for competition evaluation
# import os
# BASE_DIR = os.environ.get("SUBMISSION_PATH", os.path.dirname(os.path.abspath(__file__)))
# config_path = os.path.join(BASE_DIR, "config.yaml")
# print("Config config_path - inner point", config_path)

# TODO: remove this for competition evaluation
config_path = "config.yaml"

with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)