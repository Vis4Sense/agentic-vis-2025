import os
import csv
from pathlib import Path

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from openai import OpenAI, AzureOpenAI
import pandas as pd
import pandasql as ps

from dotenv import load_dotenv

from agents.vis_report.load_config import config

load_dotenv()

class _LLMProxy:
    def __init__(self, base, force_fc: bool):
        self._base = base
        self._force_fc = force_fc
    def with_structured_output(self, schema, **kwargs):
        if self._force_fc and "method" not in kwargs:
            kwargs["method"] = "function_calling"
        return self._base.with_structured_output(schema, **kwargs)
    def __getattr__(self, name):
        return getattr(self._base, name)

def get_llm(**kw):
    """
    Return a Chat‑compatible LLM whose backend (OpenAI, Azure, local stub…)
    is selected by env‑vars.  Extra **kw flow through so nodes can override
    temperature, max_tokens, etc. without knowing the backend.

    Mini challenge evaluation server uses azure openai to run your submission.
    You don't need to fill in the azure openai endpoint and api key, 
    but you need to fill in the openai api key and model name to run locally.
    """
    provider = os.getenv("LLM_PROVIDER", "openai")

    if provider.lower() == "azure":

        # transform parameters if using gpt-5-mini (till 12 aug 2025)
        if "gpt-5-mini" in os.getenv("AZURE_OPENAI_DEPLOYMENT"):
            # 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead
            if "max_tokens" in kw:
                kw["max_completion_tokens"] = kw.pop("max_tokens")
            # 'temperature' does not support 0.0 with this model. Only the default (1) value is supported.
            if "temperature" in kw:
                del kw["temperature"]

        llm = AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini"),
            # For submission, the default value is always gpt-4o, but you can choose from o1, o3 and o4-mini too.
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            **kw,
        )
        return _LLMProxy(llm, force_fc=True)
    elif provider.lower() == "local-echo":
        from langchain.llms.fake import FakeListLLM
        return FakeListLLM(responses=["This is a stub."])  
    else:  
        return ChatOpenAI(
            api_key =os.getenv("OPENAI_API_KEY"),
            model_name=os.getenv("OPENAI_MODEL", "gpt-4o"),
            **kw,
        )

def find_dataset_path(filename=config['dataset']):
    """
    Find the path of the dataset file
    """
    cwd = Path.cwd()
    
    possible_paths = [
        cwd / filename,
        Path(__file__).parent / filename,
        Path(__file__).parent.parent / filename,
        cwd / "studio" / filename,
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    raise FileNotFoundError(f"Could not find {filename} in any of the expected locations")


def load_dataset(dataset_path: str):
    """
    Load the dataset from the path
    """
    return pd.read_csv(dataset_path)


def get_dataset_info(dataset_path: str):
    """
    Get the information of the dataset
    """

    num_example_rows = 3

    with open(dataset_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
    
        # first five rows
        rows = [next(reader) for _ in range(num_example_rows)]

    attributes = ", ".join(header)
    example_values = "\n".join("\t".join(row) for row in rows)

    dataset_info = f"""
        There is a dataset, there are the following {len(header)} attributes:
        {attributes}

        Here are the first {num_example_rows} rows of the dataset:
        {example_values}
    """
    return dataset_info


def query_by_sql(sql_query: str):
    """
    Query the dataset by the SQL query
    """
    dataset_path = find_dataset_path()
    Papers = load_dataset(dataset_path)
    return ps.sqldf(sql_query, locals())

if __name__ == "__main__":
    sql_query = """
    SELECT\n  Conference,\n  Year,\n  Title,\n  DOI,\n  Link,\n  FirstPage,\n  LastPage,\n  PaperType,\n  Abstract,\n  \"AuthorNames-Deduped\" AS AuthorNamesDeduped,\n  \"AuthorNames\",\n  \"AuthorAffiliation\",\n  InternalReferences,\n  AuthorKeywords,\n  AminerCitationCount,\n  CitationCount_CrossRef,\n  PubsCited_CrossRef,\n  Downloads_Xplore,\n  Award,\n  GraphicsReplicabilityStamp,\n  (CASE WHEN lower(Title) LIKE '%sensemak%' THEN 'title' ELSE '' END)\n   || (CASE WHEN lower(Abstract) LIKE '%sensemak%' THEN ';abstract' ELSE '' END)\n   || (CASE WHEN lower(AuthorKeywords) LIKE '%sensemak%' THEN ';keywords' ELSE '' END)\n   || (CASE WHEN lower(InternalReferences) LIKE '%sensemak%' THEN ';internal_references' ELSE '' END) AS MatchFields\nFROM Papers\nWHERE lower(Title) LIKE '%sensemak%'\n   OR lower(Abstract) LIKE '%sensemak%'\n   OR lower(AuthorKeywords) LIKE '%sensemak%'\n   OR lower(InternalReferences) LIKE '%sensemak%'\nORDER BY Year ASC, CitationCount_CrossRef DESC;
    """
    data_selected = query_by_sql(sql_query)
    data_selected.to_csv("dataset_selected.csv", index=False)
