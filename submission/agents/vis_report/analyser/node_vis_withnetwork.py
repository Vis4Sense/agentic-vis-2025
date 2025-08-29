from typing_extensions import TypedDict
from typing import List, Union, Literal, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from helpers import get_llm, get_dataset_info

from agents.vis_report.analyser.state import State, Visualisation
from agents.vis_report.analyser.memory import memory
from agents.vis_report.memory import memory as global_memory
from agents.vis_report.load_config import config
import json
import os
import uuid
import pandas as pd
import altair as alt
import subprocess
import tempfile
import os

# from agents.vis_report.analyser.sandbox import run_in_sandbox

def visualise(state: State):
    """
    Visualise the data based on the information need.
    """

    if config["dev"]:
        if "visualisation" in state and state["visualisation"]:
            return state

    new_state = state.copy()
    question = state["analysis_schema"]["information_needed"]["question_text"]
    if is_network_analysis_llm(question):
        print("🔍 Network analysis")
        visualisation = get_antv_visualisation(state)
        new_state["visualisation"] = visualisation
    else:
        print("🔍 Vega-lite visualisation")
        # visualisation = get_vega_lite_spec(state)
        # visualisation = get_vega_lite_spec_inline(state)
        # new_state["visualisation"] = visualisation.visualisation

        # visualisation = get_altair_visualisation(state)
        visualisation = get_altair_visualisation_sandbox(state)
        new_state["visualisation"] = visualisation
    memory.add_state(new_state)
    return new_state

def is_network_analysis(question_text: str):
    network_keywords = ["co-authorship network"]
    return any(keyword in question_text.lower() for keyword in network_keywords)

def is_network_analysis_llm(question_text: str, dataset_info: str = None):
    """
    Use LLM to determine if network analysis is needed.
    """
    llm = get_llm()
    
    system_message = SystemMessage(content=f"""
    You are a data analysis expert, you need to determine if the user's question needs co-authorship network analysis to answer.
    
    Network analysis is suitable for the following cases:
    1. Analyse the co-authorship relationship, connection or interaction of entities
    2. Study co-authorship, reference, social network structure
    3. Discover co-authorship communities, clusters or patterns
    4. Analyse co-authorship centrality, influence, etc.

    It is not suitable for the following cases:
    1. Analyse the temporal relationship, connection or interaction of entities
    2. Analyse the spatial relationship, connection or interaction of entities
    
    
    Please analyse the user's question, determine if network analysis is needed. Only answer "yes" or "no".
    """)
    
    human_message = HumanMessage(content=f"User question: {question_text}")
    
    response = llm.invoke([system_message, human_message])
    return "yes" in response.content.lower()

def get_antv_visualisation(state: State):
    """
    Get the antv visualisation for the network analysis.
    """
    from agents.vis_report.analyser.network import llm_filter, construct_network, filter_network, graph_container
    topic = config["topic"]
    file_path = config["dataset"]
    G, df = construct_network(file_path)
    filtered_G, filtered_df = G, df

    if state["global_filter_state"]:
        print("Using global filter processed dataset ...")
        filtered_file_path = state["global_filter_state"]["dataset_path"]
        response = llm_filter(topic, file_path, filtered_file_path, config["domain_knowledge"])
        filters = response.filters
        print('LLM generatedfilters: ', filters)
        filtered_G, filtered_df = filter_network(G, filtered_file_path, filters)
    else:
        print("Using original dataset with LLM filter...") 
        response = llm_filter(topic, file_path, '', config["domain_knowledge"])
        filters = response.filters
        print('LLM generatedfilters: ', filters)
        filtered_G, filtered_df = filter_network(G, df, filters)

    print('number of nodes before filtering: ', len(G.nodes()))
    print('number of nodes after filtering: ', len(filtered_G.nodes()))
    print('filtered_df: ', filtered_df.shape)

    nodes_data = [{"id": node} for node in filtered_G.nodes()]
    edges_data = [{"source": u, "target": v, "value": filtered_G[u][v]["weight"], "filtered": filtered_G[u][v]["filtered"] if "filtered" in filtered_G[u][v] else True} 
                  for u, v in filtered_G.edges()]
    network_json = json.dumps({
        "nodes": nodes_data,
        "edges": edges_data
    })

    def graph_html(network_json: dict) -> str:
        """
        Get the HTML for the network
        """
        container_id = "network_" + str(uuid.uuid4())

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
        </head>
        <body>
            {graph_container(container_id, network_json, width='100%', height='100%')}
        </body>
        </html>
        """
        return html
    
    html = graph_html(network_json)
    visualisation = Visualisation(
        library="antv",
        specification=html
    )

    return visualisation

def get_vega_lite_spec_simple(state: State):
    """
    Get the vega-lite specification for the visualisation.
    """
    llm = get_llm(temperature=0.0)
    dataset_info = get_dataset_info(config["dataset"])

    system_message = SystemMessage(content=f"""
    You are an expert in creating vega-lite specifications for visualisations.

    Use this dataset: {config["dataset_url"]}

    The dataset information is as follows:
    {dataset_info}

    The task is to generate a vega-lite specificiation for the following information, but not necessary to follow the information strictly:
    {state["analysis_schema"]["information_needed"]}

    """)

    human_message = HumanMessage(content=f"""
    Please generate the vega-lite specification for the visualisation. Robustness is prioritised over complexity. Do not generate more than 2 subplots.
    """
    )

    class ResponseFormatter(BaseModel):
        visualisation: Visualisation

    response = llm.with_structured_output(ResponseFormatter).invoke([system_message, human_message])
    return response


def get_vega_lite_spec(state: State):
    """
    Get the vega-lite specification for the visualisation.
    """
    llm = get_llm(temperature=0.0)
    dataset_info = get_dataset_info(config["dataset"])

    system_message = SystemMessage(content=f"""
    You are an expert in creating vega-lite specifications for visualisations.

    Use this dataset: {config["dataset_url"]}

    The dataset information is as follows:
    {dataset_info}

    The visualisation task is to answer this question:
    {state["analysis_schema"]["information_needed"]["question_text"]}

    You might refer to the information when you designing and generating the visualisation, but not necessary to follow the information strictly as these are just suggestions and can be too complex:
    {state["analysis_schema"]["information_needed"]}

    Requirements:
    1. Generate valid vega-lite specification that can be rendered by Vega-Lite, not violating the critical expression syntax rules and not using old fashion syntax.
    2. Robustness is prioritised over complexity. You should generate one subplot, no more than 2 subplots.
    3. If the information need is too complex (e.g., the question requires more than 3 levels of data transformation), you don't need to fulfil the complete need. You may generate a visualisation that is relevant to the core need.
    4. All keys and string values must be enclosed in double quotes. Do not use single quotes or unquoted keys.
    5. Ensure all fields referenced in transformation are valid and exist in the preceding data pipeline step.
    6. Double check the data types for all fields used in the encoding block to match the transformation output.
    7. Avoid complex regular expressions in calculate transforms that may be difficult for the vega-lite engine to parse.

    CRITICAL DATA HANDLING RULES:
    8. Always handle null/empty values before operations:
    - Use (datum.field || '') pattern for string operations
    - Add filter "datum.field != null && datum.field != ''" before processing fields with high null rates

    9. For delimited fields (semicolon, comma separated):
    - First filter out empty rows: filter "datum.field != ''"
    - Then split: calculate "split(datum.field, delimiter)"
    - Always validate array before flatten: calculate "isArray(datum.array) ? datum.array : []"
    - Use simple flatten syntax: "flatten": ["field_name"]

    10. For text search operations:
        - Use indexof(lower(field), 'keyword') >= 0 instead of regex
        - Combine multiple conditions with || operator
        - Pre-clean text with lower() function

    11. For numeric fields:
        - Use toNumber() explicitly for calculations
        - For year data already in numeric format, use type: "quantitative" not "temporal"

    ERROR PREVENTION:
    12. Do NOT use: regex test(), anonymous functions, map(), filter(), reduce()
    13. Do NOT assume any field is always non-null - check the dataset info above

    """)

    human_message = HumanMessage(content=f"""
    Please generate the vega-lite specification for the visualisation.
    """
    )

    class ResponseFormatter(BaseModel):
        visualisation: Visualisation

    response = llm.with_structured_output(ResponseFormatter).invoke([system_message, human_message])
    return response

def get_vega_lite_spec_inline(state: State):
    """
    Get the vega-lite specification for the visualisation.
    """
    llm = get_llm(temperature=0.0)
    # dataset_info = get_dataset_info(config["dataset"])

    if global_memory.latest_state is not None:
        thread_dir = global_memory._get_thread_dir()
    else:
        thread_dir = "outputs_sync/vis_report"

    dataset_path = f"{thread_dir}/dataset_global_filtered.csv"

    dataset_info = get_dataset_info(dataset_path)
    print(f"Using dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    data_values = df.to_dict('records')
    print(len(data_values))
    # Limit to first 1000 rows to avoid too large specification
    if len(data_values) > 1000:
        data_values = data_values[:1000]

    system_message = SystemMessage(content=f"""
    You are an expert in creating vega-lite specifications for visualisations. 
    To answer the user's question around the topic {config['topic']}, you should use this curated dataset {dataset_path} which has already been filtered to fulfil the targeted topic. So you don't need to filter the data again.

    The dataset information is as follows:
    {dataset_info}

    The visualisation task is to answer this question:
    {state["analysis_schema"]["information_needed"]["question_text"]}

    You can simplify the question if it is too complex to prioritise the core need of providing a valid visualisation that can be rendered by Vega-Lite modern specification.

    Requirements: No need to filter the data again. You can use the dataset directly.
    1. Generate valid vega-lite specification with inline data.
    2. Use "data": {{"values": [...]}} format for the data, not URL references.
    3. Use 'test' function for string matching and 'lower' function for case-insensitive matching: test(lower(datum.field), 'pattern')
    4. Robustness is prioritised over complexity. Do not generate more than 2 subplots. Do not use more than 2 levels of data transformation.
    5. If the information need is too complex, you don't need to fulfil the complete need. You may generate a visualisation that is relevant to the core need.
    6. The data will be provided as inline values, not as a file URL.
    7. All keys and string values must be enclosed in double quotes. Do not use single quotes or unquoted keys.
    8. Ensure all fields referenced in transformation are valid and exist in the preceding data pipeline step.
    9. Double check the data types for all fields used in the encoding block to match the transformation output.
    10. Avoid complex regular expressions in calculate transforms that may be difficult for the vega-lite engine to parse.
    11. Always handle null/empty values before operations:
    - Use (datum.field || '') pattern for string operations
    - Add filter "datum.field != null && datum.field != ''" before processing fields with high null rates
    12. For delimited fields (semicolon, comma separated):
    - First filter out empty rows: filter "datum.field != ''"
    - Then split: calculate "split(datum.field, delimiter)"
    - Always validate array before flatten: calculate "isArray(datum.array) ? datum.array : []"
    - Use simple flatten syntax: "flatten": ["field_name"]
    13. For text search operations:
    - Use indexof(lower(field), 'keyword') >= 0 instead of regex
    - Combine multiple conditions with || operator
    - Pre-clean text with lower() function
    14. For numeric fields:
        - Use toNumber() explicitly for calculations
        - For year data already in numeric format, use type: "quantitative" not "temporal"
    15. Do NOT use: regex test(), anonymous functions, map(), filter(), reduce()
    16. Do NOT assume any field is always non-null - check the dataset info above
    """)

    human_message = HumanMessage(content=f"""
    Please generate the vega-lite specification for the visualisation.

    You can simplify the question if it is too complex as long as it is relevant to the question.
    
    Use this data format in your specification:
    "data": {{
        "values": {json.dumps(data_values[:100])}  // First 100 rows as example
    }}
    
    The full dataset has {len(data_values)} rows. You can use all the data or a subset as needed.
    
    IMPORTANT: In Vega-Lite calculate expressions:
    - Use 'test' function for string matching: test(datum.Title, 'automat') 
    - Use 'lower' function for case-insensitive matching: test(lower(datum.Title), 'automat')
    - Do NOT use JavaScript methods like .toLowerCase() or .indexOf()
    - Use 'lower' function instead of .toLowerCase()
    """
    )

    class ResponseFormatter(BaseModel):
        visualisation: Visualisation

    response = llm.with_structured_output(ResponseFormatter).invoke([system_message, human_message])
    return response

def get_altair_visualisation(state: State):
    """
    Get the Altair visualisation for the given state.
    Similar to get_antv_visualisation but generates Altair charts.
    """
    thread_dir = ""
    # Get dataset path similar to get_vega_lite_spec_new
    if global_memory and global_memory.latest_state is not None:
        thread_dir = global_memory._get_thread_dir()
    else:
        thread_dir = "outputs_sync/vis_report"
    
    dataset_path = f"{thread_dir}/dataset_global_filtered.csv"
    
    # Fallback to config dataset if filtered dataset doesn't exist
    if not os.path.exists(dataset_path):
        dataset_path = config["dataset"]
    
    dataset_info = get_dataset_info(dataset_path)
    print(f"Using dataset: {dataset_path}")
    
    # Read the CSV data
    df = pd.read_csv(dataset_path)
    data_values = df.to_dict('records')
    print(f"Dataset has {len(data_values)} rows")
    
    # Limit to first 1000 rows to avoid too large specification
    if len(data_values) > 1000:
        data_values = data_values[:1000]
    
    # Get the question from state
    question = state["analysis_schema"]["information_needed"]["question_text"]
    
    llm = get_llm()
    
    system_message = SystemMessage(content=f"""
    You are an expert in creating Altair visualizations for data analysis.

    The dataset information is as follows:
    {dataset_info}

    The visualisation task is to answer this question:
    {question}

    Please refer to the information need when you designing and generating the visualisation, but not necessary to follow the information strictly as these are just suggestions and can be too complex:
    {state["analysis_schema"]["information_needed"]}

    Requirements:
    1. Generate valid Altair Python code that can be executed directly to produce plots. You can generate multiple plots, but ideally no more than 3.
    2. Use 'df' as the DataFrame variable name (the data is already loaded).
    3. Use Python string methods like .lower() for case-insensitive matching, NOT Vega-Lite functions.
    4. Use pandas operations for data preprocessing if needed, then pass to Altair.
    5. Robustness is prioritised over complexity.
    6. If the information need is too complex, you can generate a visualisation that is relevant to the core need.
    7. Return only the Altair chart code, not the full Python script.
    8. Do NOT use Vega-Lite syntax like 'lower(datum.field)' - use Python syntax instead.
    9. IMPORTANT: Assign the final chart to a variable named 'chart' so it can be captured.
    10. Keep the code simple and avoid complex multi-line expressions.
    """
    )

    class ResponseFormatter(BaseModel):
        visualisation: Visualisation

    max_retries = 5
    for attempt in range(max_retries):
        try:
            print(f"\n=== Altair Generation Attempt {attempt + 1}/{max_retries} ===")
            
            # Add attempt-specific guidance to human message
            if attempt > 0:
                human_message = HumanMessage(content=f"""
                Please generate the Altair chart code for the visualisation.
                
                The DataFrame 'df' is already loaded with {len(data_values)} rows of data. 
                It is already filtered to fulfil the targeted topic, so you don't need to filter the data again. 
                However, you can still use the original dataset to generate the visualisation, which is stored in the file {config["dataset"]}, when you need to use the original dataset.
                
                IMPORTANT: In Altair code:
                - Use Python string methods like .lower() for case-insensitive matching
                - Use pandas operations for data preprocessing if needed
                - Return only the chart code, e.g., 'alt.Chart(df).mark_bar().encode(...)'
                - Make sure the code can be executed directly with the 'df' DataFrame
                - Do NOT use Vega-Lite syntax like 'lower(datum.field)' - use Python syntax instead
                - Example: Use 'df[df.Title.str.lower().str.contains("automat")]' instead of Vega-Lite transforms
                - CRITICAL: Assign the final chart to a variable named 'chart' so it can be captured
                - Example: chart = alt.Chart(data).mark_bar().encode(...)
                
                PREVIOUS ATTEMPT FAILED: You can simplify the question and information needed if it is too complex to prioritise the core need of providing a valid visualisation. Please make the code even simpler and more robust.
                Avoid complex data transformations and multi-line expressions.
                """
                )
            else:
                human_message = HumanMessage(content=f"""
                Please generate the Altair chart code for the visualisation.
                
                The DataFrame 'df' is already loaded with {len(data_values)} rows of data. The data is already filtered to fulfil the targeted topic, so you don't need to filter the data again.
                
                IMPORTANT: In Altair code:
                - Use Python string methods like .lower() for case-insensitive matching
                - Use pandas operations for data preprocessing if needed
                - Return only the chart code, e.g., 'alt.Chart(df).mark_bar().encode(...)'
                - Make sure the code can be executed directly with the 'df' DataFrame
                - Do NOT use Vega-Lite syntax like 'lower(datum.field)' - use Python syntax instead
                - Example: Use 'df[df.Title.str.lower().str.contains("automat")]' instead of Vega-Lite transforms
                - CRITICAL: Assign the final chart to a variable named 'chart' so it can be captured
                - Example: chart = alt.Chart(data).mark_bar().encode(...)
                """
                )

            response = llm.with_structured_output(ResponseFormatter).invoke([system_message, human_message])
            
            # Extract the Altair code from the response
            if hasattr(response, 'visualisation'):
                altair_code = response.visualisation['specification']
            elif isinstance(response, dict) and 'visualisation' in response:
                altair_code = response['visualisation']['specification']
            elif isinstance(response, dict) and 'specification' in response:
                altair_code = response['specification']
            else:
                raise ValueError("Cannot extract specification from response")
            
            print(f"Generated Altair code:")
            print(altair_code)
            
            # Execute the Altair code and validate the chart
            local_vars = {'df': df, 'alt': alt, 'pd': pd}
            exec(altair_code, globals(), local_vars)
            
            # Get the chart from the local namespace
            if 'chart' not in local_vars:
                raise ValueError("No 'chart' variable found in executed code")
            
            chart = local_vars['chart']
            
            # Validate that chart is a valid Altair chart object
            if not hasattr(chart, 'save') or not callable(getattr(chart, 'save', None)):
                raise ValueError("Generated object is not a valid Altair chart")
            
            # Test if the chart can be saved (this will catch many errors)
            figid = str(uuid.uuid4())
            try:
                chart.save(f"{thread_dir}/visualization_{figid}.html")
                print("✅ Altair chart saved successfully")
            except Exception as save_error:
                raise ValueError(f"Chart cannot be saved: {save_error}")
            
            # Read the generated HTML file and validate it
            with open(f"{thread_dir}/visualization_{figid}.html", "r", encoding="utf-8") as f:
                chart_html = f.read()
            
            # Check if the HTML is empty
            if len(chart_html.strip()) == 0:
                raise ValueError("Generated HTML is empty")
            
            # Only check for actual error messages, not just the word "error" in normal content
            # if "javascript error" in chart_html.lower() or "vega error" in chart_html.lower():
            #     raise ValueError("Generated HTML contains JavaScript/Vega error messages")
            
            # Extract just the chart content (remove the HTML wrapper from Altair)
            import re
            chart_match = re.search(r'<div id="altair-viz-[^"]*">(.*?)</div>', chart_html, re.DOTALL)
            if chart_match:
                chart_content = chart_match.group(0)
            else:
                # Fallback: use the entire HTML content
                body_match = re.search(r'<body>(.*?)</body>', chart_html, re.DOTALL)
                if body_match:
                    chart_content = body_match.group(1)
                else:
                    chart_content = chart_html
            
            # Validate that we have actual chart content
            if not chart_content or len(chart_content.strip()) < 100:
                raise ValueError("Generated chart content is too small or empty")
            
            unique_id = f"vis_{uuid.uuid4().hex[:8]}"
            chart_content = chart_content.replace('id="vis"', f'id="{unique_id}"')
            chart_content = chart_content.replace('#vis', f'#{unique_id}')
            chart_content = chart_content.replace('id="altair-viz-', f'id="altair-viz-{unique_id}-')
            
            wrapped_html = chart_content
            
            print("✅ Chart validation passed, HTML wrapper created")
            
            # Return the visualisation object similar to get_antv_visualisation
            visualisation = Visualisation(
                library="altair",
                specification=wrapped_html
            )
            
            return visualisation
            
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                print(f"💥 All {max_retries} attempts failed. Creating error visualisation.")
                # Return a simple error visualisation in HTML format
                error_html = f"""
                """
                error_visualisation = Visualisation(
                    library="altair",
                    specification=error_html
                )
                return error_visualisation
            else:
                print(f"🔄 Retrying... ({attempt + 2}/{max_retries})")
                continue


def run_in_sandbox(code: str, python_path: str = "python3"):
    # 1. Write LLM generated code to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as f:
        f.write(code.encode())
        script_path = f.name

    try:
        # 2. Run code with subprocess, capture stdout and stderr
        print(f"Running code: {code}")
        print(f"Script path: {script_path}")
        print(f"Python path: {python_path}")
        print("--------------------------------")
        result = subprocess.run(
            [python_path, script_path],  # Use provided python_path
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30  # 5 second timeout
        )
        print(f"Result: {result}")
        print("--------------------------------")

        return {
            "stdout": result.stdout,   # print() output
            "stderr": result.stderr,   # error output
            "exit_code": result.returncode
        }
    finally:
        os.unlink(script_path)  # 3. Delete temporary file
def get_altair_visualisation_sandbox(state: State):
    """
    Get the Altair visualisation for the given state using run_in_sandbox.
    Similar to get_altair_visualisation but uses sandbox instead of exec().
    """
    thread_dir = ""
    # Get dataset path similar to get_vega_lite_spec_new
    if global_memory and global_memory.latest_state is not None:
        thread_dir = global_memory._get_thread_dir()
    else:
        thread_dir = "outputs_sync/vis_report"
    
    dataset_path = f"{thread_dir}/dataset_global_filtered.csv"
    
    # Fallback to config dataset if filtered dataset doesn't exist
    if not os.path.exists(dataset_path):
        dataset_path = config["dataset"]
    
    dataset_info = get_dataset_info(dataset_path)
    print(f"Using dataset: {dataset_path}")
    
    # Read the CSV data
    df = pd.read_csv(dataset_path)
    data_values = df.to_dict('records')
    print(f"Dataset has {len(data_values)} rows")
    
    # Limit to first 1000 rows to avoid too large specification
    if len(data_values) > 1000:
        data_values = data_values[:1000]
    
    # Get the question from state
    question = state["analysis_schema"]["information_needed"]["question_text"]
    
    llm = get_llm()
    
    system_message = SystemMessage(content=f"""
    You are an expert in creating Altair visualizations for data analysis.

    The dataset information is as follows:
    {dataset_info}

    The visualisation task is to answer this question:
    {question}

    Please refer to the information need when you designing and generating the visualisation, but not necessary to follow the information strictly as these are just suggestions and can be too complex:
    {state["analysis_schema"]["information_needed"]}

    Requirements:
    1. Generate valid Altair Python code that can be executed directly to produce plots. You can generate multiple plots, but ideally no more than 3.
    2. Use 'df' as the DataFrame variable name (the data is already loaded).
    3. Use Python string methods like .lower() for case-insensitive matching, NOT Vega-Lite functions.
    4. Use pandas operations for data preprocessing if needed, then pass to Altair.
    5. Robustness is prioritised over complexity.
    6. If the information need is too complex, you can generate a visualisation that is relevant to the core need.
    7. Return only the Altair chart code, not the full Python script.
    8. Do NOT use Vega-Lite syntax like 'lower(datum.field)' - use Python syntax instead.
    9. IMPORTANT: Assign the final chart to a variable named 'chart' so it can be captured.
    10. Keep the code simple and avoid complex multi-line expressions.
    """
    )

    class ResponseFormatter(BaseModel):
        visualisation: Visualisation

    max_retries = 5
    for attempt in range(max_retries):
        try:
            print(f"\n=== Altair Generation Attempt {attempt + 1}/{max_retries} ===")
            
            # Add attempt-specific guidance to human message
            if attempt > 0:
                human_message = HumanMessage(content=f"""
                Please generate the Altair chart code for the visualisation.
                
                The DataFrame 'df' is already loaded with {len(data_values)} rows of data. 
                It is already filtered to fulfil the targeted topic, so you don't need to filter the data again. 
                However, you can still use the original dataset to generate the visualisation, which is stored in the file {config["dataset"]}, when you need to use the original dataset.
                
                IMPORTANT: In Altair code:
                - Use Python string methods like .lower() for case-insensitive matching
                - Use pandas operations for data preprocessing if needed
                - Return only the chart code, e.g., 'alt.Chart(df).mark_bar().encode(...)'
                - Make sure the code can be executed directly with the 'df' DataFrame
                - Do NOT use Vega-Lite syntax like 'lower(datum.field)' - use Python syntax instead
                - Example: Use 'df[df.Title.str.lower().str.contains("automat")]' instead of Vega-Lite transforms
                - CRITICAL: Assign the final chart to a variable named 'chart' so it can be captured
                - Example: chart = alt.Chart(data).mark_bar().encode(...)
                
                PREVIOUS ATTEMPT FAILED: You can simplify the question and information needed if it is too complex to prioritise the core need of providing a valid visualisation. Please make the code even simpler and more robust.
                Avoid complex data transformations and multi-line expressions.
                """
                )
            else:
                human_message = HumanMessage(content=f"""
                Please generate the Altair chart code for the visualisation.
                
                The DataFrame 'df' is already loaded with {len(data_values)} rows of data. The data is already filtered to fulfil the targeted topic, so you don't need to filter the data again.
                
                IMPORTANT: In Altair code:
                - Use Python string methods like .lower() for case-insensitive matching
                - Use pandas operations for data preprocessing if needed
                - Return only the chart code, e.g., 'alt.Chart(df).mark_bar().encode(...)'
                - Make sure the code can be executed directly with the 'df' DataFrame
                - Do NOT use Vega-Lite syntax like 'lower(datum.field)' - use Python syntax instead
                - Example: Use 'df[df.Title.str.lower().str.contains("automat")]' instead of Vega-Lite transforms
                - CRITICAL: Assign the final chart to a variable named 'chart' so it can be captured
                - Example: chart = alt.Chart(data).mark_bar().encode(...)
                """
                )

            response = llm.with_structured_output(ResponseFormatter).invoke([system_message, human_message])
            
            # Extract the Altair code from the response
            if hasattr(response, 'visualisation'):
                altair_code = response.visualisation['specification']
            elif isinstance(response, dict) and 'visualisation' in response:
                altair_code = response['visualisation']['specification']
            elif isinstance(response, dict) and 'specification' in response:
                altair_code = response['specification']
            else:
                raise ValueError("Cannot extract specification from response")
            
            print(f"Generated Altair code:")
            print(altair_code)
            
            # Execute the Altair code in sandbox
            figid = str(uuid.uuid4())
            
            # Build complete script with data loading and chart HTML output
            full_script = f"""
import pandas as pd
import altair as alt
import os

# Load dataset
df = pd.read_csv('{dataset_path}')

# User's visualization code
{altair_code}

# Validate chart and convert to HTML
if 'chart' not in locals():
    print("ERROR: No 'chart' variable found in executed code")
    exit(1)
    
# Check if chart is any valid Altair chart type
if not hasattr(chart, 'to_html') or not callable(getattr(chart, 'to_html', None)):
    print("ERROR: 'chart' is not a valid Altair chart object")
    exit(1)

# Convert chart to HTML and output to stdout
chart_html = chart.to_html()
print(chart_html)
"""
            print("--------------------------------")
            print(f"Full script: {full_script}")
            print("--------------------------------")
            result = run_in_sandbox(full_script)
            
            # Check sandbox execution result
            if result["exit_code"] != 0:
                raise ValueError(f"Sandbox execution failed: {result['stderr']}")
            
            # Get the chart HTML from stdout
            chart_html = result["stdout"]
            
            # Check if the HTML is empty
            if len(chart_html.strip()) == 0:
                raise ValueError("Generated HTML is empty")
            
            # Extract just the chart content (remove the HTML wrapper from Altair)
            import re
            chart_match = re.search(r'<div id="altair-viz-[^"]*">(.*?)</div>', chart_html, re.DOTALL)
            if chart_match:
                chart_content = chart_match.group(0)
            else:
                # Fallback: use the entire HTML content
                body_match = re.search(r'<body>(.*?)</body>', chart_html, re.DOTALL)
                if body_match:
                    chart_content = body_match.group(1)
                else:
                    chart_content = chart_html
            
            # Validate that we have actual chart content
            if not chart_content or len(chart_content.strip()) < 100:
                raise ValueError("Generated chart content is too small or empty")
            
            unique_id = f"vis_{uuid.uuid4().hex[:8]}"
            chart_content = chart_content.replace('id="vis"', f'id="{unique_id}"')
            chart_content = chart_content.replace('#vis', f'#{unique_id}')
            chart_content = chart_content.replace('id="altair-viz-', f'id="altair-viz-{unique_id}-')
            
            wrapped_html = chart_content
            
            print("✅ Chart validation passed, HTML wrapper created")
            
            # Return the visualisation object similar to get_antv_visualisation
            visualisation = Visualisation(
                library="altair",
                specification=wrapped_html
            )
            
            return visualisation
            
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                print(f"💥 All {max_retries} attempts failed. Creating error visualisation.")
                # Return a simple error visualisation in HTML format
                error_html = f"""
                <div id="vis">
                    <p>Error: {e}</p>
                </div>
                """
                error_visualisation = Visualisation(
                    library="altair",
                    specification=error_html
                )
                return error_visualisation
            else:
                print(f"🔄 Retrying... ({attempt + 2}/{max_retries})")
                continue