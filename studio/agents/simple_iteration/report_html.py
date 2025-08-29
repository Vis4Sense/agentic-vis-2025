import re
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

# Optional import for altair
try:
    import altair as alt
    ALTAR_AVAILABLE = True
except ImportError:
    ALTAR_AVAILABLE = False
    print("Warning: altair not available, some visualization features may be limited")

def markdown_to_html(md: str) -> str:
    """Very-lightweight markdown → HTML for our narratives."""
    # horizontal rules
    html = re.sub(r'^---\s*$', r'<hr/>', md, flags=re.MULTILINE)
    # headings
    html = re.sub(r'^###\s*(.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    # bold
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    # paragraphs
    parts = [p.strip() for p in html.split('\n\n') if p.strip()]
    return '\n'.join(f'<p>{p}</p>' for p in parts)

def convert_altair_to_vega_lite(altair_code: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convert Altair code to Vega-Lite specification
    
    Args:
        altair_code: Python code that creates an Altair chart
        df: DataFrame to use for the visualization
        
    Returns:
        Vega-Lite specification as a dictionary
    """
    try:
        # Create safe namespace with only allowed objects
        safe_globals = {
            "alt": alt,
            "df": df,
            "pd": pd,
            "chart": None
        }
        safe_locals = {}
        
        # Execute the Altair code
        exec(altair_code, safe_globals, safe_locals)
        
        # Get the chart from locals or globals
        chart = safe_locals.get("chart") or safe_globals.get("chart")
        
        if chart is None:
            raise ValueError("No chart object found in the Altair code")
        
        if not isinstance(chart, alt.Chart):
            raise ValueError("Generated object is not an Altair Chart")
        
        # Convert to Vega-Lite specification
        vega_spec = chart.to_dict()
        return vega_spec
        
    except Exception as e:
        print(f"Error converting Altair code to Vega-Lite: {e}")
        # Return a fallback chart
        fallback_chart = alt.Chart(df).mark_bar().encode(
            x='category',
            y='value'
        )
        return fallback_chart.to_dict()

def load_dataset(dataset_path: str = None) -> pd.DataFrame:
    """
    Load dataset from path or return default dataset
    
    Args:
        dataset_path: Path to the dataset CSV file
        
    Returns:
        DataFrame with the data
    """
    if dataset_path:
        try:
            df = pd.read_csv(dataset_path)
            return df
        except Exception as e:
            print(f"Error loading dataset from {dataset_path}: {e}")
    
    # Return default dataset if loading fails
    return pd.DataFrame({
        'category': ['A', 'B', 'C', 'D', 'E'],
        'value': [10, 20, 15, 25, 30],
        'group': ['X', 'Y', 'X', 'Y', 'X']
    })

def generate_html_report(output_state: dict, output_path: str, shared_memory=None):
    """
    Builds a simple HTML report that displays insights from the state data and shared memory.
    
    Args:
        output_state: Current state dictionary
        output_path: Path to save the HTML report
        shared_memory: Optional shared memory instance to get complete history
    """
    
    # 2. Build the HTML document
    html_lines = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='utf-8'>",
        "  <title>Insights Report</title>",
        "  <script src='https://cdn.jsdelivr.net/npm/vega@5'></script>",
        "  <script src='https://cdn.jsdelivr.net/npm/vega-lite@5'></script>",
        "  <script src='https://cdn.jsdelivr.net/npm/vega-embed@6'></script>",
        "  <style>",
        "    body { font-family: sans-serif; margin: 2em; line-height: 1.6; }",
        "    h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }",
        "    h2 { color: #34495e; margin-top: 2em; }",
        "    .insight { margin-bottom: 1.5em; padding: 1em; background-color: #f8f9fa; border-left: 4px solid #3498db; border-radius: 3px; }",
        "    .insight-number { font-weight: bold; color: #3498db; margin-bottom: 0.5em; }",
        "    .topic { font-size: 1.2em; color: #7f8c8d; margin-bottom: 1em; }",
        "    .question { font-style: italic; color: #95a5a6; margin-bottom: 1em; }",
        "    .iteration { margin-bottom: 2em; padding: 1.5em; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #e9ecef; }",
        "    .iteration h3 { color: #2c3e50; margin-top: 0; border-bottom: 2px solid #3498db; padding-bottom: 10px; }",
        "    .facts { margin: 1em 0; }",
        "    .facts pre { background-color: #f1f3f4; padding: 1em; border-radius: 3px; overflow-x: auto; font-size: 0.8em; }",
        "    .facts-output { margin: 1em 0; }",
        "    .facts-output pre { background-color: #e8f5e8; padding: 1em; border-radius: 3px; overflow-x: auto; font-size: 0.8em; }",
        "    .insights { margin: 1em 0; }",
        "    .section-title { font-weight: bold; color: #34495e; margin-bottom: 0.5em; }",
        "    .collapsible { cursor: pointer; }",
        "    .collapsible:hover { background-color: #e9ecef; }",
        "    .collapsible-content { display: none; padding: 1em; background-color: #f8f9fa; border-radius: 3px; margin-top: 0.5em; }",
        "    hr { margin: 2em 0; border: none; border-top: 1px solid #ecf0f1; }",
        "  </style>",
        "  <script>",
        "    function toggleSection(id) {",
        "      var content = document.getElementById(id);",
        "      if (content.style.display === 'none' || content.style.display === '') {",
        "        content.style.display = 'block';",
        "      } else {",
        "        content.style.display = 'none';",
        "      }",
        "    }",
        "  </script>",
        "</head>",
        "<body>",
    ]
    
    # Add topic information if available
    if "topic" in output_state:
        html_lines.append(f"<h1>Data Insights Report</h1>")
        html_lines.append(f"<div class='topic'><strong>Topic:</strong> {output_state['topic']}</div>")
    
    html_lines.append("<hr/>")
    
    # Get complete history from shared memory if available
    complete_history = []
    if shared_memory is not None:
        try:
            memory_states = shared_memory.get_history()
            # Group states by iteration and select the last state from each iteration
            iteration_groups = {}
            
            for state in memory_states:
                iteration_count = state.get("iteration_count", 0)
                if iteration_count > 0:  # Skip initial states without iteration
                    if iteration_count not in iteration_groups:
                        iteration_groups[iteration_count] = []
                    iteration_groups[iteration_count].append(state)

            # print(iteration_groups.keys())
            
            # Select the first complete state from each iteration
            for iteration_num in sorted(iteration_groups.keys()):
                states_in_iteration = iteration_groups[iteration_num]
                
                # Find the first state in this iteration that has complete information
                selected_state = None
                for state in states_in_iteration:
                    if "question" in state and "facts" in state and "insights" in state:
                        selected_state = state
                        break
                
                if selected_state:
                    iteration_data = {
                        "question": selected_state.get("question", {}),
                        "facts": selected_state.get("facts", {}),
                        "insights": selected_state.get("insights", [])
                    }
                    complete_history.append(iteration_data)
                    
        except Exception as e:
            print(f"Warning: Could not load history from shared memory: {e}")
    
    # Use iteration_history from state if available, otherwise use complete_history
    # iteration_history = output_state.get("iteration_history", [])
    # if not iteration_history and complete_history:
    iteration_history = complete_history
    
    # Display iteration history if available
    if iteration_history:
        total_iterations = len(iteration_history)
        total_insights = sum(len(iter.get("insights", [])) for iter in iteration_history)
        
        html_lines.append("<h2>Analysis Iterations</h2>")
        html_lines.append(f"<div style='margin-bottom: 1em; padding: 1em; background-color: #e8f4fd; border-radius: 5px;'>")
        html_lines.append(f"  <strong>Summary:</strong> {total_iterations} iterations completed, {total_insights} total insights generated")
        html_lines.append("</div>")
        
        for i, iteration in enumerate(iteration_history, 1):
            html_lines.append(f"<div class='iteration'>")
            html_lines.append(f"  <h3>Iteration {i}</h3>")
            
            # Display question
            if "question" in iteration and iteration["question"]:
                question_text = iteration["question"].get("question", "No question available")
                html_lines.append(f"  <div class='question'><strong>Question:</strong> {question_text}</div>")
            
            # Display facts (code and output) - make them collapsible
            if "facts" in iteration and iteration["facts"]:
                facts = iteration["facts"]
                
                # Analysis Code (collapsible)
                if facts.get("code"):
                    html_lines.append(f"  <div class='facts'>")
                    html_lines.append(f"    <div class='section-title collapsible' onclick='toggleSection(\"code-{i}\")'>📊 Analysis Code (click to expand)</div>")
                    html_lines.append(f"    <div id='code-{i}' class='collapsible-content'>")
                    html_lines.append(f"      <pre><code>{facts['code']}</code></pre>")
                    html_lines.append(f"    </div>")
                    html_lines.append(f"  </div>")
                
                # Analysis Results (collapsible)
                if facts.get("stdout"):
                    html_lines.append(f"  <div class='facts-output'>")
                    html_lines.append(f"    <div class='section-title collapsible' onclick='toggleSection(\"output-{i}\")'>📈 Analysis Results (click to expand)</div>")
                    html_lines.append(f"    <div id='output-{i}' class='collapsible-content'>")
                    html_lines.append(f"      <pre>{facts['stdout']}</pre>")
                    html_lines.append(f"    </div>")
                    html_lines.append(f"  </div>")
            
            # Display insights (without visualizations in iteration history)
            if "insights" in iteration and iteration["insights"]:
                html_lines.append(f"  <div class='insights'>")
                html_lines.append(f"    <div class='section-title'>💡 Key Insights:</div>")
                
                for j, insight in enumerate(iteration["insights"], 1):
                    html_lines.append(f"    <div class='insight'>")
                    html_lines.append(f"      <div class='insight-number'>Insight {j}</div>")
                    html_lines.append(f"      <div>{insight}</div>")
                    html_lines.append(f"    </div>")
                html_lines.append(f"  </div>")
            
            html_lines.append("</div>")
            html_lines.append("<hr/>")
    
    # Display storyline if available
    if "storyline" in output_state and output_state["storyline"] and output_state["storyline"].get("nodes"):
        storyline = output_state["storyline"]
        html_lines.append("<h2>📖 Data Exploration Storyline</h2>")
        html_lines.append(f"<div style='margin-bottom: 1em; padding: 1em; background-color: #fff3cd; border-radius: 5px; border-left: 4px solid #ffc107;'>")
        html_lines.append(f"  <strong>Theme:</strong> {storyline.get('theme', 'Data Exploration')}")
        html_lines.append("</div>")
        
        for i, story_node in enumerate(storyline["nodes"], 1):
            html_lines.append(f"<div class='iteration' style='margin-bottom: 2em;'>")
            html_lines.append(f"  <h3>📚 Story Node {i}: {story_node.get('description', '')}</h3>")
            
            # Display insights for this story node if available
            node_insights = story_node.get("insights", [])
            if node_insights:
                html_lines.append(f"  <div class='insights'>")
                html_lines.append(f"    <div class='section-title'>💡 Related Insights:</div>")
                for j, insight in enumerate(node_insights, 1):
                    html_lines.append(f"    <div class='insight'>")
                    html_lines.append(f"      <div class='insight-number'>Insight {j}</div>")
                    html_lines.append(f"      <div>{insight}</div>")
                    html_lines.append(f"    </div>")
                html_lines.append(f"  </div>")
            
            # Display visualizations for this story node
            node_visualizations = story_node.get("visualizations", [])
            if node_visualizations:
                html_lines.append(f"  <div class='insights'>")
                html_lines.append(f"    <div class='section-title'>📊 Visualizations:</div>")
                
                for j, visualization in enumerate(node_visualizations, 1):
                    if visualization.get("is_appropriate", False):
                        html_lines.append(f"    <div class='insight-visualization' style='margin-top: 1em; padding: 1em; background-color: #f0f8ff; border-radius: 5px; border-left: 3px solid #3498db;'>")
                        html_lines.append(f"      <div style='font-weight: bold; color: #3498db; margin-bottom: 0.5em;'>📊 {visualization.get('chart_type', 'Chart')}</div>")
                        html_lines.append(f"      <div style='font-style: italic; margin-bottom: 1em;'>{visualization.get('description', '')}</div>")
                        
                        # Add visualization container
                        div_id = f"story-node-{i}-vis-{j}"
                        html_lines.append(f"      <div id='{div_id}' style='margin: 0.5em 0;'></div>")
                        
                        # Add Vega-Lite visualization
                        try:
                            spec_str = visualization.get('spec', '')
                            if spec_str:
                                vega_spec = json.loads(spec_str)
                                spec_json = json.dumps(vega_spec)
                                html_lines.extend([
                                    "      <script>",
                                    f"        vegaEmbed('#{div_id}', {spec_json})",
                                    "          .catch(console.error);",
                                    "      </script>",
                                ])
                        except Exception as e:
                            html_lines.append(f"      <p style='color: #e74c3c; font-size: 0.9em;'><em>Error loading visualization: {str(e)}</em></p>")
                        
                        html_lines.append(f"    </div>")
                
                html_lines.append(f"  </div>")
            
            html_lines.append("</div>")
            html_lines.append("<hr/>")
    
    # Display current insights if available (for backward compatibility)
    elif "insights" in output_state and output_state["insights"]:
        html_lines.append("<h2>Key Insights</h2>")
        
        insights = output_state["insights"]
        # Get visualizations for current insights if available
        current_visualizations = []
        if "visualizations" in output_state and output_state["visualizations"]:
            current_visualizations = output_state["visualizations"]
        
        for i, insight in enumerate(insights, 1):
            html_lines.append(f"<div class='insight'>")
            html_lines.append(f"  <div class='insight-number'>Insight {i}</div>")
            html_lines.append(f"  <div>{insight}</div>")
            
            # Find matching visualization for this insight
            matching_visualization = None
            for vis in current_visualizations:
                if vis.get("insight", "").strip() == insight.strip():
                    matching_visualization = vis
                    break
            
            # Add visualization if found and appropriate
            if matching_visualization and matching_visualization.get("is_appropriate", False):
                html_lines.append(f"  <div class='insight-visualization' style='margin-top: 1em; padding: 1em; background-color: #f0f8ff; border-radius: 5px; border-left: 3px solid #3498db;'>")
                html_lines.append(f"    <div style='font-weight: bold; color: #3498db; margin-bottom: 0.5em;'>📊 Visualization: {matching_visualization.get('chart_type', 'Chart')}</div>")
                html_lines.append(f"    <div style='font-style: italic; margin-bottom: 1em;'>{matching_visualization.get('description', '')}</div>")
                
                # Add visualization container
                div_id = f"current-insight-vis-{i}"
                html_lines.append(f"    <div id='{div_id}' style='margin: 0.5em 0;'></div>")
                
                # Add Vega-Lite visualization
                try:
                    spec_str = matching_visualization.get('spec', '')
                    if spec_str:
                        vega_spec = json.loads(spec_str)
                        spec_json = json.dumps(vega_spec)
                        html_lines.extend([
                            "    <script>",
                            f"      vegaEmbed('#{div_id}', {spec_json})",
                            "        .catch(console.error);",
                            "    </script>",
                        ])
                except Exception as e:
                    html_lines.append(f"    <p style='color: #e74c3c; font-size: 0.9em;'><em>Error loading visualization: {str(e)}</em></p>")
                
                html_lines.append(f"  </div>")
            
            html_lines.append("</div>")
    else:
        html_lines.append("<h2>Key Insights</h2>")
        html_lines.append("<p><em>No insights available in the current state.</em></p>")



    html_lines.extend([
        "</body>",
        "</html>"
    ])

    # 4. Write out html file
    Path(output_path).write_text("\n".join(html_lines), encoding="utf-8")


def generate_html_from_backup(backup_file_path: str, output_path: str):
    """
    Generate HTML report from a memory backup JSON file.
    
    Args:
        backup_file_path: Path to the memory backup JSON file
        output_path: Path to save the HTML report
    """
    import json
    
    try:
        # Load the backup file
        with open(backup_file_path, 'r', encoding='utf-8') as f:
            backup_data = json.load(f)
        
        # Extract states from backup
        states = backup_data.get('states', [])
        if not states:
            print(f"Warning: No states found in backup file {backup_file_path}")
            return
        
        # Get the last state as the current state
        current_state = states[-1]
        
        # Group states by iteration and select the last state from each iteration
        iteration_groups = {}
        
        for state in states:
            iteration_count = state.get("iteration_count", 0)
            if iteration_count > 0:  # Skip initial states without iteration
                if iteration_count not in iteration_groups:
                    iteration_groups[iteration_count] = []
                iteration_groups[iteration_count].append(state)
        
        # Select the first complete state from each iteration
        final_states = []
        for iteration_num in sorted(iteration_groups.keys()):
            states_in_iteration = iteration_groups[iteration_num]
            
            # Find the first state in this iteration that has complete information
            selected_state = None
            for state in states_in_iteration:
                if "question" in state and "facts" in state and "insights" in state:
                    selected_state = state
                    break
            
            if selected_state:
                final_states.append(selected_state)
        
        # Create a mock shared memory object that returns the final states
        class MockSharedMemory:
            def __init__(self, states_list):
                self.states = states_list
            
            def get_history(self):
                return self.states
        
        mock_memory = MockSharedMemory(final_states)
        
        # Generate HTML report using the existing function
        generate_html_report(current_state, output_path, mock_memory)
        
        print(f"✓ HTML report generated from backup: {output_path}")
        print(f"✓ Processed {len(states)} total states from backup file")
        print(f"✓ Selected {len(final_states)} final states (one per iteration)")
        print(f"✓ Iterations found: {list(iteration_groups.keys())}")
        
    except FileNotFoundError:
        print(f"Error: Backup file not found: {backup_file_path}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in backup file: {e}")
    except Exception as e:
        print(f"Error generating HTML from backup: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # output_state = {
    #   "topic": "research on provenance",
    #   "select_data_state": {
    #     "description": "Select papers related to provenance research by filtering Title, Abstract, AuthorKeywords, or InternalReferences for provenance-related terms; include key bibliographic and impact fields and order by year (most recent first).",
    #     "sql_query": "SELECT Conference, Year, Title, DOI, Link, FirstPage, LastPage, PaperType, Abstract, \"AuthorNames-Deduped\", AuthorNames, AuthorAffiliation, InternalReferences, AuthorKeywords, AminerCitationCount, CitationCount_CrossRef, PubsCited_CrossRef, Downloads_Xplore, Award, GraphicsReplicabilityStamp\nFROM Papers\nWHERE lower(Title) LIKE '%provenanc%'\n   OR lower(Abstract) LIKE '%provenanc%'\n   OR lower(AuthorKeywords) LIKE '%provenanc%'\n   OR lower(InternalReferences) LIKE '%provenanc%'\n   OR lower(Title) LIKE '%lineage%'\n   OR lower(Abstract) LIKE '%lineage%'\n   OR lower(AuthorKeywords) LIKE '%lineage%'\nORDER BY Year DESC;",
    #     "dataset_path": "outputs/simple_iteration/default_thread/dataset_selected.csv"
    #   },
    #   "question": {
    #     "question": "For each year in the dataset, count papers that explicitly mention “provenance” (case-insensitive) in Title, Abstract, AuthorKeywords, or InternalReferences; compute the proportion of such papers relative to all papers that year; produce a table of yearly counts and proportions and also report the top 5 conferences (by total count) that publish papers mentioning provenance across the full time span.",
    #     "handled": False,
    #     "spec": ""
    #   },
    #   "facts": {
    #     "code": "import pandas as pd\n\n# Read dataset\npath = 'outputs/simple_iteration/default_thread/dataset_selected.csv'\ndf = pd.read_csv(path, dtype=str)\n\n# Normalize year (extract 4-digit year if present)\ndf['Year_str'] = df.get('Year', '').astype(str).str.extract(r'(\\d{4})')[0].fillna('Unknown')\n\n# Fields to search for the word 'provenance'\nsearch_fields = ['Title', 'Abstract', 'AuthorKeywords', 'InternalReferences']\n\n# Build provenance presence flag (case-insensitive whole-word match)\nmask = False\nfor col in search_fields:\n    mask = mask | df.get(col, '').fillna('').str.contains(r'\\bprovenance\\b', case=False, regex=True)\n\ndf['mentions_provenance'] = mask.astype(int)\n\n# Yearly counts and proportions\nyearly = (df.groupby('Year_str')\n            .agg(total_papers=('Title', 'size'), provenance_papers=('mentions_provenance', 'sum'))\n            .reset_index())\n# sort years numerically when possible, with 'Unknown' last\nyearly['year_sort'] = yearly['Year_str'].apply(lambda x: int(x) if x != 'Unknown' else 10**9)\nyearly = yearly.sort_values('year_sort').drop(columns='year_sort')\nyearly['proportion'] = (yearly['provenance_papers'] / yearly['total_papers']).fillna(0)\n\n# Top 5 conferences by total provenance-mentioning papers across full span\nconf_stats = (df.groupby(df.get('Conference', 'Unknown'))\n                .agg(total_papers=('Title', 'size'), provenance_papers=('mentions_provenance', 'sum'))\n                .reset_index()\n                .sort_values('provenance_papers', ascending=False))\ntop5_confs = conf_stats.head(5)\n\n# Print concise facts as required\nprint('### Begin of facts')\nprint('Yearly table: columns = Year, total_papers, provenance_papers, proportion')\nprint(yearly[['Year_str', 'total_papers', 'provenance_papers', 'proportion']].to_string(index=False))\nprint('### End of facts')\n\nprint('### Begin of facts')\nprint('Top 5 conferences by count of papers mentioning \"provenance\" (conference, provenance_papers, total_papers)')\nprint(top5_confs[['Conference', 'provenance_papers', 'total_papers']].to_string(index=False))\nprint('### End of facts')\n",
    #     "stdout": "### Begin of facts\nYearly table: columns = Year, total_papers, provenance_papers, proportion\nYear_str  total_papers  provenance_papers  proportion\n    2004             2                  2    1.000000\n    2007             1                  1    1.000000\n    2008             1                  1    1.000000\n    2010             2                  2    1.000000\n    2013             3                  3    1.000000\n    2014             1                  1    1.000000\n    2015             5                  5    1.000000\n    2016             2                  2    1.000000\n    2017             2                  2    1.000000\n    2018             2                  2    1.000000\n    2019             4                  4    1.000000\n    2020             2                  1    0.500000\n    2021             2                  2    1.000000\n    2022             2                  2    1.000000\n    2023             2                  2    1.000000\n    2024             6                  5    0.833333\n### End of facts\n### Begin of facts\nTop 5 conferences by count of papers mentioning \"provenance\" (conference, provenance_papers, total_papers)\nConference  provenance_papers  total_papers\n      VAST                 19            19\n       Vis                 12            13\n   InfoVis                  4             5\n    SciVis                  2             2\n### End of facts\n",
    #     "stderr": "",
    #     "exit_code": 0
    #   },
    #   "insights": [
    #     "Yearly counts: between 2004–2024 the dataset contains 39 papers total; 37 of these explicitly mention “provenance” in Title, Abstract, AuthorKeywords, or InternalReferences.",
    #     "Per-year provenance counts and proportions (Year: provenance_papers / total_papers = proportion): 2004: 2/2 = 1.00; 2007: 1/1 = 1.00; 2008: 1/1 = 1.00; 2010: 2/2 = 1.00; 2013: 3/3 = 1.00; 2014: 1/1 = 1.00; 2015: 5/5 = 1.00; 2016: 2/2 = 1.00; 2017: 2/2 = 1.00; 2018: 2/2 = 1.00; 2019: 4/4 = 1.00; 2020: 1/2 = 0.50; 2021: 2/2 = 1.00; 2022: 2/2 = 1.00; 2023: 2/2 = 1.00; 2024: 5/6 ≈ 0.8333.",
    #     "Overall prevalence: provenance is mentioned in ~94.9% of papers across the full span (37/39).",
    #     "Notable deviations: only 2020 (50%) and 2024 (~83.3%) fall below 100% — every other year in the table shows 100% of papers mentioning provenance.",
    #     "Top venues publishing papers that mention provenance (provenance_papers / total_papers): VAST 19/19; Vis 12/13; InfoVis 4/5; SciVis 2/2. (These are the top conferences shown in the provided facts.)",
    #     "Interpretation: mentions of provenance are extremely common in this subset (near-universal in most years and concentrated in VAST and Vis), suggesting either strong topical focus in these venues or selection/filtering of papers that emphasizes provenance-related work."
    #   ]
    # }
    # # Import shared memory for testing
    # from agents.simple_iteration.memory import shared_memory
    # generate_html_report(output_state, "output.html", shared_memory)

    generate_html_from_backup("outputs/simple_iteration/thread_20250814_022434-sensemaking-faird/memory_backup.json", "output.html")

