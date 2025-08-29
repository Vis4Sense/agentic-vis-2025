import re
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

def clean_duplicate_graph_declarations(html_content: str) -> str:
    """
    Clean up duplicate 'const { Graph } = G6' declarations in the HTML content.
    Keeps only the first occurrence and removes subsequent ones.
    """
    # Pattern to match 'const { Graph } = G6' declarations
    graph_declaration_pattern = r'const\s*\{\s*Graph\s*\}\s*=\s*G6'
    
    # Find all occurrences
    matches = list(re.finditer(graph_declaration_pattern, html_content))
    
    if len(matches) <= 1:
        # No duplicates found, return original content
        return html_content
    
    # Keep the first occurrence, remove the rest
    result = html_content
    # Remove from last to first to avoid index shifting issues
    for match in reversed(matches[1:]):
        start, end = match.span()
        # Remove the entire line containing the declaration
        line_start = html_content.rfind('\n', 0, start) + 1
        line_end = html_content.find('\n', end)
        if line_end == -1:
            line_end = len(html_content)
        
        result = result[:line_start] + result[line_end:]
    
    return result

def markdown_to_html(md: str) -> str:
    """Convert markdown to HTML with basic formatting."""
    if not md:
        return ""
    
    # horizontal rules
    html = re.sub(r'^---\s*$', r'<hr class="my-8 border-gray-300"/>', md, flags=re.MULTILINE)
    # headings
    html = re.sub(r'^###\s*(.+)$', r'<h3 class="text-xl font-semibold text-gray-800 mt-6 mb-3">\1</h3>', html, flags=re.MULTILINE)
    # bold
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong class="font-semibold">\1</strong>', html)
    # paragraphs
    parts = [p.strip() for p in html.split('\n\n') if p.strip()]
    return '\n'.join(f'<p class="text-gray-700 leading-relaxed mb-4">{p}</p>' for p in parts)

def extract_facts_from_text(text: str) -> List[str]:
    """Extract facts from text that contains ### Begin of facts and ### End of facts markers."""
    if not text:
        return []
    
    facts = []
    pattern = r'### Begin of facts\n(.*?)\n### End of facts'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        fact_lines = [line.strip() for line in match.split('\n') if line.strip()]
        facts.extend(fact_lines)
    
    return facts

def generate_html_report(output_state: dict, output_path: str):
    """
    Builds a beautiful HTML report from the vis_report state data.
    
    Args:
        output_state: State dictionary containing report_outline and other data
        output_path: Path to save the HTML report
    """
    
    # Extract report outline from state
    report_outline = output_state.get("report_outline", [])
    config = output_state.get("config", {})
    
    # Build the HTML document
    html_lines = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='utf-8'>",
        "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "  <title>Research Report: Automated Visualization</title>",
        "  <script src='https://cdn.jsdelivr.net/npm/vega@5'></script>",
        "  <script src='https://cdn.jsdelivr.net/npm/vega-lite@5'></script>",
        "  <script src='https://cdn.jsdelivr.net/npm/vega-embed@6'></script>",
        "  <script src='https://cdn.tailwindcss.com'></script>",
        "  <script>",
        "    tailwind.config = {",
        "      theme: {",
        "        extend: {",
        "          typography: ({ theme }) => ({",
        "            DEFAULT: { css: { color: theme('colors.gray.800') } }",
        "          })",
        "        }",
        "      }",
        "    }",
        "  </script>",
        "</head>",
        "<body class='bg-white text-gray-800'>",

        # Progress Bar
        "  <div id='progress-bar' class='fixed top-0 left-0 h-1 bg-black z-[9999]' style='width:0%'></div>"

        # Top Navigation Bar
        "  <nav class='sticky top-0 bg-white border-b border-gray-200 z-50'>",
        "    <div class='w-[95%] mx-auto px-6 py-3 flex justify-between items-center'>",
        "      <a href='#' class='font-bold text-lg'>Agentic VIS Report</a>",
        #     PC nav
        "      <div class='hidden md:flex items-center space-x-10 text-sm'>",
        "        <a href='https://www.visagent.org/' target='_blank' class='hover:text-primary-600 text-lg'>Challenge</a>",
        "        <a href='https://github.com/the-lord-of-stars/mini-template' target='_blank'>",
        "          <img src='https://cdnjs.cloudflare.com/ajax/libs/simple-icons/9.16.0/github.svg' alt='GitHub' class='w-6 h-6'>",
        "        </a>",
        "      </div>",

        #     Hamburger for mobile
        "      <button id='menu-btn' class='md:hidden flex items-center focus:outline-none'>",
        "        <svg xmlns='http://www.w3.org/2000/svg' class='w-6 h-6' fill='none' viewBox='0 0 24 24' stroke='currentColor'>",
        "          <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M4 6h16M4 12h16M4 18h16' />",
        "        </svg>",
        "      </button>",
        "    </div>",
        "    <div id='mobile-menu' class='hidden md:hidden bg-white border-t border-gray-200'>",
        "      <div class='px-6 py-3 flex flex-col space-y-4'>",
        "        <a href='https://www.visagent.org/' target='_blank' class='hover:text-primary-600 text-lg'>Official Challenge Website</a>",
        "        <a href='https://github.com/the-lord-of-stars/mini-template' target='_blank' class='flex items-center space-x-4'>",
        "          <img src='https://cdnjs.cloudflare.com/ajax/libs/simple-icons/9.16.0/github.svg' alt='GitHub' class='w-6 h-6'>",
        "          <span class='text-lg'>Project GitHub Repo</span>",
        "        </a>",
        "      </div>",
        "    </div>",
        "  </nav>",

        "  <script>",
        "    const btn = document.getElementById('menu-btn');",
        "    const menu = document.getElementById('mobile-menu');",
        "    btn.addEventListener('click', () => { menu.classList.toggle('hidden'); });",
        "  </script>",


        # TOC
        "  <nav class='hidden lg:block fixed top-20 left-6 w-[15%] h-[80vh] overflow-y-auto",
        "               p-4 text-sm'",
        "       aria-label='Table of contents'>",
        "    <h2 class='font-semibold text-gray-900 mb-3'>Table of contents</h2>",
        "    <ol id='toc-list' class='space-y-2'></ol>",
        "  </nav>",


        #  Main container
        "  <main class='w-[90%] max-w-3xl mx-auto px-6 py-12 lg:w-[75%] lg:ml-[20%] lg:max-w-none lg:mx-0'>",
        "    <header class='text-center mb-12'>",
        f"      <h1 class='text-4xl font-bold text-gray-900 mb-4'>{output_state['config']['topic']}</h1>",
        # Add Date
        f"       <p class='text-lg text-gray-600'>Generated on {datetime.now().strftime('%Y-%m-%d')} </p>",
        "    </header>",
        "    <article class='prose prose-lg prose-gray'>"
    ]
    
    # Process each section
    for section in report_outline:
        section_number = section.get("section_number", 0)
        section_name = section.get("section_name", "Untitled Section")
        section_description = section.get("section_description", "")
        content = section.get("content", [])

        html_lines.extend([
            f"<section class='mb-16'>",
            f"  <h2 class='text-3xl font-bold mb-6 border-b border-gray-200 pb-2'>",
            f"    {section_number}. {section_name}",
            f"  </h2>",
            f"  <div class='space-y-10'>"
        ])

        for idx, content_item in enumerate(content):
            content_type = content_item.get("type", "")
            text = content_item.get("text", "")

            # --- Introduction Text ---
            if content_type == "introduction":
                html_lines.extend([
                    "    <div class='prose prose-lg prose-gray max-w-none'>",
                    f"      {markdown_to_html(text)}",
                    "    </div>"
                ])
            elif content_type == "visualisation":
                visualisation = content_item.get("visualisation", {})
                vis_text = content_item.get("text", "")
                facts = content_item.get("facts", "")

                html_lines.extend([
                    "    <div class='my-8'>",
                    "      <div class='bg-gray-50 rounded-lg shadow-sm p-4'>"
                ])

                # Add visualization if available
                if visualisation and visualisation.get("specification"):
                    try:
                        div_id = f"vis-{section_number}-{content_item.get('id', idx)}"

                        library = visualisation.get("library", "vega-lite")
                        if library == "vega-lite":
                            spec = json.loads(visualisation["specification"])
                            html_lines.extend([
                                f"        <div class='mb-4 flex justify-center'>",
                                f"          <div class='inline-block mx-auto overflow-x-auto max-w-full'>",
                                f"            <div id='{div_id}'></div>",
                                f"          </div>",
                                f"        </div>",
                                "        <script>",
                                f"          vegaEmbed('#{div_id}', {json.dumps(spec)}, {{",
                                "            actions: false,",
                                "            renderer: 'canvas',",
                                "          })",
                                "          .catch(console.error);",
                                "        </script>"
                            ])
                        elif library == "antv":
                            spec = visualisation["specification"]
                            html_lines.extend([
                                f"        <div class='mb-4 flex justify-center'>",
                                f"          <div class='inline-block mx-auto overflow-x-auto max-w-full'>",
                                f"            <div id='{div_id}'></div>",
                                f"          </div>",
                                f"        </div>",
                                f"        {spec}"
                            ])
                        elif library == "altair":
                            spec = visualisation["specification"]
                            html_lines.extend([
                                f"        <div class='mb-4 flex justify-center'>",
                                f"          <div class='inline-block mx-auto overflow-x-auto max-w-full'>",
                                f"            <div id='{div_id}'></div>",
                                f"            {spec}",
                                f"          </div>",
                                f"        </div>",
                            ])
                        else:
                            html_lines.append(f"        <p class='text-red-600'>Error loading visualization</p>")

                    except json.JSONDecodeError:
                        html_lines.append("        <p class='text-red-600'>Error loading visualization</p>")

                # Add visualization description
                if vis_text:
                    html_lines.extend([
                        "        <p class='text-sm text-gray-600 mt-2 text-center'>",
                        f"          {markdown_to_html(vis_text)}",
                        "        </p>"
                    ])

                # TODO: Add facts if available


                html_lines.extend([
                    "      </div>",
                    "    </div>"
                ])

        # End of Section
        html_lines.extend([
            "  </div>",
            "</section>"
        ])

    html_lines.extend([
        "<script>",
        "document.addEventListener('DOMContentLoaded', function () {",
        "  const toc = document.getElementById('toc');",
        "  if (!toc) return;",
        "",
        "  const headers = document.querySelectorAll('article h2, article h3');",
        "  headers.forEach(h => {",
        "    let id = h.textContent.trim().replace(/\\s+/g, '-').toLowerCase();",
        "    h.setAttribute('id', id);",
        "",
        "    const link = document.createElement('a');",
        "    link.href = '#' + id;",
        "    link.textContent = h.textContent;",
        "    link.className = (h.tagName === 'H2'",
        "      ? 'block font-semibold mb-1'"
        "      : 'block ml-4 text-gray-500') + ' hover:text-primary-600';",
        "",
        "    toc.appendChild(link);",
        "  });",
        "});",
        "</script>"
    ])

    # Footer
    html_lines.extend([
        "    </article>",
        "  </main>",

        "  <footer class='border-t mt-12 py-6 text-center text-sm text-gray-500'>",
        "    © 2025 Agentic VIS",
        "  </footer>",

        # TOC Script
        "  <script>",
        "  document.addEventListener('DOMContentLoaded', function () {",
        "  const tocList = document.getElementById('toc-list');",
        "  const headers = document.querySelectorAll('article h2, article h3');",
        "  const links = [];",
        "",
        "  headers.forEach((h, idx) => {",

            # Map Unique IDs to headers and create TOC items
        "    let id = h.textContent.trim().replace(/\\s+/g, '-').toLowerCase() + '-' + idx;",
        "    h.setAttribute('id', id);",
        "",
        "    const li = document.createElement('li');",
        "    li.className = 'flex items-center space-x-2';",
        "",
        "    const dot = document.createElement('span');",
        "    dot.className = 'w-2 h-2 rounded-full border border-gray-300';",
        "    li.appendChild(dot);",
        "",
        "    const link = document.createElement('a');",
        "    link.href = '#' + id;",
        "",

            # Split long text into multiple lines
        "    const text = h.textContent.trim();",
        "    link.textContent = text.length > 40 ? text.slice(0, 37) + '…' : text;",
        "    link.title = text;",
        "",
        "    link.className = 'block truncate text-gray-400 hover:text-gray-600';",
        "    if (h.tagName === 'H3') {",
        "      link.className += ' ml-4';",
        "    }",
        "",

            # Smooth scrolling
        "    link.addEventListener('click', function (e) {",
        "      e.preventDefault();",
        "      const target = document.getElementById(id);",
        "      if (target) {",
        "        const y = target.getBoundingClientRect().top + window.scrollY - 100;",
        "        window.scrollTo({ top: y, behavior: 'smooth' });",
        "      }",
        "    });",
        "",
        "    li.appendChild(link);",
        "    tocList.appendChild(li);",
        "    links.push({id, link, el: h, dot});",
        "  });",
        "",

            # Highlight current section
        "  function onScroll() {",
        "    let scrollPos = document.documentElement.scrollTop || document.body.scrollTop;",
        "    let current;",
        "    links.forEach(item => {",
        "      if (item.el.offsetTop - 120 <= scrollPos) {",
        "        current = item;",
        "      }",
        "    });",
        "    links.forEach(item => {",
        "      item.link.classList.remove('font-semibold', 'text-black');",
        "      item.link.classList.add('text-gray-400');",
        "      item.dot.className = 'w-2 h-2 rounded-full border border-gray-300';",
        "    });",
        "    if (current) {",
        "      current.link.classList.remove('text-gray-400');",
        "      current.link.classList.add('font-semibold', 'text-black');",
        "      current.dot.className = 'w-2 h-2 rounded-full bg-black';",
        "    }",
        "  }",
        "  window.addEventListener('scroll', onScroll);",
        "  onScroll();",
        "});",
        "</script>",

        # Progress Bar Script
        "<script>",
        "window.addEventListener('scroll', function () {",
        "  const docHeight = document.documentElement.scrollHeight - window.innerHeight;",
        "  const scrollTop = window.scrollY || document.documentElement.scrollTop;",
        "  const progress = (scrollTop / docHeight) * 100;",
        "  document.getElementById('progress-bar').style.width = progress + '%';",
        "});",
        "</script>"
    
        "</body>",
        "</html>"
    ])

    # Clean up duplicate Graph declarations
    html_content = "\n".join(html_lines)
    html_content = clean_duplicate_graph_declarations(html_content)
    
    # Write out HTML file
    Path(output_path).write_text(html_content, encoding="utf-8")
    print(f"✓ HTML report generated: {output_path}")

def generate_html_from_state_file(state_file_path: str, output_path: str):
    """
    Generate HTML report from a state JSON file.
    
    Args:
        state_file_path: Path to the state JSON file
        output_path: Path to save the HTML report
    """
    try:
        # Load the state file
        with open(state_file_path, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
        
        # Generate HTML report
        generate_html_report(state_data, output_path)
        
        print(f"✓ HTML report generated from state file: {output_path}")
        
    except FileNotFoundError:
        print(f"Error: State file not found: {state_file_path}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in state file: {e}")
    except Exception as e:
        print(f"Error generating HTML from state file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Example usage
    state_file = "../../../outputs_sync/vis_report/thread_20250819_143215/state.json"
    output_file = "../../../outputs_sync/vis_report/thread_20250819_143215/report.html"
    
    if Path(state_file).exists():
        generate_html_from_state_file(state_file, output_file)
    else:
        print(f"State file not found: {state_file}")
        print("Please provide a valid state file path.")
