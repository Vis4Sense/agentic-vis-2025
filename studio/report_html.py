import re
import json
from pathlib import Path

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

def generate_html_report(output_state: dict, output_path: str):
    """
    Builds an interactive HTML report from output_state["message"],
    treating each ```json ...``` block as a Vega-Lite spec,
    interleaving it with narrative converted from simple Markdown.
    """
    content = output_state["message"]
    fence_re = re.compile(r'```json\s*\n(.*?)```', re.DOTALL)
    parts = []
    last_end = 0

    # 1. Split into narrative vs. JSON spec blocks
    for m in fence_re.finditer(content):
        narrative = content[last_end:m.start()].strip()
        if narrative:
            parts.append(("html", markdown_to_html(narrative)))
        json_code = m.group(1).strip()
        try:
            spec = json.loads(json_code)
            parts.append(("vega", spec))
        except json.JSONDecodeError:
            # fallback: show code block
            parts.append(("html", f'<pre><code>{json_code}</code></pre>'))
        last_end = m.end()

    # tail narrative
    tail = content[last_end:].strip()
    if tail:
        parts.append(("html", markdown_to_html(tail)))

    # 2. Build the HTML document
    html_lines = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='utf-8'>",
        "  <title>Interactive Vega-Lite Report</title>",
        "  <script src='https://cdn.jsdelivr.net/npm/vega@5'></script>",
        "  <script src='https://cdn.jsdelivr.net/npm/vega-lite@5'></script>",
        "  <script src='https://cdn.jsdelivr.net/npm/vega-embed@6'></script>",
        "  <style>",
        "    body { font-family: sans-serif; margin: 2em; }",
        "    h3 { margin-top: 1em; }",
        "    hr { margin: 1.5em 0; }",
        "  </style>",
        "</head>",
        "<body>",
    ]

    # 3. Interleave narrative and charts
    vis_counter = 0
    for kind, data in parts:
        if kind == "html":
            html_lines.append(data)
        else:  # vega spec
            div_id = f"vis{vis_counter}"
            html_lines.append(f"  <div id='{div_id}'></div>")
            spec_json = json.dumps(data)
            html_lines.extend([
                "  <script>",
                f"    vegaEmbed('#{div_id}', {spec_json})",
                "      .catch(console.error);",
                "  </script>",
            ])
            vis_counter += 1

    html_lines.extend([
        "</body>",
        "</html>"
    ])

    # 4. Write out in html file
    Path(output_path).write_text("\n".join(html_lines), encoding="utf-8")
