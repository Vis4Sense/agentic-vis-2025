import re
import io
import matplotlib
# 1. Use headless Agg backend so plt.show() can't close figures
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

def generate_pdf_report(output_state: dict, output_path: str):
    content = output_state["message"]

    # 1. Prepare pdf
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # 2. One shared namespace so later blocks can reuse variables
    ns = {"plt": plt}
    plt.show = lambda *a, **k: None  # no-op show

    # 3. Iterate through all python code blocks
    pattern = re.compile(r'```python\s*\n(.*?)```', re.DOTALL)
    pos = 0
    for m in pattern.finditer(content):
        # 3a. Narrative before this block
        narrative = content[pos:m.start()].strip()
        if narrative:
            story.append(Paragraph(narrative, styles["Normal"]))
            story.append(Spacer(1, 0.2 * inch))

        # 3b. Execute this code block
        code = m.group(1)
        exec(code, ns)

        # 3c. Capture any new figures this block produced
        for num in plt.get_fignums():
            fig = plt.figure(num)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            plt.close(fig)

            img = Image(buf, width=6.5 * inch, height=4 * inch)
            img.hAlign = "CENTER"
            story.append(img)
            story.append(Spacer(1, 0.2 * inch))

        pos = m.end()

    # 4. Anything after the last code block
    tail = content[pos:].strip()
    if tail:
        story.append(Paragraph(tail, styles["Normal"]))
        story.append(Spacer(1, 0.2 * inch))

    # 5. Build it
    doc.build(story)
