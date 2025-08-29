import subprocess
import tempfile
import os
import shutil
from helpers import load_dataset

def run_in_sandbox(code: str, python_path: str = "python3"):
    # 1. Write LLM generated code to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as f:
        f.write(code.encode())
        script_path = f.name

    try:
        # 2. Run code with subprocess, capture stdout and stderr
        result = subprocess.run(
            [python_path, script_path],  # Use provided python_path
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5  # 5 second timeout
        )

        return {
            "stdout": result.stdout,   # print() output
            "stderr": result.stderr,   # error output
            "exit_code": result.returncode
        }
    finally:
        os.unlink(script_path)  # 3. Delete temporary file


# # 测试代码
# llm_code = """
# print("Hello from sandbox!")
# for i in range(3):
#     print("i =", i)
# """

# result = run_in_sandbox(llm_code)

# print("标准输出:")
# print(result["stdout"])
# print("错误输出:")
# print(result["stderr"])
# print("退出码:", result["exit_code"])

def run_in_sandbox_with_venv(code: str, venv_path: str="./mini-template-env/"):
    # Activate virtual environment
    if os.name == 'nt':  # Windows
        python_path = os.path.join(venv_path, 'Scripts', 'python.exe')
    else:  # Unix/Linux/Mac
        python_path = os.path.join(venv_path, 'bin', 'python')
    
    return run_in_sandbox(code, python_path)


def run_visualization_in_sandbox(code: str, dataset_path: str, output_filename: str, memory_dir: str, venv_path: str = "./mini-template-env/"):
    """
    Run visualization code in sandbox and copy generated images to memory directory
    
    Args:
        code: LLM generated visualization code
        dataset_path: Dataset file path
        output_filename: Output image filename (e.g., "insight_1.png")
        memory_dir: Memory directory path
        venv_path: Virtual environment path
    
    Returns:
        dict: Dictionary containing execution results
    """
    # Create temporary working directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create complete script in temporary directory
        script_content = f"""
import pandas as pd
import altair as alt
import os

# Load dataset
df = pd.read_csv('{dataset_path}')

# User's visualization code
{code}

# Save image to temporary directory
if 'chart' in locals() and isinstance(chart, alt.Chart):
    temp_image_path = os.path.join('{temp_dir}', '{output_filename}')
    chart.save(temp_image_path)
    print(f"SUCCESS: Chart saved to {{temp_image_path}}")
else:
    print("ERROR: No valid chart object found")
    print("Available variables:", list(locals().keys()))
"""
        
        # Write script file
        script_path = os.path.join(temp_dir, "visualization_script.py")
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        try:
            # Determine Python path
            if os.name == 'nt':  # Windows
                python_path = os.path.join(venv_path, 'Scripts', 'python.exe')
            else:  # Unix/Linux/Mac
                python_path = os.path.join(venv_path, 'bin', 'python')
            
            # Run script
            result = subprocess.run(
                [python_path, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,  # Increase timeout to 30 seconds
                cwd=temp_dir  # Run in temporary directory
            )
            
            # Check if image was successfully generated
            temp_image_path = os.path.join(temp_dir, output_filename)
            if result.returncode == 0 and os.path.exists(temp_image_path):
                # Ensure memory directory exists
                os.makedirs(memory_dir, exist_ok=True)
                
                # Copy image to memory directory
                final_image_path = os.path.join(memory_dir, output_filename)
                shutil.copy2(temp_image_path, final_image_path)
                
                return {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.returncode,
                    "success": True,
                    "image_path": final_image_path
                }
            else:
                return {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.returncode,
                    "success": False,
                    "image_path": None
                }
                
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": "Execution timeout",
                "exit_code": 1,
                "success": False,
                "image_path": None
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "exit_code": 1,
                "success": False,
                "image_path": None
            }
