import os
import json

from datetime import datetime


class Memory:
    """
    Manage intermediate states and results
    """

    def __init__(self, thread_id: str | None = None, output_dir: str = "outputs_sync/vis_report/"):
        if thread_id is None:
            self._generate_thread_id()
        else:
            self.thread_id = thread_id

        self.output_dir = output_dir
        self._ensure_output_dir()
        self.latest_state = None

    def _generate_thread_id(self):
        """Generate a thread ID if none is provided"""
        self.thread_id = f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _ensure_output_dir(self):
        """Ensure the output directory exists"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _get_thread_dir(self):
        thread_dir = os.path.join(self.output_dir, self.thread_id)
        if not os.path.exists(thread_dir):
            os.makedirs(thread_dir)
        return thread_dir

    def save_file(self, filepath: str, content: str):
        filepath = os.path.join(self._get_thread_dir(), filepath)
        with open(filepath, "w") as f:
            f.write(content)

    def save_state(self, state: dict):
        self.save_file("state.json", json.dumps(state, indent=4))
        print(f"📝 State saved to {self._get_thread_dir()}/state.json")
        self.latest_state = state

    def load_state_from_thread(self, thread_id: str):
        thread_dir = os.path.join(self.output_dir, thread_id)
        if not os.path.exists(thread_dir):
            raise FileNotFoundError(f"Thread directory {thread_dir} not found")
        with open(os.path.join(thread_dir, "state.json"), "r") as f:
            return json.load(f)


memory = Memory(output_dir="outputs_sync/vis_report/")
