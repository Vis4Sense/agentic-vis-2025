import os
import json

from datetime import datetime

from agents.vis_report.memory import memory as global_memory


class Memory:
    """
    Manage intermediate states and results
    """

    def __init__(self):
        self.history_states = []
        

    def add_state(self, state: dict):
        self.history_states.append(state)

    def update_state(self, state: dict):
        self.add_state(state)
        global_memory.save_state(state)

    def get_latest_state(self):
        return self.history_states[-1]


memory = Memory()
