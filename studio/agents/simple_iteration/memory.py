import os
import json
from typing import Any, List, Dict, Tuple
from datetime import datetime


class SimpleMemory:
    """
    Very simple memory system that just saves states to files.
    Each memory instance is tied to a specific thread.
    """
    
    def __init__(self, thread_id: str | None = None, output_dir: str = "outputs/simple_iteration/"):
        self.thread_id = thread_id
        self.output_dir = output_dir
        self.states: List[Dict[str, Any]] = []  # list of states for this thread
        self._ensure_output_dir()

    def set_thread_id(self, thread_id: str):
        self.thread_id = thread_id
    
    def _ensure_output_dir(self):
        """Ensure the output directory exists"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def _get_thread_dir(self) -> str:
        """Get the directory for this thread"""
        if self.thread_id is None:
            # Use a default thread ID if none is set
            self.thread_id = "default_thread"
        
        thread_dir = os.path.join(self.output_dir, self.thread_id)
        if not os.path.exists(thread_dir):
            os.makedirs(thread_dir)
        return thread_dir
    
    def _save_backup(self):
        """Save a backup of all states to a file"""
        thread_dir = self._get_thread_dir()
        backup_file = os.path.join(thread_dir, "memory_backup.json")
        
        backup_data = {
            "thread_id": self.thread_id,
            "last_updated": datetime.now().isoformat(),
            "total_states": len(self.states),
            "states": self.states
        }
        
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False, default=str)
    
    def save_state(self, state: Dict[str, Any]) -> None:
        """Save a state to memory and backup file"""
        self.states.append(state)
        self._save_backup()
        print(f"Saved state {len(self.states)} for thread {self.thread_id}")
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get all states for this thread"""
        return self.states.copy()
    
    def export_questions_and_insights(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Export all questions and insights from memory with deduplication.
        
        Returns:
            Tuple containing:
            - List of unique questions (each question is a dict with 'question', 'handled', 'spec' fields)
            - List of unique insights (strings)
        """
        questions_seen = set()
        insights_seen = set()
        unique_questions = []
        unique_insights = []
        
        for state in self.states:
            # Extract questions
            if 'question' in state and state['question']:
                question = state['question']
                if isinstance(question, dict) and 'question' in question:
                    question_text = question['question']
                    if question_text and question_text not in questions_seen:
                        questions_seen.add(question_text)
                        unique_questions.append(question)
            
            # Extract insights
            if 'insights' in state and state['insights']:
                for insight in state['insights']:
                    if isinstance(insight, str) and insight.strip() and insight not in insights_seen:
                        insights_seen.add(insight)
                        unique_insights.append(insight)
        
        return unique_questions, unique_insights
    
    def clear(self) -> None:
        """Clear memory for this thread"""
        self.states = []
        # Also remove the thread directory
        thread_dir = self._get_thread_dir()
        if os.path.exists(thread_dir):
            import shutil
            shutil.rmtree(thread_dir)


# Global memory instance that can be accessed by all nodes
shared_memory = SimpleMemory(output_dir="outputs/simple_iteration/")
