"""
Persistent Memory & Athlete Profile Manager for the AI Personal Trainer.
Handles SQLite-backed conversation history (via LangChain SQLChatMessageHistory)
and structured athlete profile storage (biomechanical constraints, split, goals)
so context persists across CLI restarts.
"""

import os
import json
from typing import Dict, Any, Optional
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory


class AthleteMemoryManager:
    def __init__(
        self,
        db_path: str = "sqlite:///data/athlete_memory.db",
        profiles_dir: str = "data/profiles"
    ):
        """
        Initializes persistent SQLite chat history and JSON athlete profile storage.
        """
        self.db_path = db_path
        self.profiles_dir = profiles_dir

        # Ensure data directories exist
        os.makedirs("data", exist_ok=True)
        os.makedirs(self.profiles_dir, exist_ok=True)

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Returns a persistent SQLite-backed chat history instance for the given session_id.
        Can be passed directly into RunnableWithMessageHistory.
        """
        return SQLChatMessageHistory(
            session_id=session_id,
            connection=self.db_path
        )

    def save_athlete_profile(self, session_id: str, profile_data: Dict[str, Any]) -> None:
        """
        Saves or updates structured long-term attributes for an athlete
        (e.g., height, weight, training split, biomechanical constraints).
        """
        file_path = os.path.join(self.profiles_dir, f"{session_id}.json")
        existing = self.load_athlete_profile(session_id)
        existing.update(profile_data)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=4, ensure_ascii=False)

    def load_athlete_profile(self, session_id: str) -> Dict[str, Any]:
        """
        Loads the athlete's structured profile from disk, or returns an empty dict.
        """
        file_path = os.path.join(self.profiles_dir, f"{session_id}.json")
        if not os.path.exists(file_path):
            return {}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def format_profile_for_prompt(self, session_id: str) -> str:
        """
        Formats the athlete's long-term profile into a clean text block
        ready to be injected into the LLM system prompt.
        """
        profile = self.load_athlete_profile(session_id)
        if not profile:
            return "No structured athlete profile on file yet."

        lines = [f"- {key.replace('_', ' ').title()}: {value}" for key, value in profile.items()]
        return "\n".join(lines)

    def clear_session_history(self, session_id: str) -> None:
        """Clears all chat messages for a specific session_id."""
        history = self.get_session_history(session_id)
        history.clear()


if __name__ == "__main__":
    # Quick diagnostic test for persistent memory and profile
    memory_manager = AthleteMemoryManager()
    test_id = "athlete_01"

    # Save sample athlete constraints & split
    memory_manager.save_athlete_profile(test_id, {
        "goal": "Hypertrophy & Lean Bulking",
        "training_split": "Non-repeating ABC (3x/week)",
        "biomechanical_constraints": "Avoid lying leg curl due to stature/alignment; substitute with seated or standing leg curl.",
        "equipment_notes": "Uses Smith machine for heavy solo bench presses and lifting straps for heavy pulls."
    })

    print("--- Formatted Athlete Profile ---")
    print(memory_manager.format_profile_for_prompt(test_id))

    # Test persistent SQLite chat history
    chat_history = memory_manager.get_session_history(test_id)
    chat_history.add_user_message("Log my leg day preference: seated leg curls only.")
    print(f"\nMessages stored in SQLite for '{test_id}': {len(chat_history.messages)}")