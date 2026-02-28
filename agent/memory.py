from typing import Dict, List

from utils.config import logger


class ConversationMemory:
    """Simple in-memory chat history for the agent."""

    def __init__(self, max_turns: int = 10) -> None:
        self.max_turns = max_turns
        self._messages: List[Dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        # Keep only the most recent messages
        if len(self._messages) > self.max_turns:
            self._messages = self._messages[-self.max_turns :]
        logger.debug("Memory appended role=%s content_len=%d", role, len(content))

    def get(self) -> List[Dict[str, str]]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()
        logger.info("Conversation memory cleared")

