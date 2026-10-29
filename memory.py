"""Bounded in-process conversation memory, serialized per session."""
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
import threading
from typing import Iterator


class MemoryCapacityError(RuntimeError):
    pass


@dataclass
class Session:
    turns: deque = field(default_factory=deque)
    lock: threading.RLock = field(default_factory=threading.RLock)

    def messages(self) -> list[dict]:
        return deepcopy([message for turn in self.turns for message in turn])

    def append(self, turn: list[dict]) -> None:
        # Store complete turns, keeping assistant tool calls paired with results.
        self.turns.append(deepcopy(turn))


class ConversationMemory:
    def __init__(self, max_turns: int = 10, max_sessions: int = 1000):
        self.max_turns = max_turns
        self.max_sessions = max_sessions
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    @contextmanager
    def session(self, session_id: str) -> Iterator[Session]:
        if not isinstance(session_id, str) or not 1 <= len(session_id) <= 128 or not session_id.strip():
            raise ValueError("session_id must contain 1–128 characters")
        with self._lock:
            if session_id not in self._sessions:
                if len(self._sessions) >= self.max_sessions:
                    raise MemoryCapacityError("会话数量已达上限，请重启服务或配置更大的 MAX_SESSIONS。")
                self._sessions[session_id] = Session(turns=deque(maxlen=self.max_turns))
            session = self._sessions[session_id]
        with session.lock:
            yield session

    def clear(self, session_id: str) -> None:
        with self.session(session_id) as session:
            session.turns.clear()
