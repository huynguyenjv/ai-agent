"""R10 — cross-session memory store."""

from __future__ import annotations

from server.agent.memory_store import MemoryStore, SQLiteMemoryStorage, enable_memory


def _store():
    return MemoryStore(SQLiteMemoryStorage(":memory:"))


class TestMemoryStore:
    def test_remember_and_recall_keyword(self):
        m = _store()
        m.remember("user1", "user prefers pytest and black formatting")
        m.remember("user1", "the project uses fastapi and langgraph")
        out = m.recall("user1", "which test framework do we use", top_k=2)
        assert out and any("pytest" in r["content"] for r in out)

    def test_scope_isolation(self):
        m = _store()
        m.remember("user1", "secret note for user1")
        assert m.recall("user2", "secret note") == []

    def test_forget(self):
        m = _store()
        m.remember("u", "a")
        m.remember("u", "b")
        assert m.forget("u") == 2
        assert m.recall("u", "a") == []

    def test_empty_inputs_noop(self):
        m = _store()
        m.remember("", "x")
        m.remember("u", "")
        assert m.recall("u", "x") == []

    def test_enable_memory_off_by_default(self, monkeypatch):
        monkeypatch.delenv("ENABLE_MEMORY", raising=False)
        assert enable_memory() is False
