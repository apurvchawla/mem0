import importlib
import json
import os
import sys
import types
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient


# Import the API with a lightweight Mem0 stub so the unit tests never need a
# database, provider credentials, or a live vector store.
_previous_mem0 = sys.modules.get("mem0")
_previous_api_keys = os.environ.get("MEM0_API_KEYS")
os.environ["MEM0_API_KEYS"] = json.dumps(
    {
        "alice-key": {"user_id": "alice"},
        "bob-key": {"user_id": "bob"},
        "admin-key": {"user_id": "admin", "is_admin": True},
    }
)

_fake_mem0 = types.ModuleType("mem0")
_fake_memory_class = type("Memory", (), {"from_config": staticmethod(lambda _config: Mock())})
_fake_mem0.Memory = _fake_memory_class
sys.modules["mem0"] = _fake_mem0
try:
    api = importlib.import_module("server.main")
finally:
    if _previous_mem0 is None:
        sys.modules.pop("mem0", None)
    else:
        sys.modules["mem0"] = _previous_mem0
    if _previous_api_keys is None:
        os.environ.pop("MEM0_API_KEYS", None)
    else:
        os.environ["MEM0_API_KEYS"] = _previous_api_keys

client = TestClient(api.app)


def auth(key="alice-key"):
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def memory(monkeypatch):
    fake_memory = Mock()
    fake_memory.add.return_value = [{"id": "memory-1", "memory": "hello"}]
    fake_memory.get_all.return_value = []
    fake_memory.search.return_value = {"results": []}
    fake_memory.get.return_value = {"id": "memory-1", "memory": "hello", "user_id": "alice"}
    monkeypatch.setattr(api, "MEMORY_INSTANCE", fake_memory)
    return fake_memory


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("GET", "/memories", {}),
        ("GET", "/memories/memory-1", {}),
        ("POST", "/memories", {"json": {"messages": [{"role": "user", "content": "hello"}]}}),
        ("POST", "/search", {"json": {"query": "hello"}}),
        ("PUT", "/memories/memory-1", {"json": {"memory": "updated"}}),
        ("GET", "/memories/memory-1/history", {}),
        ("DELETE", "/memories/memory-1", {}),
        ("DELETE", "/memories", {}),
        ("POST", "/configure", {"json": {}}),
        ("POST", "/reset", {}),
    ],
)
def test_all_memory_and_admin_routes_require_authentication(memory, method, path, kwargs):
    response = client.request(method, path, **kwargs)
    assert response.status_code == 401


def test_invalid_api_key_is_rejected(memory):
    response = client.get("/memories", headers=auth("not-a-valid-key"))
    assert response.status_code == 401


def test_create_always_uses_authenticated_owner(memory):
    response = client.post(
        "/memories",
        headers=auth(),
        json={"messages": [{"role": "user", "content": "hello"}], "agent_id": "agent-1"},
    )

    assert response.status_code == 200
    memory.add.assert_called_once_with(
        messages=[{"role": "user", "content": "hello"}], user_id="alice", agent_id="agent-1"
    )


def test_conflicting_user_id_is_rejected_for_create_and_search(memory):
    create = client.post(
        "/memories",
        headers=auth(),
        json={"messages": [{"role": "user", "content": "hello"}], "user_id": "bob"},
    )
    search = client.post("/search", headers=auth(), json={"query": "hello", "user_id": "bob"})

    assert create.status_code == 403
    assert search.status_code == 403
    memory.add.assert_not_called()
    memory.search.assert_not_called()


def test_list_and_delete_all_reject_conflicting_user_id(memory):
    listed = client.get("/memories?user_id=bob", headers=auth())
    deleted = client.delete("/memories?user_id=bob", headers=auth())

    assert listed.status_code == deleted.status_code == 403
    memory.get_all.assert_not_called()
    memory.delete_all.assert_not_called()


def test_list_search_and_delete_all_are_scoped_to_authenticated_owner(memory):
    listed = client.get("/memories?agent_id=agent-1", headers=auth())
    searched = client.post("/search", headers=auth(), json={"query": "hello", "run_id": "run-1"})
    deleted = client.delete("/memories?agent_id=agent-1", headers=auth())

    assert listed.status_code == searched.status_code == deleted.status_code == 200
    memory.get_all.assert_called_once_with(user_id="alice", agent_id="agent-1")
    memory.search.assert_called_once_with(query="hello", user_id="alice", run_id="run-1")
    memory.delete_all.assert_called_once_with(user_id="alice", agent_id="agent-1")


def test_search_filters_cannot_override_owner_or_session_scope(memory):
    response = client.post(
        "/search",
        headers=auth(),
        json={"query": "hello", "filters": {"OR": [{"user_id": "bob"}, {"topic": "work"}]}},
    )

    assert response.status_code == 400
    memory.search.assert_not_called()


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("GET", "/memories/memory-1", {}),
        ("PUT", "/memories/memory-1", {"json": {"memory": "updated"}}),
        ("GET", "/memories/memory-1/history", {}),
        ("DELETE", "/memories/memory-1", {}),
    ],
)
def test_memory_id_operations_hide_other_users_memories(memory, method, path, kwargs):
    memory.get.return_value = {"id": "memory-1", "memory": "private", "user_id": "bob"}

    response = client.request(method, path, headers=auth("alice-key"), **kwargs)

    assert response.status_code == 404
    memory.update.assert_not_called()
    memory.history.assert_not_called()
    memory.delete.assert_not_called()


def test_owner_can_read_own_memory(memory):
    response = client.get("/memories/memory-1", headers=auth("alice-key"))

    assert response.status_code == 200
    assert response.json()["user_id"] == "alice"
    memory.get.assert_called_once_with("memory-1")


def test_non_admin_cannot_configure_or_reset(memory):
    configure = client.post("/configure", headers=auth("alice-key"), json={})
    reset = client.post("/reset", headers=auth("alice-key"))

    assert configure.status_code == reset.status_code == 403
    memory.reset.assert_not_called()


def test_admin_can_configure(memory, monkeypatch):
    configured_memory = Mock()
    from_config = Mock(return_value=configured_memory)
    monkeypatch.setattr(api.Memory, "from_config", from_config)

    response = client.post("/configure", headers=auth("admin-key"), json={"test": True})

    assert response.status_code == 200
    from_config.assert_called_once_with({"test": True})
    assert api.MEMORY_INSTANCE is configured_memory


def test_admin_can_reset(memory):
    response = client.post("/reset", headers=auth("admin-key"))

    assert response.status_code == 200
    memory.reset.assert_called_once_with()
