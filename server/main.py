import json
import logging
import os
import secrets
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from mem0 import Memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()

# JSON map of bearer API key -> trusted identity. Example:
# MEM0_API_KEYS='{"<user-key>": {"user_id": "alice"}, "<admin-key>": {"user_id": "admin", "is_admin": true}}'
try:
    API_KEYS = json.loads(os.environ.get("MEM0_API_KEYS", "{}"))
except json.JSONDecodeError as exc:
    raise RuntimeError("MEM0_API_KEYS must contain valid JSON") from exc

if not isinstance(API_KEYS, dict) or not API_KEYS:
    raise RuntimeError("MEM0_API_KEYS must be a non-empty JSON object")
for api_key, identity in API_KEYS.items():
    if (
        not isinstance(api_key, str)
        or not api_key
        or not isinstance(identity, dict)
        or not isinstance(identity.get("user_id"), str)
        or not identity["user_id"].strip()
        or not isinstance(identity.get("is_admin", False), bool)
    ):
        raise RuntimeError(
            "Each MEM0_API_KEYS entry must map a key to a user_id and optional boolean is_admin"
        )


POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "postgres")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
POSTGRES_COLLECTION_NAME = os.environ.get("POSTGRES_COLLECTION_NAME", "memories")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "mem0graph")

MEMGRAPH_URI = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
MEMGRAPH_USERNAME = os.environ.get("MEMGRAPH_USERNAME", "memgraph")
MEMGRAPH_PASSWORD = os.environ.get("MEMGRAPH_PASSWORD", "mem0graph")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HISTORY_DB_PATH = os.environ.get("HISTORY_DB_PATH", "/app/history/history.db")

DEFAULT_CONFIG = {
    "version": "v1.1",
    "vector_store": {
        "provider": "pgvector",
        "config": {
            "host": POSTGRES_HOST,
            "port": int(POSTGRES_PORT),
            "dbname": POSTGRES_DB,
            "user": POSTGRES_USER,
            "password": POSTGRES_PASSWORD,
            "collection_name": POSTGRES_COLLECTION_NAME,
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {"url": NEO4J_URI, "username": NEO4J_USERNAME, "password": NEO4J_PASSWORD},
    },
    "llm": {"provider": "openai", "config": {"api_key": OPENAI_API_KEY, "temperature": 0.2, "model": "gpt-4.1-nano-2025-04-14"}},
    "embedder": {"provider": "openai", "config": {"api_key": OPENAI_API_KEY, "model": "text-embedding-3-small"}},
    "history_db_path": HISTORY_DB_PATH,
}


MEMORY_INSTANCE = Memory.from_config(DEFAULT_CONFIG)

app = FastAPI(
    title="Mem0 REST APIs",
    description="A REST API for managing and searching memories for your AI Agents and Apps.",
    version="1.0.0",
)


bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Principal:
    user_id: str
    is_admin: bool = False


def current_principal(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Principal:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Bearer API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    supplied_key = credentials.credentials.encode("utf-8")
    matched_identity = None
    for configured_key, identity in API_KEYS.items():
        matches = secrets.compare_digest(supplied_key, configured_key.encode("utf-8"))
        if matches:
            matched_identity = identity
    if matched_identity is not None:
        return Principal(
            user_id=matched_identity["user_id"], is_admin=matched_identity.get("is_admin", False)
        )

    raise HTTPException(
        status_code=401,
        detail="Invalid API key",
        headers={"WWW-Authenticate": "Bearer"},
    )


def admin_principal(principal: Principal = Depends(current_principal)) -> Principal:
    if not principal.is_admin:
        raise HTTPException(status_code=403, detail="Admin API key required")
    return principal


def enforce_owner(requested_user_id: Optional[str], principal: Principal) -> str:
    if requested_user_id is not None and requested_user_id != principal.user_id:
        raise HTTPException(status_code=403, detail="Cannot access another user's memories")
    return principal.user_id


def reject_scope_overrides(filters: Optional[Dict[str, Any]]) -> None:
    """Prevent caller-controlled filters from changing tenant or session scope."""
    scope_fields = {"user_id", "agent_id", "run_id"}

    def check(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested_value in value.items():
                if key in scope_fields:
                    raise HTTPException(
                        status_code=400,
                        detail=f"'{key}' must be supplied as a top-level scope, not in filters",
                    )
                check(nested_value)
        elif isinstance(value, list):
            for nested_value in value:
                check(nested_value)

    check(filters)


def get_owned_memory(memory_id: str, user_id: str) -> Dict[str, Any]:
    memory = MEMORY_INSTANCE.get(memory_id)
    if not isinstance(memory, dict) or memory.get("user_id") != user_id:
        # Conceal whether another user's memory ID exists.
        raise HTTPException(status_code=404, detail="Memory not found")
    return memory


class Message(BaseModel):
    role: str = Field(..., description="Role of the message (user or assistant).")
    content: str = Field(..., description="Message content.")


class MemoryCreate(BaseModel):
    messages: List[Message] = Field(..., description="List of messages to store.")
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query.")
    user_id: Optional[str] = None
    run_id: Optional[str] = None
    agent_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None


@app.post("/configure", summary="Configure Mem0")
def set_config(config: Dict[str, Any], _principal: Principal = Depends(admin_principal)):
    """Set memory configuration."""
    global MEMORY_INSTANCE
    MEMORY_INSTANCE = Memory.from_config(config)
    return {"message": "Configuration set successfully"}


@app.post("/memories", summary="Create memories")
def add_memory(memory_create: MemoryCreate, principal: Principal = Depends(current_principal)):
    """Store new memories under the authenticated owner."""
    enforce_owner(memory_create.user_id, principal)
    params = {
        k: v
        for k, v in memory_create.model_dump().items()
        if v is not None and k not in {"messages", "user_id"}
    }
    try:
        response = MEMORY_INSTANCE.add(
            messages=[m.model_dump() for m in memory_create.messages], user_id=principal.user_id, **params
        )
        return JSONResponse(content=response)
    except Exception as e:
        logging.exception("Error in add_memory:")  # This will log the full traceback
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories", summary="Get memories")
def get_all_memories(
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    principal: Principal = Depends(current_principal),
):
    """Retrieve only memories belonging to the authenticated owner."""
    owner_id = enforce_owner(user_id, principal)
    try:
        params = {
            k: v
            for k, v in {"user_id": owner_id, "run_id": run_id, "agent_id": agent_id}.items()
            if v is not None
        }
        return MEMORY_INSTANCE.get_all(**params)
    except Exception as e:
        logging.exception("Error in get_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories/{memory_id}", summary="Get a memory")
def get_memory(memory_id: str, principal: Principal = Depends(current_principal)):
    """Retrieve a specific memory by ID if it belongs to the authenticated owner."""
    return get_owned_memory(memory_id, principal.user_id)


@app.post("/search", summary="Search memories")
def search_memories(search_req: SearchRequest, principal: Principal = Depends(current_principal)):
    """Search only memories belonging to the authenticated owner."""
    owner_id = enforce_owner(search_req.user_id, principal)
    reject_scope_overrides(search_req.filters)
    try:
        params = {
            k: v
            for k, v in search_req.model_dump().items()
            if v is not None and k not in {"query", "user_id"}
        }
        return MEMORY_INSTANCE.search(query=search_req.query, user_id=owner_id, **params)
    except Exception as e:
        logging.exception("Error in search_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/memories/{memory_id}", summary="Update a memory")
def update_memory(
    memory_id: str,
    updated_memory: Dict[str, Any],
    principal: Principal = Depends(current_principal),
):
    """Update an existing memory with new content.
    
    Args:
        memory_id (str): ID of the memory to update
        updated_memory (str): New content to update the memory with
        
    Returns:
        dict: Success message indicating the memory was updated
    """
    get_owned_memory(memory_id, principal.user_id)
    try:
        return MEMORY_INSTANCE.update(memory_id=memory_id, data=updated_memory)
    except Exception as e:
        logging.exception("Error in update_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories/{memory_id}/history", summary="Get memory history")
def memory_history(memory_id: str, principal: Principal = Depends(current_principal)):
    """Retrieve history for a memory owned by the authenticated user."""
    get_owned_memory(memory_id, principal.user_id)
    try:
        return MEMORY_INSTANCE.history(memory_id=memory_id)
    except Exception as e:
        logging.exception("Error in memory_history:")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories/{memory_id}", summary="Delete a memory")
def delete_memory(memory_id: str, principal: Principal = Depends(current_principal)):
    """Delete a specific memory if it belongs to the authenticated user."""
    get_owned_memory(memory_id, principal.user_id)
    try:
        MEMORY_INSTANCE.delete(memory_id=memory_id)
        return {"message": "Memory deleted successfully"}
    except Exception as e:
        logging.exception("Error in delete_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories", summary="Delete all memories")
def delete_all_memories(
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    principal: Principal = Depends(current_principal),
):
    """Delete only memories belonging to the authenticated user."""
    owner_id = enforce_owner(user_id, principal)
    try:
        params = {
            k: v
            for k, v in {"user_id": owner_id, "run_id": run_id, "agent_id": agent_id}.items()
            if v is not None
        }
        MEMORY_INSTANCE.delete_all(**params)
        return {"message": "All relevant memories deleted"}
    except Exception as e:
        logging.exception("Error in delete_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset", summary="Reset all memories")
def reset_memory(_principal: Principal = Depends(admin_principal)):
    """Completely reset stored memories."""
    try:
        MEMORY_INSTANCE.reset()
        return {"message": "All memories reset"}
    except Exception as e:
        logging.exception("Error in reset_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", summary="Redirect to the OpenAPI documentation", include_in_schema=False)
def home():
    """Redirect to the OpenAPI documentation."""
    return RedirectResponse(url="/docs")
