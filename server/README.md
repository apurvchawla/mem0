# Mem0 REST API Server

Mem0 provides a REST API server (written using FastAPI). Users can perform all operations through REST endpoints. The API also includes OpenAPI documentation, accessible at `/docs` when the server is running.

## Features

- **Create memories:** Create memories based on messages for a user, agent, or run.
- **Retrieve memories:** Get all memories for a given user, agent, or run.
- **Search memories:** Search stored memories based on a query.
- **Update memories:** Update an existing memory.
- **Delete memories:** Delete a specific memory or all memories for a user, agent, or run.
- **Reset memories:** Globally reset all memories (admin API key required).
- **OpenAPI Documentation:** Accessible via `/docs` endpoint.

## Authentication and ownership

Every memory endpoint requires an API key in the bearer authorization header. Configure `MEM0_API_KEYS` as a JSON object mapping each unique key to a trusted `user_id`; an admin key also needs `"is_admin": true` to call `/configure` or `/reset`.

```dotenv
MEM0_API_KEYS='{"<user-api-key>": {"user_id": "alice"}, "<admin-api-key>": {"user_id": "admin", "is_admin": true}}'
```

Use strong, secret values in deployment configuration—do not use the placeholder keys shown in `.env.example`. Requests are automatically scoped to the mapped `user_id`; passing a different `user_id` is rejected. Send the key as `Authorization: Bearer <api-key>`.

## Running the server

Follow the instructions in the [docs](https://docs.mem0.ai/open-source/features/rest-api) to run the server. Set `MEM0_API_KEYS` before starting; the server refuses to start if the key map is missing or invalid.
