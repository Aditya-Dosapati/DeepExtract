"""Run a credential-safe smoke test against a live backend instance."""

import json
import os
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")


def call_api(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    token: str | None = None,
) -> tuple[int, dict[str, Any]]:
    """Call one JSON endpoint without printing credentials or bearer tokens."""
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(  # noqa: S310 - BASE_URL is operator-controlled for this smoke test.
        f"{BASE_URL}{path}", data=body, headers=headers, method=method
    )
    try:
        with urlopen(request, timeout=10) as response:  # noqa: S310
            content = json.loads(response.read()) if response.status != 204 else {}
            return response.status, content
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


def main() -> None:
    """Verify registration, login, authenticated identity, and thread creation."""
    suffix = uuid4().hex[:12]
    email = f"api-smoke-{suffix}@example.com"
    password = f"Smoke-{uuid4().hex}-A7!"
    register_status, _ = call_api(
        "POST",
        "/api/v1/auth/register",
        {"username": f"smoke-{suffix}", "email": email, "password": password},
    )
    login_status, login = call_api(
        "POST", "/api/v1/auth/login", {"email": email, "password": password}
    )
    token = str(login.get("access_token", ""))
    me_status, _ = call_api("GET", "/api/v1/users/me", token=token)
    thread_status, _ = call_api("POST", "/api/v1/threads", {"title": "Smoke test"}, token=token)
    statuses = {
        "register": register_status,
        "login": login_status,
        "current_user": me_status,
        "create_thread": thread_status,
    }
    print(json.dumps(statuses, sort_keys=True))
    if statuses != {"register": 201, "login": 200, "current_user": 200, "create_thread": 201}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
