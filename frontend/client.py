"""High-performance typed API client for the DeepExtract dashboard."""

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx


class APIError(RuntimeError):
    """Safe API error suitable for display in the dashboard."""


class APIClient:
    """Call the backend with persistent connection pooling and robust error handling."""

    def __init__(self, base_url: str, token: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._session = httpx.Client(
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        )

    @property
    def headers(self) -> dict[str, str]:
        """Build authorization headers only when a session is authenticated."""
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        headers = {**self.headers, **kwargs.pop("headers", {})}
        timeout = kwargs.pop("timeout", 60)
        try:
            response = self._session.request(
                method,
                f"{self.base_url}{path}",
                headers=headers,
                timeout=timeout,
                **kwargs,
            )
        except httpx.RequestError as exc:
            raise APIError(
                "The DeepExtract backend is unreachable. Please check system status."
            ) from exc
        self._raise_for_status(response)
        return response

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        """Raise a safe dashboard error for an unsuccessful backend response."""
        if not response.is_error:
            return
        try:
            payload = response.json()
            if isinstance(payload, dict):
                error_obj = payload.get("error", {})
                if isinstance(error_obj, dict):
                    message = error_obj.get("message", "Request failed")
                else:
                    message = str(payload.get("detail", "Request failed"))
            else:
                message = "Request failed"
        except (ValueError, TypeError):
            message = f"Server returned error status {response.status_code}"
        raise APIError(str(message))

    def register(self, username: str, email: str, password: str) -> None:
        """Create an account."""
        self._request(
            "POST",
            "/api/v1/auth/register",
            json={"username": username, "email": email, "password": password},
        )

    def login(self, email: str, password: str) -> str:
        """Exchange credentials for an access token."""
        response = self._request(
            "POST", "/api/v1/auth/login", json={"email": email, "password": password}
        )
        return str(response.json()["access_token"])

    def current_user(self) -> dict[str, Any]:
        """Return the current authenticated account."""
        return dict(self._request("GET", "/api/v1/users/me").json())

    def metrics(self) -> dict[str, Any]:
        """Return owner-scoped knowledge base and activity metrics."""
        return dict(self._request("GET", "/api/v1/metrics/overview").json())

    def health(self, path: str) -> tuple[bool, float]:
        """Probe one public health endpoint and report elapsed milliseconds."""
        started = time.perf_counter()
        try:
            self._request("GET", path, timeout=5)
        except APIError:
            return False, (time.perf_counter() - started) * 1000
        return True, (time.perf_counter() - started) * 1000

    def service_health(self, url: str) -> tuple[bool, float]:
        """Probe an absolute internal service health URL."""
        started = time.perf_counter()
        try:
            response = self._session.get(url, timeout=5)
            ready = response.status_code == 200
        except httpx.RequestError:
            ready = False
        return ready, (time.perf_counter() - started) * 1000

    def threads(self) -> list[dict[str, Any]]:
        """List conversation threads."""
        return list(self._request("GET", "/api/v1/threads").json())

    def create_thread(self, title: str) -> dict[str, Any]:
        """Create a conversation thread."""
        return dict(self._request("POST", "/api/v1/threads", json={"title": title}).json())

    def delete_thread(self, thread_id: str) -> None:
        """Delete one conversation thread and its checkpoints."""
        self._request("DELETE", f"/api/v1/threads/{thread_id}")

    def documents(self) -> list[dict[str, Any]]:
        """List indexed documents."""
        return list(self._request("GET", "/api/v1/documents").json())

    def upload(self, file_name: str, data: bytes, mime_type: str, thread_id: str) -> dict[str, Any]:
        """Upload and synchronously index a document."""
        form = {"thread_id": thread_id} if thread_id else {}
        response = self._request(
            "POST",
            "/api/v1/documents/upload",
            files={"file": (Path(file_name).name, data, mime_type)},
            data=form,
            timeout=120,
        )
        return dict(response.json())

    def delete_document(self, document_id: str) -> None:
        """Delete one owned document and its chunks."""
        self._request("DELETE", f"/api/v1/documents/{document_id}")

    def sync_google_drive(self) -> dict[str, Any]:
        """Synchronize the server-configured Google Drive folder."""
        response = self._request(
            "POST",
            "/api/v1/data-sources/google-drive/sync",
            timeout=300,
        )
        return dict(response.json())

    def history(self, thread_id: str) -> list[dict[str, Any]]:
        """Load persisted chat history."""
        return list(self._request("GET", f"/api/v1/chat/{thread_id}/history").json())

    def chat(self, thread_id: str, question: str) -> dict[str, Any]:
        """Submit one grounded agent question."""
        response = self._request("POST", f"/api/v1/chat/{thread_id}", json={"question": question})
        return dict(response.json())

    def stream_chat(self, thread_id: str, question: str) -> Iterator[tuple[str, Any]]:
        """Yield parsed server-sent events for one grounded agent turn."""
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/api/v1/chat/{thread_id}/stream",
                headers=self.headers,
                json={"question": question},
                timeout=90,
            ) as response:
                self._raise_for_status(response)
                event_name = "message"
                data_lines: list[str] = []
                for line in response.iter_lines():
                    if line.startswith("event:"):
                        event_name = line.removeprefix("event:").strip()
                    elif line.startswith("data:"):
                        data_lines.append(line.removeprefix("data:").strip())
                    elif not line and data_lines:
                        yield event_name, json.loads("\n".join(data_lines))
                        event_name = "message"
                        data_lines = []
        except httpx.RequestError as exc:
            raise APIError("The research assistant stream was interrupted.") from exc
