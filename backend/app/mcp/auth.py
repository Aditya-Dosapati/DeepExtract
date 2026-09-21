"""MCP bearer-token verification backed by application JWTs."""

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken

from backend.app.auth.security import InvalidTokenError, decode_access_token
from backend.app.core.config import Settings


class ApplicationTokenVerifier:
    """Adapt application access JWTs to the MCP SDK token verifier protocol."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def verify_token(self, token: str) -> AccessToken | None:
        """Return MCP access metadata only for a valid unexpired application JWT."""
        try:
            claims = decode_access_token(token, self.settings)
        except InvalidTokenError:
            return None
        subject = str(claims.user_id)
        return AccessToken(
            token=token,
            client_id=subject,
            subject=subject,
            scopes=["user"],
            expires_at=int(claims.expires_at.timestamp()),
        )


def authenticated_subject() -> str:
    """Return the verified MCP subject from request-local authentication context."""
    access_token = get_access_token()
    if access_token is None or access_token.subject is None:
        raise PermissionError("Authentication is required")
    return access_token.subject
