"""Password and access-token security tests."""

from uuid import uuid4

import pytest
from pydantic import SecretStr

from backend.app.auth.security import (
    InvalidTokenError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from backend.app.core.config import Settings


def test_password_hash_is_salted_and_verifiable() -> None:
    password = "Correct-Horse-Battery-7"
    first_hash = hash_password(password)
    second_hash = hash_password(password)

    assert first_hash != password
    assert first_hash != second_hash
    assert verify_password(password, first_hash)
    assert not verify_password("wrong-password", first_hash)


def test_access_token_round_trip_and_tamper_rejection() -> None:
    settings = Settings(
        app_env="test", jwt_secret=SecretStr("unit-test-secret-with-at-least-32-characters")
    )
    user_id = uuid4()
    token, expires_in = create_access_token(user_id, settings)

    claims = decode_access_token(token, settings)

    assert claims.user_id == user_id
    assert expires_in == 3600
    with pytest.raises(InvalidTokenError):
        decode_access_token(f"{token}tampered", settings)
