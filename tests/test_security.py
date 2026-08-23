"""Unit tests for mcp_servers/security.py -- pure functions, no MCP
transport or subprocess involved, so these run fast and need nothing
beyond the standard library."""

from mcp_servers.security import is_secret_filename, redact_secrets


def test_env_files_are_flagged():
    assert is_secret_filename(".env")
    assert is_secret_filename(".env.production")


def test_key_and_credential_files_are_flagged():
    assert is_secret_filename("id_rsa")
    assert is_secret_filename("server.pem")
    assert is_secret_filename("credentials.json")
    assert is_secret_filename(".npmrc")


def test_ordinary_source_files_are_not_flagged():
    """Deliberately conservative: a file merely mentioning "secret" in its
    name is not itself a credential file, and should still be readable
    (redact_secrets is the backstop for any real value inside it)."""
    assert not is_secret_filename("app.py")
    assert not is_secret_filename("secrets_test.py")
    assert not is_secret_filename("README.md")


def test_redacts_github_token():
    text = "export GITHUB_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz0123456789"
    result = redact_secrets(text)
    assert "ghp_abcdefghijklmnopqrstuvwxyz0123456789" not in result
    assert "REDACTED" in result


def test_redacts_private_key_block():
    text = "before\n-----BEGIN PRIVATE KEY-----\nMIIBVQ...fakekeydata\n-----END PRIVATE KEY-----\nafter"
    result = redact_secrets(text)
    assert "MIIBVQ" not in result
    assert "before" in result and "after" in result


def test_redacts_credential_in_url():
    text = "connecting to postgres://admin:hunter2@db.internal:5432/app"
    result = redact_secrets(text)
    assert "hunter2" not in result


def test_redacts_generic_assignment_but_keeps_key_name():
    text = 'API_KEY = "sk-live-abcdefghijklmnop"'
    result = redact_secrets(text)
    assert "abcdefghijklmnop" not in result
    assert "API_KEY" in result  # key name preserved as context


def test_leaves_ordinary_text_untouched():
    text = "def add(a, b):\n    return a + b\n"
    assert redact_secrets(text) == text
