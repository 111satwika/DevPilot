"""Shared secret-hygiene helpers for MCP servers, used wherever tool
output could carry file/command content back into the model's context.

Nothing in the codebase filtered secrets before this. The design doc's
§19 (Security) and Production Considerations §1 have required it since
the project's own plan was written -- "never let MCP servers read .env
files or credentials directly into LLM context" and "redact known secret
patterns ... from tool output before it reaches the model" -- but
`read_file` had zero filtering, and command/git/docker output was never
scrubbed either. Gap-fix; full rationale in
DevPilot_AI_Implementation_Log.html Entry 41.

Two layers, matching the "mechanism over pattern-matching" preference
already established elsewhere in this project (Terminal's `shell=False`,
Database's read-only connection): block known-credential filenames
outright (structural -- read_file refuses them entirely, the same way
Filesystem refuses a path outside the sandbox), and redact secret-shaped
substrings from whatever content *does* get returned (a pattern-matched
backstop for secrets embedded in files/output that aren't blocked by
name -- e.g. a hardcoded key inside an ordinary .py config file, or a
token embedded in a `pip install git+https://user:token@...` command's
stdout). The backstop is explicitly a backstop, not a guarantee -- same
honesty as the read-only-connection precedent in mcp_servers/database.

Lives in mcp_servers/ (not backend/ or llm/) so every layer -- MCP server
tool functions and llm/agent.py's audit logging alike -- can import it
without creating an import cycle. Same reasoning as mcp_servers/
workspace.py living here.
"""

import fnmatch
import re

_SECRET_FILENAME_GLOBS = (
    ".env", ".env.*",
    "*.pem", "*.key", "*.pfx", "*.p12", "*.keystore",
    "id_rsa", "id_rsa.pub", "id_ed25519", "id_ed25519.pub",
    "id_dsa", "id_dsa.pub", "id_ecdsa", "id_ecdsa.pub",
    "credentials.json", "*.credentials",
    ".npmrc", ".netrc", "*.asc",
)


def is_secret_filename(name: str) -> bool:
    """True if `name` (a basename, not a full path) matches a known
    credential-file pattern. Deliberately does NOT match vague globs like
    `*secret*`/`*password*` -- those false-positive on ordinary source
    files (e.g. `secrets_test.py`) that merely discuss secrets rather than
    contain one; a real embedded value in such a file is still caught by
    redact_secrets() below as the backstop layer."""
    lowered = name.lower()
    return any(fnmatch.fnmatch(lowered, pattern) for pattern in _SECRET_FILENAME_GLOBS)


# Each entry: (label used in the redaction placeholder, compiled pattern).
# Patterns with exactly 2 capturing groups are treated as (key, value) --
# the key name is preserved in the output (useful context: "a token was
# redacted here") while only the value is replaced.
_SECRET_PATTERNS: list[tuple[str, "re.Pattern[str]"]] = [
    ("private key", re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----"
        r"[\s\S]*?"
        r"-----END (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----"
    )),
    ("aws access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("credential in url", re.compile(r"\b\w+://[^\s/@'\"]+:[^\s/@'\"]+@[^\s'\"]+")),
    # Boundaries are (?<![A-Za-z0-9]) / (?![A-Za-z0-9]) rather than \b, so a
    # SCREAMING_SNAKE_CASE name like GITHUB_TOKEN still matches (the `_`
    # doesn't count as a word boundary for \b since it's a word character)
    # while a same-family substring like "tokenizer" still correctly does
    # NOT match, since the very next char ('i') IS alnum. This is still a
    # naive regex over source text, not real secret-entropy detection --
    # a variable like `access_token = get_token()` will also be redacted
    # even though its value is a function call, not a literal secret. That
    # false-positive is an accepted tradeoff for a backstop: over-redacting
    # occasionally is a smaller cost than a real secret leaking through.
    ("assigned secret", re.compile(
        r"(?i)(?<![A-Za-z0-9])(api[_-]?key|secret[_-]?key|access[_-]?key|secret|token|"
        r"password|passwd|pwd)(?![A-Za-z0-9])\s*[:=]\s*['\"]?([^\s'\"]{6,})['\"]?"
    )),
]


def redact_secrets(text: str) -> str:
    """Replace anything shaped like a real secret with a typed
    placeholder. A pattern-matched backstop, not a guarantee -- see the
    module docstring."""
    if not text:
        return text
    redacted = text
    for label, pattern in _SECRET_PATTERNS:
        if pattern.groups >= 2:
            redacted = pattern.sub(lambda m, label=label: f"{m.group(1)}=[REDACTED:{label}]", redacted)
        else:
            redacted = pattern.sub(f"[REDACTED:{label}]", redacted)
    return redacted
