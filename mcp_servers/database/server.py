"""DevPilot Database MCP Server.

Stage 6: three tools against a SQLite demo database (PostgreSQL isn't
available in this environment — user-confirmed deviation, tool signatures
kept engine-agnostic for a later swap). Full rationale logged in
DevPilot_AI_Implementation_Log.html Entry 10.
"""

import sqlite3
from pathlib import Path

from mcp.server import MCPServer

DB_PATH = Path(__file__).resolve().parent / "devpilot_demo.db"
MAX_ROWS = 500
_READ_PREFIXES = ("select", "with")

mcp = MCPServer("DevPilot Database")


def _seed_if_missing() -> None:
    """Create and populate the demo database on first run only."""
    if DB_PATH.exists():
        return

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE stages (
                id INTEGER PRIMARY KEY,
                stage_number INTEGER NOT NULL,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                date_completed TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO stages (stage_number, name, status, date_completed) "
            "VALUES (?, ?, ?, ?)",
            [
                (1, "MCP Fundamentals (Filesystem MCP)", "done", "2026-08-09"),
                (2, "Multiple Tools (Filesystem MCP)", "done", "2026-08-09"),
                (3, "Terminal MCP", "done", "2026-08-10"),
                (4, "GitHub MCP", "done", "2026-08-10"),
                (5, "Browser / Documentation MCP", "done", "2026-08-10"),
                (6, "Database MCP", "in_progress", None),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def _readonly_connection() -> sqlite3.Connection:
    """The real defense: a connection that structurally cannot write."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def _existing_tables(conn: sqlite3.Connection) -> set[str]:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    )
    return {row["name"] for row in cursor.fetchall()}


_seed_if_missing()


@mcp.tool()
def list_tables() -> list[str]:
    """List table names in the DevPilot demo database."""
    conn = _readonly_connection()
    try:
        return sorted(_existing_tables(conn))
    finally:
        conn.close()


@mcp.tool()
def describe_table(table: str) -> list[dict]:
    """Return column name/type/nullable/primary-key info for a table."""
    conn = _readonly_connection()
    try:
        if table not in _existing_tables(conn):
            raise ValueError(f"No such table: '{table}'")

        # `table` is validated against sqlite_master above, so it's safe to
        # interpolate here — PRAGMA doesn't support parameter binding for
        # identifiers the way normal queries support binding values.
        cursor = conn.execute(f"PRAGMA table_info({table})")
        return [
            {
                "name": row["name"],
                "type": row["type"],
                "nullable": not row["notnull"],
                "primary_key": bool(row["pk"]),
            }
            for row in cursor.fetchall()
        ]
    finally:
        conn.close()


@mcp.tool()
def execute_read_query(query: str) -> list[dict]:
    """Run a read-only SQL query against the demo database, capped at 500 rows."""
    stripped = query.strip().lower()
    if not stripped.startswith(_READ_PREFIXES):
        raise ValueError(
            "Only SELECT/WITH queries are allowed. "
            f"Got a statement starting with: '{query.strip()[:20]}'"
        )

    conn = _readonly_connection()
    try:
        cursor = conn.execute(query)
        rows = cursor.fetchmany(MAX_ROWS)
        return [dict(row) for row in rows]
    except sqlite3.OperationalError as exc:
        raise ValueError(f"Query failed: {exc}") from exc
    finally:
        conn.close()


if __name__ == "__main__":
    mcp.run()
