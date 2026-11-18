"""Connexion SQLite et migrations de schéma."""
from __future__ import annotations

import sqlite3
from pathlib import Path

# Chaque élément est une migration. La version du schéma est stockée
# dans PRAGMA user_version : pour modifier la base plus tard, on AJOUTE
# une entrée à la fin de cette liste, sans toucher aux précédentes.
MIGRATIONS = [
    """
    CREATE TABLE conversations (
        id          INTEGER PRIMARY KEY,
        title       TEXT,
        created_at  TEXT NOT NULL,
        updated_at  TEXT NOT NULL,
        archived    INTEGER NOT NULL DEFAULT 0
    );

    CREATE TABLE messages (
        id               INTEGER PRIMARY KEY,
        conversation_id  INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        role             TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
        content          TEXT NOT NULL,
        emotion          TEXT,
        emotion_score    REAL,
        created_at       TEXT NOT NULL
    );

    CREATE INDEX idx_messages_conv ON messages(conversation_id, id);

    CREATE TABLE settings (
        key    TEXT PRIMARY KEY,
        value  TEXT NOT NULL
    );
    """,
    """
    ALTER TABLE conversations ADD COLUMN activity_seq INTEGER NOT NULL DEFAULT 0;
    UPDATE conversations SET activity_seq = id;
    """,
]


def migrate(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for number, script in enumerate(MIGRATIONS[version:], start=version + 1):
        conn.executescript(script)
        conn.execute(f"PRAGMA user_version = {number}")
    conn.commit()


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Ouvre la base (créée si besoin) et applique les migrations."""
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")  # indispensable pour ON DELETE CASCADE
    migrate(conn)
    return conn