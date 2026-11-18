"""API de stockage : c'est la seule porte d'entrée vers la base."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from assistant_vocal.config import ROOT_DIR, AppConfig
from assistant_vocal.storage.database import connect


@dataclass
class Conversation:
    id: int
    title: str
    created_at: str
    updated_at: str
    archived: bool


@dataclass
class Message:
    id: int
    conversation_id: int
    role: str
    content: str
    emotion: str | None
    emotion_score: float | None
    created_at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _conv(row: sqlite3.Row) -> Conversation:
    return Conversation(
        id=row["id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived=bool(row["archived"]),
    )


def _msg(row: sqlite3.Row) -> Message:
    return Message(**{k: row[k] for k in row.keys()})


class ConversationStore:
    def __init__(self, db_path: str | Path = ":memory:"):
        self.conn = connect(db_path)

    def close(self) -> None:
        self.conn.close()
    def _next_seq(self) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(activity_seq), 0) + 1 FROM conversations"
        ).fetchone()
        return row[0]

    # --- conversations ---------------------------------------------------

    def create_conversation(self, title: str | None = None) -> Conversation:
        now = _now()
        title = title or f"Conversation du {datetime.now():%d/%m/%Y %H:%M}"
        cur = self.conn.execute(
            "INSERT INTO conversations (title, created_at, updated_at, activity_seq)"
            "VALUES (?, ?, ?, ?)",
            (title, now, now, self._next_seq()),
        )
        self.conn.commit()
        return self.get_conversation(cur.lastrowid)

    def get_conversation(self, conversation_id: int) -> Conversation | None:
        row = self.conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        return _conv(row) if row else None

    def get_last_conversation(self) -> Conversation | None:
        row = self.conn.execute(
            "SELECT * FROM conversations WHERE archived = 0 "
            "ORDER BY activity_seq DESC LIMIT 1"
        ).fetchone()
        return _conv(row) if row else None

    def list_conversations(self, include_archived: bool = False) -> list[Conversation]:
        query = "SELECT * FROM conversations"
        if not include_archived:
            query += " WHERE archived = 0"
        query += " ORDER BY activity_seq DESC"
        return [_conv(r) for r in self.conn.execute(query)]

    def archive_conversation(self, conversation_id: int, archived: bool = True) -> None:
        self.conn.execute(
            "UPDATE conversations SET archived = ? WHERE id = ?",
            (int(archived), conversation_id),
        )
        self.conn.commit()

    def delete_conversation(self, conversation_id: int) -> bool:
        """Supprime la conversation ET ses messages. Retourne True si elle existait."""
        cur = self.conn.execute(
            "DELETE FROM conversations WHERE id = ?", (conversation_id,)
        )
        self.conn.commit()
        return cur.rowcount > 0

    # --- messages --------------------------------------------------------

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        emotion: str | None = None,
        emotion_score: float | None = None,
    ) -> Message:
        now = _now()
        cur = self.conn.execute(
            "INSERT INTO messages (conversation_id, role, content, emotion, "
            "emotion_score, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (conversation_id, role, content, emotion, emotion_score, now),
        )
        self.conn.execute(
            "UPDATE conversations SET updated_at = ?, activity_seq = ? WHERE id = ?",
            (now, self._next_seq(), conversation_id),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT * FROM messages WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
        return _msg(row)

    def get_messages(self, conversation_id: int, limit: int | None = None) -> list[Message]:
        """Messages dans l'ordre chronologique. Avec limit : les N derniers."""
        if limit:
            rows = self.conn.execute(
                "SELECT * FROM (SELECT * FROM messages WHERE conversation_id = ? "
                "ORDER BY id DESC LIMIT ?) ORDER BY id",
                (conversation_id, limit),
            )
        else:
            rows = self.conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            )
        return [_msg(r) for r in rows]

    # --- préférences -----------------------------------------------------

    def set_setting(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self.conn.commit()

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default


def open_store(cfg: AppConfig) -> ConversationStore:
    """Ouvre la base indiquée dans la config (chemin relatif à la racine du projet)."""
    path = Path(cfg.storage.db_path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return ConversationStore(path)