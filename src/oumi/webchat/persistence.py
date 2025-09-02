import os
import sqlite3
import threading
import time
import uuid
import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence, List, Dict, Tuple, Any


class WebchatDB:
    """Lightweight SQLite persistence for sessions, conversations, branches and messages.

    Phase 1a: Dual-write only. The in-memory structures remain the source of truth
    for active sessions; we persist writes so that future phases can hydrate from DB.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        default_path = os.path.expanduser("~/.oumi/webchat.sqlite")
        self.db_path = db_path or os.environ.get("OUMI_DB_PATH", default_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # Initialize schema
        with self._connect() as conn:
            self._init_schema(conn)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        cur = conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at REAL,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS branches (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                name TEXT,
                parent_branch_id TEXT,
                created_at REAL,
                metadata TEXT,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL,
                metadata TEXT,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            );

            CREATE TABLE IF NOT EXISTS branch_messages (
                branch_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                PRIMARY KEY(branch_id, seq),
                FOREIGN KEY(branch_id) REFERENCES branches(id),
                FOREIGN KEY(message_id) REFERENCES messages(id)
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                current_conversation_id TEXT,
                current_branch_id TEXT,
                created_at REAL,
                metadata TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conv_created ON messages(conversation_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_branch_msgs_branch ON branch_messages(branch_id, seq);
            CREATE INDEX IF NOT EXISTS idx_branch_msgs_message ON branch_messages(message_id);
            CREATE INDEX IF NOT EXISTS idx_branches_parent ON branches(parent_branch_id);
            CREATE INDEX IF NOT EXISTS idx_branches_conversation ON branches(conversation_id);
            """
        )
        conn.commit()

    # Helpers
    def _now(self) -> float:
        return time.time()

    def _next_seq(self, conn: sqlite3.Connection, branch_id: str) -> int:
        cur = conn.cursor()
        cur.execute("SELECT COALESCE(MAX(seq), -1) + 1 FROM branch_messages WHERE branch_id=?", (branch_id,))
        (next_seq,) = cur.fetchone()
        return int(next_seq or 0)

    # Public API
    def ensure_session(self, session_id: str) -> None:
        with self._lock, self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM sessions WHERE session_id=?", (session_id,))
            if cur.fetchone() is None:
                cur.execute(
                    "INSERT INTO sessions(session_id, created_at) VALUES (?, ?)",
                    (session_id, self._now()),
                )
                conn.commit()

    def ensure_conversation(self, session_id: str) -> str:
        """Ensure a conversation exists and is linked to the session; returns conversation_id."""
        with self._lock, self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT current_conversation_id FROM sessions WHERE session_id=?",
                (session_id,),
            )
            row = cur.fetchone()
            if row and row[0]:
                return row[0]
            # Create new conversation
            conv_id = f"conv_{uuid.uuid4().hex}"
            cur.execute(
                "INSERT INTO conversations(id, created_at) VALUES (?, ?)",
                (conv_id, self._now()),
            )
            cur.execute(
                "INSERT OR IGNORE INTO sessions(session_id, created_at) VALUES (?, ?)",
                (session_id, self._now()),
            )
            cur.execute(
                "UPDATE sessions SET current_conversation_id=? WHERE session_id=?",
                (conv_id, session_id),
            )
            conn.commit()
            return conv_id

    def ensure_branch(self, conversation_id: str, branch_id: str, name: Optional[str] = None, parent_branch_id: Optional[str] = None) -> None:
        with self._lock, self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM branches WHERE id=?", (branch_id,))
            if cur.fetchone() is None:
                cur.execute(
                    "INSERT INTO branches(id, conversation_id, name, parent_branch_id, created_at) VALUES (?, ?, ?, ?, ?)",
                    (branch_id, conversation_id, name, parent_branch_id, self._now()),
                )
                conn.commit()

    def append_message_to_branch(self, conversation_id: str, branch_id: str, role: str, content: str, created_at: Optional[float] = None) -> str:
        """Insert a new message in conversation and link it to the branch with next seq."""
        created = created_at or self._now()
        msg_id = f"msg_{uuid.uuid4().hex}"
        with self._lock, self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO messages(id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                (msg_id, conversation_id, role, content, created),
            )
            seq = self._next_seq(conn, branch_id)
            cur.execute(
                "INSERT INTO branch_messages(branch_id, message_id, seq) VALUES (?, ?, ?)",
                (branch_id, msg_id, seq),
            )
            conn.commit()
        return msg_id

    def bulk_add_branch_history(self, conversation_id: str, branch_id: str, messages: Sequence[dict]) -> None:
        """Populate a branch with an existing list of message dicts.

        Phase 1a note: We insert new message rows; dedup is not attempted.
        """
        with self._lock, self._connect() as conn:
            cur = conn.cursor()
            next_seq = self._next_seq(conn, branch_id)
            for msg in messages:
                msg_id = f"msg_{uuid.uuid4().hex}"
                role = msg.get("role", "user")
                content = str(msg.get("content", ""))
                created = float(msg.get("timestamp", time.time()))
                cur.execute(
                    "INSERT INTO messages(id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                    (msg_id, conversation_id, role, content, created),
                )
                cur.execute(
                    "INSERT INTO branch_messages(branch_id, message_id, seq) VALUES (?, ?, ?)",
                    (branch_id, msg_id, next_seq),
                )
                next_seq += 1
            conn.commit()

    def set_session_current_branch(self, session_id: str, conversation_id: str, branch_id: str) -> None:
        with self._lock, self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT OR IGNORE INTO sessions(session_id, created_at) VALUES (?, ?)",
                (session_id, self._now()),
            )
            cur.execute(
                "UPDATE sessions SET current_conversation_id=?, current_branch_id=? WHERE session_id=?",
                (conversation_id, branch_id, session_id),
            )
            conn.commit()

    def get_branch_messages(self, branch_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a branch in sequence order."""
        with self._lock, self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT m.id, m.role, m.content, m.created_at, m.metadata, bm.seq
                FROM messages m
                JOIN branch_messages bm ON m.id = bm.message_id  
                WHERE bm.branch_id = ?
                ORDER BY bm.seq ASC
                """,
                (branch_id,)
            )
            results = []
            for row in cur.fetchall():
                msg_id, role, content, created_at, metadata, seq = row
                results.append({
                    "id": msg_id,
                    "role": role,
                    "content": content,
                    "timestamp": created_at,
                    "metadata": json.loads(metadata) if metadata else {},
                    "seq": seq
                })
            return results

    def get_session_branches(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all branches for a session's current conversation."""
        with self._lock, self._connect() as conn:
            cur = conn.cursor()
            # First get the current conversation for this session
            cur.execute(
                "SELECT current_conversation_id FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            row = cur.fetchone()
            if not row or not row[0]:
                return []
            
            conversation_id = row[0]
            
            # Get all branches for this conversation
            cur.execute(
                """
                SELECT b.id, b.name, b.parent_branch_id, b.created_at, b.metadata,
                       COUNT(bm.message_id) as message_count
                FROM branches b
                LEFT JOIN branch_messages bm ON b.id = bm.branch_id
                WHERE b.conversation_id = ?
                GROUP BY b.id, b.name, b.parent_branch_id, b.created_at, b.metadata
                ORDER BY b.created_at ASC
                """,
                (conversation_id,)
            )
            
            results = []
            for row in cur.fetchall():
                branch_id, name, parent_id, created_at, metadata, msg_count = row
                results.append({
                    "id": branch_id,
                    "name": name or branch_id,
                    "parent_branch_id": parent_id,
                    "created_at": created_at,
                    "message_count": msg_count,
                    "metadata": json.loads(metadata) if metadata else {}
                })
            return results

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information including current conversation and branch."""
        with self._lock, self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT current_conversation_id, current_branch_id, created_at FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            row = cur.fetchone()
            if not row:
                return None
            
            conv_id, branch_id, created_at = row
            return {
                "session_id": session_id,
                "current_conversation_id": conv_id,
                "current_branch_id": branch_id,
                "created_at": created_at
            }

    def hydrate_session(self, session_id: str, branch_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load complete session state from database for hydrating in-memory structures.
        
        Returns session info + current branch messages, or None if session not found.
        """
        session_info = self.get_session_info(session_id)
        if not session_info:
            return None
            
        # Use provided branch_id or fall back to session's current branch
        target_branch_id = branch_id or session_info["current_branch_id"]
        if not target_branch_id:
            # Session exists but has no current branch - return empty
            return {
                "session_info": session_info,
                "branches": [],
                "current_messages": [],
                "current_branch_id": None
            }
        
        # Get all branches for the session
        branches = self.get_session_branches(session_id)
        
        # Get messages for the target branch
        messages = self.get_branch_messages(target_branch_id)
        
        return {
            "session_info": session_info,
            "branches": branches,
            "current_messages": messages,
            "current_branch_id": target_branch_id
        }

    def get_conversation_title(self, conversation_id: str) -> Optional[str]:
        """Get the title of a conversation."""
        with self._lock, self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT title FROM conversations WHERE id = ?", (conversation_id,))
            row = cur.fetchone()
            return row[0] if row else None

    def update_conversation_title(self, conversation_id: str, title: str) -> None:
        """Update the title of a conversation."""
        with self._lock, self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (title, conversation_id)
            )
            conn.commit()

