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

    TARGET_SCHEMA_VERSION = 2  # Increment when adding migrations

    def __init__(self, db_path: Optional[str] = None) -> None:
        default_path = os.path.expanduser("~/.oumi/webchat.sqlite")
        self.db_path = db_path or os.environ.get("OUMI_DB_PATH", default_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # Initialize schema
        with self._connect() as conn:
            self._init_schema(conn)
            self._init_schema_version_and_migrate(conn)

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
                content_hash TEXT,
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
            CREATE INDEX IF NOT EXISTS idx_messages_content_hash ON messages(content_hash);
            CREATE INDEX IF NOT EXISTS idx_branch_msgs_branch ON branch_messages(branch_id, seq);
            CREATE INDEX IF NOT EXISTS idx_branch_msgs_message ON branch_messages(message_id);
            CREATE INDEX IF NOT EXISTS idx_branches_parent ON branches(parent_branch_id);
            CREATE INDEX IF NOT EXISTS idx_branches_conversation ON branches(conversation_id);
            
            -- FTS virtual table for search (optional, created on demand)
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(content, content=messages, content_rowid=rowid);
            """
        )
        conn.commit()

    # --- Schema versioning and migrations ---
    def _table_exists(self, conn: sqlite3.Connection, table: str) -> bool:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,))
        return cur.fetchone() is not None

    def _trigger_exists(self, conn: sqlite3.Connection, name: str) -> bool:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM sqlite_master WHERE type='trigger' AND name=?", (name,))
        return cur.fetchone() is not None

    def _get_schema_version(self, conn: sqlite3.Connection) -> int:
        if not self._table_exists(conn, "schema_version"):
            return 0
        cur = conn.cursor()
        cur.execute("SELECT version FROM schema_version WHERE id=1")
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def _set_schema_version(self, conn: sqlite3.Connection, version: int) -> None:
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (id INTEGER PRIMARY KEY CHECK (id = 1), version INTEGER NOT NULL)"
        )
        cur.execute("INSERT INTO schema_version(id, version) VALUES (1, ?) ON CONFLICT(id) DO UPDATE SET version=excluded.version", (version,))
        conn.commit()

    def _ensure_fts_triggers(self, conn: sqlite3.Connection) -> None:
        """Create FTS5 external content triggers for messages table if missing."""
        # Insert trigger
        if not self._trigger_exists(conn, "messages_ai_fts"):
            conn.execute(
                """
                CREATE TRIGGER messages_ai_fts AFTER INSERT ON messages BEGIN
                  INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
                END;
                """
            )
        # Delete trigger
        if not self._trigger_exists(conn, "messages_ad_fts"):
            conn.execute(
                """
                CREATE TRIGGER messages_ad_fts AFTER DELETE ON messages BEGIN
                  INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.rowid, old.content);
                END;
                """
            )
        # Update trigger
        if not self._trigger_exists(conn, "messages_au_fts"):
            conn.execute(
                """
                CREATE TRIGGER messages_au_fts AFTER UPDATE ON messages BEGIN
                  INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.rowid, old.content);
                  INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
                END;
                """
            )
        conn.commit()

    def _rebuild_fts(self, conn: sqlite3.Connection) -> None:
        """Rebuild the FTS index from the content table."""
        try:
            conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
            conn.commit()
        except sqlite3.OperationalError:
            # Fallback: populate by full copy if rebuild unsupported
            cur = conn.cursor()
            cur.execute("DELETE FROM messages_fts")
            cur.execute("INSERT INTO messages_fts(rowid, content) SELECT rowid, content FROM messages")
            conn.commit()

    def _init_schema_version_and_migrate(self, conn: sqlite3.Connection) -> None:
        """Ensure schema_version exists and perform any pending migrations."""
        current = self._get_schema_version(conn)
        if current == 0:
            # Fresh install or pre-versioned DB: set baseline to 1 after ensuring tables exist
            self._set_schema_version(conn, 1)
            current = 1

        # Migration: v1 -> v2 (add FTS triggers and rebuild index)
        if current < 2:
            # Ensure FTS table exists (older DBs may not have it)
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(content, content=messages, content_rowid=rowid)"
            )
            self._ensure_fts_triggers(conn)
            self._rebuild_fts(conn)
            self._set_schema_version(conn, 2)

    # Helpers
    def _now(self) -> float:
        return time.time()

    def _next_seq(self, conn: sqlite3.Connection, branch_id: str) -> int:
        cur = conn.cursor()
        cur.execute("SELECT COALESCE(MAX(seq), -1) + 1 FROM branch_messages WHERE branch_id=?", (branch_id,))
        (next_seq,) = cur.fetchone()
        return int(next_seq or 0)

    def _hash_content(self, role: str, content: str, metadata: Optional[str] = None) -> str:
        """Generate a content hash for deduplication."""
        content_str = f"{role}:{content}"
        if metadata:
            content_str += f":{metadata}"
        return hashlib.sha256(content_str.encode('utf-8')).hexdigest()

    def _find_existing_message(self, conn: sqlite3.Connection, conversation_id: str, content_hash: str) -> Optional[str]:
        """Find existing message with same content hash in this conversation."""
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM messages WHERE conversation_id = ? AND content_hash = ? LIMIT 1",
            (conversation_id, content_hash)
        )
        row = cur.fetchone()
        return row[0] if row else None

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

    def append_message_to_branch(
        self,
        conversation_id: str,
        branch_id: str,
        role: str,
        content: str,
        created_at: Optional[float] = None,
        metadata: Optional[str] = None,
        *,
        force_new: bool = False,
    ) -> str:
        """Insert a new message in conversation and link it to the branch with next seq.
        
        Uses content deduplication - if identical message exists, reuses it.
        """
        created = created_at or self._now()
        content_hash = self._hash_content(role, content, metadata)
        
        with self._lock, self._connect() as conn:
            cur = conn.cursor()
            
            msg_id = None
            # Try to find existing message with same content (unless forcing new row)
            if not force_new:
                existing_msg_id = self._find_existing_message(conn, conversation_id, content_hash)
                if existing_msg_id:
                    msg_id = existing_msg_id
            
            if msg_id is None:
                # Create new message
                msg_id = f"msg_{uuid.uuid4().hex}"
                cur.execute(
                    "INSERT INTO messages(id, conversation_id, role, content, content_hash, created_at, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (msg_id, conversation_id, role, content, content_hash, created, metadata),
                )
            
            # Add to branch sequence (always create new branch_message entry)
            seq = self._next_seq(conn, branch_id)
            cur.execute(
                "INSERT INTO branch_messages(branch_id, message_id, seq) VALUES (?, ?, ?)",
                (branch_id, msg_id, seq),
            )
            conn.commit()
        return msg_id

    def bulk_add_branch_history(self, conversation_id: str, branch_id: str, messages: Sequence[dict]) -> None:
        """Populate a branch with an existing list of message dicts.

        Uses content deduplication - reuses existing messages when possible.
        """
        with self._lock, self._connect() as conn:
            cur = conn.cursor()
            next_seq = self._next_seq(conn, branch_id)
            for msg in messages:
                role = msg.get("role", "user")
                content = str(msg.get("content", ""))
                created = float(msg.get("timestamp", time.time()))
                metadata = json.dumps(msg.get("metadata", {})) if msg.get("metadata") else None
                
                # Generate content hash for deduplication
                content_hash = self._hash_content(role, content, metadata)
                
                # Try to find existing message
                existing_msg_id = self._find_existing_message(conn, conversation_id, content_hash)
                
                if existing_msg_id:
                    # Reuse existing message
                    msg_id = existing_msg_id
                else:
                    # Create new message
                    msg_id = f"msg_{uuid.uuid4().hex}"
                    cur.execute(
                        "INSERT INTO messages(id, conversation_id, role, content, content_hash, created_at, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (msg_id, conversation_id, role, content, content_hash, created, metadata),
                    )
                
                # Add to branch sequence
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
        
        # Get all branches for the session
        branches = self.get_session_branches(session_id)
        
        if not target_branch_id and branches:
            # If no current branch set but branches exist, use the first one
            target_branch_id = branches[0]["id"]
            
        if not target_branch_id:
            # Session exists but has no branches - return empty
            return {
                "session_info": session_info,
                "branches": [],
                "current_messages": [],
                "current_branch_id": None
            }
        
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

    def search_messages(self, query: str, conversation_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Search messages using FTS. Optionally filter by conversation."""
        with self._lock, self._connect() as conn:
            cur = conn.cursor()
            
            if conversation_id:
                # Search within specific conversation
                cur.execute(
                    """
                    SELECT m.id, m.conversation_id, m.role, m.content, m.created_at
                    FROM messages_fts fts
                    JOIN messages m ON fts.rowid = m.rowid
                    WHERE fts.content MATCH ? AND m.conversation_id = ?
                    ORDER BY fts.rank
                    LIMIT ?
                    """,
                    (query, conversation_id, limit)
                )
            else:
                # Search all messages
                cur.execute(
                    """
                    SELECT m.id, m.conversation_id, m.role, m.content, m.created_at
                    FROM messages_fts fts
                    JOIN messages m ON fts.rowid = m.rowid
                    WHERE fts.content MATCH ?
                    ORDER BY fts.rank
                    LIMIT ?
                    """,
                    (query, limit)
                )
            
            results = []
            for row in cur.fetchall():
                msg_id, conv_id, role, content, created_at = row
                results.append({
                    "id": msg_id,
                    "conversation_id": conv_id,
                    "role": role,
                    "content": content,
                    "timestamp": created_at
                })
            return results

    def get_conversation_stats(self, conversation_id: str) -> Dict[str, Any]:
        """Get statistics for a conversation."""
        with self._lock, self._connect() as conn:
            cur = conn.cursor()
            
            # Count total messages
            cur.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conversation_id,))
            total_messages = cur.fetchone()[0]
            
            # Count branches
            cur.execute("SELECT COUNT(*) FROM branches WHERE conversation_id = ?", (conversation_id,))
            total_branches = cur.fetchone()[0]
            
            # Get creation time
            cur.execute("SELECT created_at FROM conversations WHERE id = ?", (conversation_id,))
            created_at = cur.fetchone()
            created_at = created_at[0] if created_at else None
            
            return {
                "conversation_id": conversation_id,
                "total_messages": total_messages,
                "total_branches": total_branches,
                "created_at": created_at
            }

    def list_conversations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all conversations with metadata."""
        with self._lock, self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT c.id, c.title, c.created_at, c.metadata,
                       COUNT(DISTINCT m.id) as message_count,
                       COUNT(DISTINCT b.id) as branch_count
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                LEFT JOIN branches b ON c.id = b.conversation_id
                GROUP BY c.id, c.title, c.created_at, c.metadata
                ORDER BY c.created_at DESC
                LIMIT ?
                """,
                (limit,)
            )
            
            results = []
            for row in cur.fetchall():
                conv_id, title, created_at, metadata, msg_count, branch_count = row
                results.append({
                    "id": conv_id,
                    "title": title,
                    "created_at": created_at,
                    "metadata": json.loads(metadata) if metadata else {},
                    "message_count": msg_count or 0,
                    "branch_count": branch_count or 0
                })
            return results
            
    def branch_has_children(self, conversation_id: str, branch_id: str) -> bool:
        """Check if a branch has any child branches or is referenced in graph edges.
        
        Args:
            conversation_id: The conversation ID
            branch_id: The branch ID to check
            
        Returns:
            True if branch has children, False otherwise
        """
        with self._lock, self._connect() as conn:
            cur = conn.cursor()
            
            # First check for branches that have this branch as parent
            cur.execute(
                "SELECT 1 FROM branches WHERE conversation_id = ? AND parent_branch_id = ? LIMIT 1",
                (conversation_id, branch_id)
            )
            if cur.fetchone():
                return True
                
            # Then check for graph edges if the table exists
            try:
                # Check if table exists first
                cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='graph_edges'"
                )
                if cur.fetchone():
                    # Check if branch is source in any graph edge
                    cur.execute(
                        "SELECT 1 FROM graph_edges WHERE conversation_id = ? AND src_branch_id = ? LIMIT 1",
                        (conversation_id, branch_id)
                    )
                    if cur.fetchone():
                        return True
            except Exception as e:
                # If graph edges table doesn't exist or other error, log but continue
                logger.debug(f"Could not check graph edges for branch {branch_id}: {e}")
                
            return False
    
    def branch_is_current(self, session_id: str, branch_id: str) -> bool:
        """Check if a branch is the current branch for a session.
        
        Args:
            session_id: The session ID
            branch_id: The branch ID to check
            
        Returns:
            True if branch is current, False otherwise
        """
        with self._lock, self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT current_branch_id FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            row = cur.fetchone()
            if not row:
                return False
                
            return row[0] == branch_id
    
    def delete_branch(self, conversation_id: str, branch_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Delete a branch and all its messages from the database.
        
        Performs safety checks before deletion:
        - Branch must exist
        - Branch must not be 'main'
        - Branch must not be current for any session (if session_id is provided)
        - Branch must not have any children or graph references
        
        Args:
            conversation_id: The conversation ID
            branch_id: The branch ID to delete
            session_id: Optional session ID to check if branch is current
            
        Returns:
            Dict with keys:
            - success: Whether deletion was successful
            - reason: Reason for failure if not successful
            - deleted_message_count: Number of messages deleted (if successful)
        """
        if branch_id == "main":
            return {"success": False, "reason": "Cannot delete the main branch", "deleted_message_count": 0}
        
        # Use a transaction for atomicity
        with self._lock, self._connect() as conn:
            try:
                conn.execute("BEGIN TRANSACTION")
                cur = conn.cursor()
                
                # Check if branch exists
                cur.execute(
                    "SELECT 1 FROM branches WHERE conversation_id = ? AND id = ? LIMIT 1",
                    (conversation_id, branch_id)
                )
                if not cur.fetchone():
                    conn.rollback()
                    return {"success": False, "reason": f"Branch '{branch_id}' not found", "deleted_message_count": 0}
                
                # Check if branch is current (if session_id provided)
                if session_id:
                    cur.execute(
                        "SELECT current_branch_id FROM sessions WHERE session_id = ?",
                        (session_id,)
                    )
                    row = cur.fetchone()
                    if row and row[0] == branch_id:
                        conn.rollback()
                        return {"success": False, "reason": "Cannot delete current branch (switch first)", "deleted_message_count": 0}
                
                # Check if branch has children
                cur.execute(
                    "SELECT COUNT(*) FROM branches WHERE conversation_id = ? AND parent_branch_id = ?",
                    (conversation_id, branch_id)
                )
                child_count = cur.fetchone()[0]
                if child_count > 0:
                    conn.rollback()
                    return {"success": False, "reason": f"Branch has {child_count} child branches; delete descendants first", "deleted_message_count": 0}
                
                # Check for graph edges if table exists
                edge_count = 0
                try:
                    # Check if table exists
                    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='graph_edges'")
                    if cur.fetchone():
                        # Check for source edges
                        cur.execute(
                            "SELECT COUNT(*) FROM graph_edges WHERE conversation_id = ? AND src_branch_id = ?",
                            (conversation_id, branch_id)
                        )
                        edge_count = cur.fetchone()[0]
                        if edge_count > 0:
                            conn.rollback()
                            return {"success": False, "reason": f"Branch has {edge_count} graph edges; delete those first", "deleted_message_count": 0}
                except Exception as e:
                    # Log but continue if graph_edges table doesn't exist
                    logger.debug(f"Could not check graph edges for branch {branch_id}: {e}")
                
                # Count messages for reporting
                cur.execute(
                    "SELECT COUNT(*) FROM branch_messages WHERE branch_id = ?",
                    (branch_id,)
                )
                message_count = cur.fetchone()[0]
                
                # Delete branch messages
                cur.execute(
                    "DELETE FROM branch_messages WHERE branch_id = ?",
                    (branch_id,)
                )
                
                # Delete branch from branches table
                cur.execute(
                    "DELETE FROM branches WHERE conversation_id = ? AND id = ?",
                    (conversation_id, branch_id)
                )
                
                # Try to delete from graph_edges if table exists
                try:
                    # Check if table exists
                    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='graph_edges'")
                    if cur.fetchone():
                        # Delete any graph edges involving this branch
                        cur.execute(
                            "DELETE FROM graph_edges WHERE conversation_id = ? AND (src_branch_id = ? OR dst_branch_id = ?)",
                            (conversation_id, branch_id, branch_id)
                        )
                except Exception as e:
                    # Log but continue if graph_edges table doesn't exist
                    logger.debug(f"Could not delete graph edges for branch {branch_id}: {e}")
                
                # Clean up orphaned messages (optional - could be done in a separate maintenance task)
                # This would remove messages no longer referenced by any branch
                
                conn.commit()
                
                return {
                    "success": True, 
                    "reason": "", 
                    "deleted_message_count": message_count,
                }
            
            except Exception as e:
                # Ensure transaction is rolled back on error
                conn.rollback()
                logger.error(f"Failed to delete branch {branch_id}: {e}")
                return {
                    "success": False, 
                    "reason": f"Database error: {str(e)}", 
                    "deleted_message_count": 0,
                }
