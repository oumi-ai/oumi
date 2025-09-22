"""Graph store implementation for chat branches.

This module provides a graph-based representation of chat branches to enable
more advanced operations and visualizations of conversations.
"""

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import logging


class GraphStore:
    """Stores a graph representation of conversation branches.
    
    Uses a separate table in the same SQLite database to track branch
    relationships and enable graph operations on the conversation history.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize graph store with connection to SQLite database.
        
        Args:
            db_path: Path to SQLite database file. If None, uses default path.
        """
        default_path = os.path.expanduser("~/.oumi/webchat.sqlite")
        self.db_path = db_path or os.environ.get("OUMI_DB_PATH", default_path)
        self._lock = threading.Lock()
        
        # Ensure database directory exists
        if self.db_path:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize schema
        self._logger = logging.getLogger("oumi")
        with self._connect() as conn:
            self._init_schema(conn)
            self._detect_column_compat(conn)
        self._logger.debug(f"[GraphStore] Initialized at {self.db_path} src_col={getattr(self,'_src_col',None)} dst_col={getattr(self,'_dst_col',None)}")
        # Default column names; may be overwritten by _detect_column_compat
        if not hasattr(self, '_src_col'):
            self._src_col = 'source_branch_id'
        if not hasattr(self, '_dst_col'):
            self._dst_col = 'target_branch_id'
    
    def _connect(self) -> sqlite3.Connection:
        """Connect to SQLite database with proper settings.
        
        Returns:
            SQLite connection object
        """
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn
    
    def _init_schema(self, conn: sqlite3.Connection) -> None:
        """Initialize graph_edges table if it doesn't exist.
        
        Args:
            conn: SQLite connection
        """
        cur = conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS graph_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                source_branch_id TEXT NOT NULL,
                target_branch_id TEXT NOT NULL,
                relationship_type TEXT DEFAULT 'parent',
                created_at REAL NOT NULL,
                metadata TEXT,
                UNIQUE(conversation_id, source_branch_id, target_branch_id),
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_graph_edges_conv_source ON graph_edges(conversation_id, source_branch_id);
            CREATE INDEX IF NOT EXISTS idx_graph_edges_conv_target ON graph_edges(conversation_id, target_branch_id);
            """
        )
        conn.commit()

    def _detect_column_compat(self, conn: sqlite3.Connection) -> None:
        """Detect existing column names for backward compatibility.

        Older databases used src_branch_id/dst_branch_id. Newer uses
        source_branch_id/target_branch_id. Set internal column names accordingly.
        """
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info('graph_edges')")
            cols = [r[1] for r in cur.fetchall()]
            if not cols:
                # Table missing; leave defaults
                return
            if 'source_branch_id' in cols:
                self._src_col = 'source_branch_id'
            elif 'src_branch_id' in cols:
                self._src_col = 'src_branch_id'
            if 'target_branch_id' in cols:
                self._dst_col = 'target_branch_id'
            elif 'dst_branch_id' in cols:
                self._dst_col = 'dst_branch_id'
            # Also detect message-id columns
            if 'source_message_id' in cols:
                self._src_msg_col = 'source_message_id'
            elif 'src_message_id' in cols:
                self._src_msg_col = 'src_message_id'
            if 'target_message_id' in cols:
                self._dst_msg_col = 'target_message_id'
            elif 'dst_message_id' in cols:
                self._dst_msg_col = 'dst_message_id'
            self._logger.debug(f"[GraphStore] Detected graph_edges columns: {cols} -> src={getattr(self,'_src_col',None)} dst={getattr(self,'_dst_col',None)}")
        except Exception:
            # Fail open with defaults
            self._src_col = 'source_branch_id'
            self._dst_col = 'target_branch_id'
            # Message columns unknown by default
    
    def add_edge(self, conversation_id: str, source_branch_id: str, target_branch_id: str, 
                relationship_type: str = 'parent', metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a directed edge between two branches.
        
        Args:
            conversation_id: ID of the conversation
            source_branch_id: ID of the source branch
            target_branch_id: ID of the target branch
            relationship_type: Type of relationship (default: 'parent')
            metadata: Optional JSON string with additional metadata
        """
        if not conversation_id or not source_branch_id or not target_branch_id:
            return
            
        with self._lock, self._connect() as conn:
            cur = conn.cursor()
            
            # Check if branches exist
            cur.execute(
                "SELECT 1 FROM branches WHERE id = ? AND conversation_id = ?",
                (source_branch_id, conversation_id)
            )
            if not cur.fetchone():
                # Source branch doesn't exist
                return
                
            cur.execute(
                "SELECT 1 FROM branches WHERE id = ? AND conversation_id = ?",
                (target_branch_id, conversation_id)
            )
            if not cur.fetchone():
                # Target branch doesn't exist
                return
            
            # Add edge (will be ignored if it already exists due to UNIQUE constraint)
            self._logger.info(f"[GraphStore] add_edge conv={conversation_id} src={source_branch_id} dst={target_branch_id} cols=({self._src_col},{self._dst_col})")
            try:
                cur.execute(
                    f"""
                    INSERT OR IGNORE INTO graph_edges(
                        conversation_id, {self._src_col}, {self._dst_col}, 
                        relationship_type, created_at, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (conversation_id, source_branch_id, target_branch_id, 
                     relationship_type, time.time(), json.dumps(metadata) if metadata is not None else None)
                )
            except sqlite3.OperationalError as oe:
                # Fallback to legacy column names if present
                self._logger.warning(f"[GraphStore] primary insert failed: {oe}. Attempting legacy columns.")
                legacy_src = 'src_branch_id'
                legacy_dst = 'dst_branch_id'
                try:
                    cur.execute(
                        f"""
                        INSERT OR IGNORE INTO graph_edges(
                            conversation_id, {legacy_src}, {legacy_dst}, 
                            relationship_type, created_at, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (conversation_id, source_branch_id, target_branch_id, 
                         relationship_type, time.time(), json.dumps(metadata) if metadata is not None else None)
                    )
                    # Update detected columns for next time
                    self._src_col, self._dst_col = legacy_src, legacy_dst
                    self._logger.info("[GraphStore] Legacy insert succeeded; switched to legacy column names.")
                except Exception as oe2:
                    self._logger.error(f"[GraphStore] legacy insert failed: {oe2}")
                    raise
            conn.commit()
    
    def add_edge_for_branch_tail(self, conversation_id: str, branch_id: str) -> None:
        """Add an edge from the current branch to the last message branch.
        
        This special edge type captures message flow in the conversation history.
        Used for dual-write during normal chat operations.
        
        Args:
            conversation_id: ID of the conversation
            branch_id: ID of the branch with a new message
        """
        if not conversation_id or not branch_id:
            return
        
        with self._lock, self._connect() as conn:
            cur = conn.cursor()
            
            # We're adding an edge from the branch to itself as a "tail" relationship
            # This represents adding a message to the branch
            self._logger.info(f"[GraphStore] add_tail conv={conversation_id} branch={branch_id} cols=({self._src_col},{self._dst_col})")
            try:
                cur.execute(
                    f"""
                    INSERT OR IGNORE INTO graph_edges(
                        conversation_id, {self._src_col}, {self._dst_col}, 
                        relationship_type, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (conversation_id, branch_id, branch_id, 'tail', time.time())
                )
            except sqlite3.OperationalError as oe:
                self._logger.warning(f"[GraphStore] primary tail insert failed: {oe}. Attempting legacy columns.")
                legacy_src = 'src_branch_id'
                legacy_dst = 'dst_branch_id'
                try:
                    cur.execute(
                        f"""
                        INSERT OR IGNORE INTO graph_edges(
                            conversation_id, {legacy_src}, {legacy_dst}, 
                            relationship_type, created_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (conversation_id, branch_id, branch_id, 'tail', time.time())
                    )
                    self._src_col, self._dst_col = legacy_src, legacy_dst
                    self._logger.info("[GraphStore] Legacy tail insert succeeded; switched to legacy column names.")
                except Exception as oe2:
                    self._logger.error(f"[GraphStore] legacy tail insert failed: {oe2}")
                    raise
            conn.commit()

    def add_edge_for_message_tail(self, conversation_id: str, message_id: str) -> None:
        """Add a tail edge using message ID columns when schema uses message-based edges.

        If message-id columns are not present, this call is a no-op.
        """
        if not conversation_id or not message_id:
            return
        with self._lock, self._connect() as conn:
            cur = conn.cursor()
            try:
                # Ensure table exists and detect columns if needed
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='graph_edges'")
                if not cur.fetchone():
                    self._logger.debug("[GraphStore] add_edge_for_message_tail: graph_edges table missing; skipping")
                    return
                # Lazy detect if not set
                if not hasattr(self, '_src_msg_col') or not hasattr(self, '_dst_msg_col'):
                    self._detect_column_compat(conn)
                if not getattr(self, '_src_msg_col', None) or not getattr(self, '_dst_msg_col', None):
                    self._logger.debug("[GraphStore] add_edge_for_message_tail: message-id columns not present; skipping")
                    return
                self._logger.info(f"[GraphStore] add_msg_tail conv={conversation_id} msg={message_id} cols=({self._src_msg_col},{self._dst_msg_col})")
                cur.execute(
                    f"""
                    INSERT OR IGNORE INTO graph_edges(
                        conversation_id, {self._src_msg_col}, {self._dst_msg_col},
                        relationship_type, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (conversation_id, message_id, message_id, 'tail', time.time())
                )
                conn.commit()
            except Exception as e:
                self._logger.warning(f"[GraphStore] add_edge_for_message_tail failed: {e}")
    
    def get_branch_connections(self, conversation_id: str, branch_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get all connections for a specific branch.
        
        Args:
            conversation_id: ID of the conversation
            branch_id: ID of the branch
            
        Returns:
            Dictionary with incoming and outgoing connections
        """
        with self._lock, self._connect() as conn:
            cur = conn.cursor()
            
            # Get outgoing connections
            cur.execute(
                f"""
                SELECT {self._dst_col}, relationship_type, created_at, metadata
                FROM graph_edges
                WHERE conversation_id = ? AND {self._src_col} = ?
                ORDER BY created_at ASC
                """,
                (conversation_id, branch_id)
            )
            outgoing = []
            for row in cur.fetchall():
                target_id, rel_type, created_at, metadata = row
                outgoing.append({
                    "branch_id": target_id,
                    "relationship_type": rel_type,
                    "created_at": created_at,
                    "metadata": self._parse_metadata(metadata)
                })
            
            # Get incoming connections
            cur.execute(
                f"""
                SELECT {self._src_col}, relationship_type, created_at, metadata
                FROM graph_edges
                WHERE conversation_id = ? AND {self._dst_col} = ?
                ORDER BY created_at ASC
                """,
                (conversation_id, branch_id)
            )
            incoming = []
            for row in cur.fetchall():
                source_id, rel_type, created_at, metadata = row
                incoming.append({
                    "branch_id": source_id,
                    "relationship_type": rel_type,
                    "created_at": created_at,
                    "metadata": self._parse_metadata(metadata)
                })
                
            return {
                "outgoing": outgoing,
                "incoming": incoming
            }
    
    def get_conversation_graph(self, conversation_id: str) -> Dict[str, Any]:
        """Get the full graph structure for a conversation.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Dictionary with nodes and edges representing the graph
        """
        with self._lock, self._connect() as conn:
            cur = conn.cursor()
            
            # Get all branches (nodes)
            cur.execute(
                """
                SELECT b.id, b.name, b.parent_branch_id, b.created_at,
                       COUNT(bm.message_id) as message_count
                FROM branches b
                LEFT JOIN branch_messages bm ON b.id = bm.branch_id
                WHERE b.conversation_id = ?
                GROUP BY b.id
                ORDER BY b.created_at ASC
                """,
                (conversation_id,)
            )
            
            nodes = []
            for row in cur.fetchall():
                branch_id, name, parent_id, created_at, msg_count = row
                nodes.append({
                    "id": branch_id,
                    "name": name or branch_id,
                    "parent_id": parent_id,
                    "created_at": created_at,
                    "message_count": msg_count
                })
            
            # Get all edges
            cur.execute(
                """
                SELECT source_branch_id, target_branch_id, relationship_type, created_at
                FROM graph_edges
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                """,
                (conversation_id,)
            )
            
            edges = []
            for row in cur.fetchall():
                source_id, target_id, rel_type, created_at = row
                edges.append({
                    "source": source_id,
                    "target": target_id,
                    "type": rel_type,
                    "created_at": created_at
                })
                
            return {
                "nodes": nodes,
                "edges": edges
            }
    
    def _parse_metadata(self, metadata_str: Optional[str]) -> Optional[Dict[str, Any]]:
        """Parse JSON metadata string into dict.
        
        Args:
            metadata_str: JSON string or None
            
        Returns:
            Dict or None if parsing fails or input is None
        """
        if metadata_str is None:
            return None
            
        try:
            return json.loads(metadata_str)
        except (json.JSONDecodeError, TypeError):
            return None
    
    def backfill_graph_from_branches(self, conversation_id: str) -> int:
        """Backfill graph edges based on branch parent relationships.
        
        This is used to populate the graph structure for existing conversations.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Number of edges created
        """
        edges_created = 0
        
        with self._lock, self._connect() as conn:
            cur = conn.cursor()
            
            # Get all branches with parent relationships
            cur.execute(
                """
                SELECT id, parent_branch_id
                FROM branches
                WHERE conversation_id = ? AND parent_branch_id IS NOT NULL
                """,
                (conversation_id,)
            )
            
            for row in cur.fetchall():
                branch_id, parent_id = row
                
                # Add edge from parent to branch
                cur.execute(
                    """
                    INSERT OR IGNORE INTO graph_edges(
                        conversation_id, source_branch_id, target_branch_id, 
                        relationship_type, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (conversation_id, parent_id, branch_id, 'parent', time.time())
                )
                
                if cur.rowcount > 0:
                    edges_created += 1
                    
                # Add "tail" edge for each branch to itself
                cur.execute(
                    """
                    INSERT OR IGNORE INTO graph_edges(
                        conversation_id, source_branch_id, target_branch_id, 
                        relationship_type, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (conversation_id, branch_id, branch_id, 'tail', time.time())
                )
                
                if cur.rowcount > 0:
                    edges_created += 1
            
            conn.commit()
            
        return edges_created
    
    def delete_branch(self, conversation_id: str, branch_id: str) -> int:
        """Delete all graph edges related to a specific branch.
        
        Args:
            conversation_id: ID of the conversation
            branch_id: ID of the branch to remove
            
        Returns:
            Number of edges deleted
        """
        if not conversation_id or not branch_id:
            return 0
        
        with self._lock, self._connect() as conn:
            try:
                cur = conn.cursor()
                
                # Delete edges where this branch is either source or target
                cur.execute(
                    """
                    DELETE FROM graph_edges 
                    WHERE conversation_id = ? AND (source_branch_id = ? OR target_branch_id = ?)
                    """,
                    (conversation_id, branch_id, branch_id)
                )
                
                deleted_count = cur.rowcount
                conn.commit()
                return deleted_count
                
            except Exception as e:
                conn.rollback()
                import logging
                logging.warning(f"Failed to delete graph edges for branch {branch_id}: {e}")
                return 0
    
    def branch_has_graph_references(self, conversation_id: str, branch_id: str) -> bool:
        """Check if a branch is referenced in any graph edges as a source.
        
        Args:
            conversation_id: ID of the conversation
            branch_id: ID of the branch to check
            
        Returns:
            True if branch is referenced in graph, False otherwise
        """
        if not conversation_id or not branch_id:
            return False
        
        with self._lock, self._connect() as conn:
            cur = conn.cursor()
            
            # Check if branch is a source in any edge (other than to itself)
            cur.execute(
                """
                SELECT 1 FROM graph_edges 
                WHERE conversation_id = ? AND source_branch_id = ? AND target_branch_id != ?
                LIMIT 1
                """,
                (conversation_id, branch_id, branch_id)
            )
            
            return bool(cur.fetchone())
