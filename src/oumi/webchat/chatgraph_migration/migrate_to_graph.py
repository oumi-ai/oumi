#!/usr/bin/env python
"""Migration script to set up graph_edges table and backfill existing data.

Usage:
  python -m chatgraph_migration.migrate_to_graph

This script:
1. Creates the graph_edges table if it doesn't exist
2. Backfills edges for all existing conversations
3. Reports statistics on the migration
"""

import os
import sys
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Any

from graph_store import GraphStore


def get_all_conversation_ids(db_path: str) -> List[str]:
    """Get all conversation IDs from the database.
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        List of conversation IDs
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id FROM conversations")
    return [row[0] for row in cur.fetchall()]


def get_conversation_stats(db_path: str, conversation_id: str) -> Dict[str, Any]:
    """Get statistics for a conversation.
    
    Args:
        db_path: Path to SQLite database
        conversation_id: ID of the conversation
        
    Returns:
        Dictionary with conversation statistics
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Count total messages
    cur.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conversation_id,))
    total_messages = cur.fetchone()[0]
    
    # Count branches
    cur.execute("SELECT COUNT(*) FROM branches WHERE conversation_id = ?", (conversation_id,))
    total_branches = cur.fetchone()[0]
    
    return {
        "conversation_id": conversation_id,
        "total_messages": total_messages,
        "total_branches": total_branches,
    }


def main():
    """Run the migration to graph structure."""
    # Get database path
    default_path = os.path.expanduser("~/.oumi/webchat.sqlite")
    db_path = os.environ.get("OUMI_DB_PATH", default_path)
    
    # Ensure database directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting migration to graph structure using database: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"Error: Database file {db_path} does not exist.")
        print("Run the webchat server first to create the initial database.")
        sys.exit(1)
    
    # Initialize GraphStore
    graph_store = GraphStore(db_path)
    
    # Get all conversation IDs
    conversation_ids = get_all_conversation_ids(db_path)
    print(f"Found {len(conversation_ids)} conversations to migrate")
    
    total_edges_created = 0
    start_time = time.time()
    
    # Process each conversation
    for i, conv_id in enumerate(conversation_ids):
        stats = get_conversation_stats(db_path, conv_id)
        print(f"Processing conversation {i+1}/{len(conversation_ids)}: {conv_id} "
              f"({stats['total_branches']} branches, {stats['total_messages']} messages)")
        
        # Backfill graph edges
        edges_created = graph_store.backfill_graph_from_branches(conv_id)
        total_edges_created += edges_created
        
        print(f"  - Created {edges_created} edges for conversation {conv_id}")
    
    elapsed_time = time.time() - start_time
    print(f"\nMigration completed in {elapsed_time:.2f} seconds")
    print(f"Created a total of {total_edges_created} graph edges")
    print("\nGraph structure is now available for all conversations.")
    print("The next step is to enable dual-write for new messages.")


if __name__ == "__main__":
    main()