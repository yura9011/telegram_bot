import logging
import sqlite3
import json
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

class PersistentHybridMemory:
    """Manages persistent hybrid memory using SQLite and embeddings."""
    
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        database_config = config.get("database", {})
        self.db_path = database_config.get("db_path", "chat_memory.db")
        self.short_term_limit = database_config.get("short_term_limit", 10)
        self.top_k = database_config.get("top_k", 3)
        
        # Initialize embedding model
        embedding_model_config = config.get("embedding_model", {})
        model_name = embedding_model_config.get(
            "model_name", 
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize database
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._cursor = self._conn.cursor()
        self._create_table()

    def _create_table(self):
        """Creates the messages table and necessary indexes."""
        self._cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER,
                role TEXT,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                embedding TEXT
            )
        ''')
        
        self._cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_chat_timestamp 
            ON messages (chat_id, timestamp DESC)
        ''')
        
        self._conn.commit()

    def add_message(self, chat_id: int, role: str, content: str):
        """Adds a message with its embedding to the database."""
        embedding = self.embedding_model.encode(content).tolist()
        embedding_str = json.dumps(embedding)
        
        self._cursor.execute('''
            INSERT INTO messages (chat_id, role, content, embedding)
            VALUES (?, ?, ?, ?)
        ''', (chat_id, role, content, embedding_str))
        
        self._conn.commit()

    def get_recent_messages(self, chat_id: int) -> List[Dict[str, str]]:
        """Retrieves recent messages (short-term memory)."""
        self._cursor.execute('''
            SELECT role, content FROM messages
            WHERE chat_id = ?
            ORDER BY timestamp ASC
            LIMIT ?
        ''', (chat_id, self.short_term_limit))
        
        rows = self._cursor.fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def get_similar_messages(self, chat_id: int, user_message: str) -> List[Dict[str, str]]:
        """Retrieves relevant messages based on embedding similarity."""
        query_embedding = self.embedding_model.encode(user_message)
        
        self._cursor.execute('''
            SELECT role, content, embedding FROM messages
            WHERE chat_id = ?
        ''', (chat_id,))
        
        rows = self._cursor.fetchall()
        similarities = []
        
        for row in rows:
            try:
                emb = np.array(json.loads(row["embedding"]))
                similarity = np.dot(query_embedding, emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(emb)
                )
                similarities.append((
                    similarity, 
                    {"role": row["role"], "content": row["content"]}
                ))
            except Exception as e:
                self.logger.error(f"Error processing embedding: {e}")
                continue
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in similarities[:self.top_k]]

    def get_context(self, chat_id: int, user_message: str) -> str:
        """Combines short-term and long-term memory into context string."""
        messages = []
        
        # Add recent messages
        recent = self.get_recent_messages(chat_id)
        for msg in recent:
            messages.append(f"{msg['role']}: {msg['content']}")
        
        # Add relevant historical messages
        similar = self.get_similar_messages(chat_id, user_message)
        for msg in similar:
            messages.append(f"{msg['role']} (remembered): {msg['content']}")
        
        return "\n".join(messages)