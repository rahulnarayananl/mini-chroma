"""
Lightweight Vector Database - ChromaDB-like Architecture
Uses ONNX Runtime + HNSW + SQLite + FastAPI
No PyTorch or sentence-transformers dependencies
"""

import os
import json
import sqlite3
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import onnxruntime as ort
from tokenizers import Tokenizer
import hnswlib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path("./data")
SQLITE_DB_PATH = DATA_DIR / "documents.db"
HNSW_INDEX_PATH = DATA_DIR / "vectors.bin"
MODEL_DIR = Path("./models")
ONNX_MODEL_PATH = MODEL_DIR / "model.onnx"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"

# Model configuration for all-MiniLM-L6-v2
EMBEDDING_DIM = 384
MAX_SEQ_LENGTH = 256
HNSW_SPACE = 'cosine'
HNSW_EF_CONSTRUCTION = 200
HNSW_M = 16


# ============================================================================
# Pydantic Models for API
# ============================================================================

# Collection models
class CreateCollectionRequest(BaseModel):
    name: str
    metadata: Optional[Dict[str, Any]] = None
    distance_metric: str = 'cosine'


class UpdateCollectionRequest(BaseModel):
    metadata: Optional[Dict[str, Any]] = None


class CollectionInfo(BaseModel):
    name: str
    metadata: Optional[Dict[str, Any]]
    distance_metric: str
    count: int


# Document operation models
class BatchAddRequest(BaseModel):
    ids: Optional[List[str]] = None
    documents: Optional[List[str]] = None
    metadatas: Optional[List[Dict[str, Any]]] = None
    embeddings: Optional[List[List[float]]] = None


class UpdateRequest(BaseModel):
    ids: List[str]
    documents: Optional[List[str]] = None
    metadatas: Optional[List[Dict[str, Any]]] = None
    embeddings: Optional[List[List[float]]] = None


class UpsertRequest(BaseModel):
    ids: Optional[List[str]] = None
    documents: Optional[List[str]] = None
    metadatas: Optional[List[Dict[str, Any]]] = None
    embeddings: Optional[List[List[float]]] = None


class DeleteRequest(BaseModel):
    ids: Optional[List[str]] = None
    where: Optional[Dict[str, Any]] = None
    where_document: Optional[Dict[str, Any]] = None


class GetRequest(BaseModel):
    ids: Optional[List[str]] = None
    where: Optional[Dict[str, Any]] = None
    where_document: Optional[Dict[str, Any]] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
    include: List[str] = ["documents", "metadatas"]


class QueryRequest(BaseModel):
    query_texts: Optional[List[str]] = None
    query_embeddings: Optional[List[List[float]]] = None
    n_results: int = 10
    where: Optional[Dict[str, Any]] = None
    where_document: Optional[Dict[str, Any]] = None
    include: List[str] = ["documents", "metadatas", "distances"]


# Legacy models (kept for backward compatibility)
class AddRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None
    id: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    id: str
    text: str
    metadata: Optional[Dict[str, Any]]
    distance: float


# ============================================================================
# ONNX Embedder (No PyTorch, No sentence-transformers)
# ============================================================================

class ONNXEmbedder:
    """
    Handles text embedding using ONNX Runtime and tokenizers library.
    Mimics sentence-transformers behavior but uses only ONNX inference.
    """

    def __init__(self, model_path: Path, tokenizer_path: Path):
        """
        Initialize the ONNX embedder.

        Args:
            model_path: Path to the ONNX model file
            tokenizer_path: Path to the tokenizer.json file
        """
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}. "
                f"Download from: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 "
                f"and convert to ONNX, or use: optimum-cli export onnx --model sentence-transformers/all-MiniLM-L6-v2 {MODEL_DIR}"
            )

        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {tokenizer_path}. "
                f"Download from: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/tokenizer.json"
            )

        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(
            str(model_path),
            providers=['CPUExecutionProvider']
        )

        # Load tokenizer
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

        # Enable truncation and padding
        self.tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
        self.tokenizer.enable_padding(length=MAX_SEQ_LENGTH)

        print(f"✓ ONNX Embedder initialized with model: {model_path}")

    def mean_pooling(self, token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """
        Perform mean pooling on token embeddings, considering attention mask.
        This is the standard pooling strategy used by sentence-transformers.

        Args:
            token_embeddings: Shape (batch_size, seq_length, hidden_size)
            attention_mask: Shape (batch_size, seq_length)

        Returns:
            Pooled embeddings of shape (batch_size, hidden_size)
        """
        # Expand attention mask to match token_embeddings dimensions
        input_mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)

        # Sum embeddings, weighted by attention mask
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)

        # Sum attention mask to get the number of valid tokens
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)

        # Return mean
        return sum_embeddings / sum_mask

    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        L2 normalize embeddings.

        Args:
            embeddings: Shape (batch_size, hidden_size)

        Returns:
            Normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        return embeddings / norms

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            Normalized embeddings of shape (len(texts), EMBEDDING_DIM)
        """
        if not texts:
            return np.array([])

        # Tokenize all texts
        encodings = [self.tokenizer.encode(text) for text in texts]

        # Prepare inputs for ONNX model
        input_ids = np.array([enc.ids for enc in encodings], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask for enc in encodings], dtype=np.int64)

        # Run ONNX inference
        onnx_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

        # Some models also need token_type_ids
        try:
            onnx_inputs['token_type_ids'] = np.zeros_like(input_ids, dtype=np.int64)
            outputs = self.session.run(None, onnx_inputs)
        except Exception:
            # If token_type_ids not needed, retry without it
            del onnx_inputs['token_type_ids']
            outputs = self.session.run(None, onnx_inputs)

        # First output is typically the last hidden state
        token_embeddings = outputs[0]

        # Apply mean pooling
        embeddings = self.mean_pooling(token_embeddings, attention_mask)

        # Normalize embeddings
        embeddings = self.normalize_embeddings(embeddings)

        return embeddings


# ============================================================================
# Vector Database (HNSW + SQLite)
# ============================================================================

class VectorDB:
    """
    Complete vector database implementation with collections support.
    Manages embeddings (HNSW per collection), documents (SQLite), and persistence.
    """

    def __init__(self, data_dir: Path = DATA_DIR):
        """
        Initialize the vector database.

        Args:
            data_dir: Directory to store persistent data
        """
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize embedder
        self.embedder = ONNXEmbedder(ONNX_MODEL_PATH, TOKENIZER_PATH)

        # Initialize SQLite document store
        self._init_sqlite()

        # Dictionary to hold HNSW indexes per collection
        self.indexes: Dict[str, hnswlib.Index] = {}
        self.next_hnsw_ids: Dict[str, int] = {}

        # Load existing collections and their indexes
        self._load_collections()

        total_docs = sum(self.count(coll_id) for coll_id in self.indexes.keys())
        print(f"✓ VectorDB initialized ({len(self.indexes)} collections, {total_docs} documents)")

    def _init_sqlite(self):
        """Initialize SQLite database for collections and documents."""
        self.conn = sqlite3.connect(str(SQLITE_DB_PATH), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        cursor = self.conn.cursor()

        # Collections table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS collections (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                metadata TEXT,
                distance_metric TEXT DEFAULT 'cosine'
            )
        """)

        # Documents table with collection_id
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT NOT NULL,
                collection_id TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata TEXT,
                hnsw_id INTEGER NOT NULL,
                PRIMARY KEY (id, collection_id),
                FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE,
                UNIQUE (collection_id, hnsw_id)
            )
        """)

        self.conn.commit()
        print(f"✓ SQLite DB ready at {SQLITE_DB_PATH}")

    def _load_collections(self):
        """Load all existing collections and their HNSW indexes."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, distance_metric FROM collections")
        collections = cursor.fetchall()

        for coll in collections:
            coll_id = coll['id']
            distance_metric = coll['distance_metric']

            # Get document count for this collection
            cursor.execute("SELECT COUNT(*) as count FROM documents WHERE collection_id = ?", (coll_id,))
            doc_count = cursor.fetchone()['count']

            # Load or create HNSW index for this collection
            index_path = self.data_dir / f"vectors_{coll_id}.bin"
            index = hnswlib.Index(space=distance_metric, dim=EMBEDDING_DIM)

            if index_path.exists() and doc_count > 0:
                index.load_index(str(index_path), max_elements=doc_count)
                print(f"✓ Loaded HNSW index for collection '{coll['name']}' ({doc_count} docs)")
            else:
                initial_capacity = max(doc_count, 1000)
                index.init_index(
                    max_elements=initial_capacity,
                    ef_construction=HNSW_EF_CONSTRUCTION,
                    M=HNSW_M
                )
                print(f"✓ Created HNSW index for collection '{coll['name']}'")

            index.set_ef(50)
            self.indexes[coll_id] = index

            # Track next HNSW ID for this collection
            cursor.execute("SELECT MAX(hnsw_id) as max_id FROM documents WHERE collection_id = ?", (coll_id,))
            result = cursor.fetchone()
            self.next_hnsw_ids[coll_id] = (result['max_id'] or -1) + 1

    # ========================================================================
    # Collection Management
    # ========================================================================

    def create_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None,
                         distance_metric: str = 'cosine') -> str:
        """Create a new collection."""
        import uuid
        coll_id = str(uuid.uuid4())

        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO collections (id, name, metadata, distance_metric) VALUES (?, ?, ?, ?)",
                (coll_id, name, json.dumps(metadata) if metadata else None, distance_metric)
            )
            self.conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"Collection '{name}' already exists")

        # Create HNSW index for this collection
        index = hnswlib.Index(space=distance_metric, dim=EMBEDDING_DIM)
        index.init_index(
            max_elements=1000,
            ef_construction=HNSW_EF_CONSTRUCTION,
            M=HNSW_M
        )
        index.set_ef(50)

        self.indexes[coll_id] = index
        self.next_hnsw_ids[coll_id] = 0

        # Save the new index
        self._save_index(coll_id)

        return coll_id

    def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, metadata, distance_metric FROM collections")
        collections = []

        for row in cursor.fetchall():
            cursor.execute("SELECT COUNT(*) as count FROM documents WHERE collection_id = ?", (row['id'],))
            count = cursor.fetchone()['count']

            collections.append({
                'id': row['id'],
                'name': row['name'],
                'metadata': json.loads(row['metadata']) if row['metadata'] else None,
                'distance_metric': row['distance_metric'],
                'count': count
            })

        return collections

    def get_collection(self, name: str) -> Optional[Dict[str, Any]]:
        """Get collection by name."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, metadata, distance_metric FROM collections WHERE name = ?", (name,))
        row = cursor.fetchone()

        if not row:
            return None

        cursor.execute("SELECT COUNT(*) as count FROM documents WHERE collection_id = ?", (row['id'],))
        count = cursor.fetchone()['count']

        return {
            'id': row['id'],
            'name': row['name'],
            'metadata': json.loads(row['metadata']) if row['metadata'] else None,
            'distance_metric': row['distance_metric'],
            'count': count
        }

    def get_collection_by_id(self, coll_id: str) -> Optional[Dict[str, Any]]:
        """Get collection by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, metadata, distance_metric FROM collections WHERE id = ?", (coll_id,))
        row = cursor.fetchone()

        if not row:
            return None

        cursor.execute("SELECT COUNT(*) as count FROM documents WHERE collection_id = ?", (row['id'],))
        count = cursor.fetchone()['count']

        return {
            'id': row['id'],
            'name': row['name'],
            'metadata': json.loads(row['metadata']) if row['metadata'] else None,
            'distance_metric': row['distance_metric'],
            'count': count
        }

    def update_collection(self, name: str, metadata: Dict[str, Any]) -> bool:
        """Update collection metadata."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE collections SET metadata = ? WHERE name = ?",
            (json.dumps(metadata), name)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def delete_collection(self, name: str) -> bool:
        """Delete a collection and all its documents."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM collections WHERE name = ?", (name,))
        row = cursor.fetchone()

        if not row:
            return False

        coll_id = row['id']

        # Delete from SQLite (CASCADE will delete documents)
        cursor.execute("DELETE FROM collections WHERE id = ?", (coll_id,))
        self.conn.commit()

        # Remove HNSW index from memory and disk
        if coll_id in self.indexes:
            del self.indexes[coll_id]
            del self.next_hnsw_ids[coll_id]

        index_path = self.data_dir / f"vectors_{coll_id}.bin"
        if index_path.exists():
            index_path.unlink()

        return True

    # ========================================================================
    # Document Operations
    # ========================================================================

    def batch_add(self, collection_name: str, ids: Optional[List[str]] = None,
                  documents: Optional[List[str]] = None,
                  metadatas: Optional[List[Dict[str, Any]]] = None,
                  embeddings: Optional[List[List[float]]] = None) -> List[str]:
        """
        Batch add documents to a collection.

        Args:
            collection_name: Name of the collection
            ids: Optional list of document IDs
            documents: Optional list of document texts
            metadatas: Optional list of metadata dicts
            embeddings: Optional pre-computed embeddings

        Returns:
            List of document IDs
        """
        # Get collection
        coll = self.get_collection(collection_name)
        if not coll:
            raise ValueError(f"Collection '{collection_name}' not found")
        coll_id = coll['id']

        # Validate inputs
        if documents is None and embeddings is None:
            raise ValueError("Either documents or embeddings must be provided")

        # Determine batch size
        batch_size = len(documents) if documents else len(embeddings)

        # Generate IDs if not provided
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in range(batch_size)]
        elif len(ids) != batch_size:
            raise ValueError("Length of ids must match documents/embeddings")

        # Validate metadatas
        if metadatas is not None and len(metadatas) != batch_size:
            raise ValueError("Length of metadatas must match documents/embeddings")

        # Generate embeddings if not provided
        if embeddings is None:
            embeddings = self.embedder.embed(documents)
        else:
            embeddings = np.array(embeddings)

        # Get HNSW index for this collection
        index = self.indexes[coll_id]

        # Prepare batch insert
        cursor = self.conn.cursor()
        insert_data = []
        hnsw_ids = []

        for i in range(batch_size):
            doc_id = ids[i]
            text = documents[i] if documents else ""
            metadata = metadatas[i] if metadatas else None
            embedding = embeddings[i]

            # Get HNSW ID
            hnsw_id = self.next_hnsw_ids[coll_id]
            self.next_hnsw_ids[coll_id] += 1
            hnsw_ids.append(hnsw_id)

            # Resize index if needed
            if hnsw_id >= index.get_max_elements():
                new_size = index.get_max_elements() * 2
                index.resize_index(new_size)

            insert_data.append((
                doc_id,
                coll_id,
                text,
                json.dumps(metadata) if metadata else None,
                hnsw_id
            ))

        # Add all embeddings to HNSW in one call
        index.add_items(embeddings, hnsw_ids)

        # Batch insert into SQLite
        try:
            cursor.executemany(
                "INSERT INTO documents (id, collection_id, text, metadata, hnsw_id) VALUES (?, ?, ?, ?, ?)",
                insert_data
            )
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            self.conn.rollback()
            raise ValueError(f"Duplicate document ID: {e}")

        # Save index
        self._save_index(coll_id)

        return ids

    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None, doc_id: Optional[str] = None,
            collection_name: str = "default") -> str:
        """
        Add a single document (legacy method, now uses collections).

        Args:
            text: Document text to embed and store
            metadata: Optional metadata dictionary
            doc_id: Optional document ID (auto-generated if not provided)
            collection_name: Collection to add to (default: "default")

        Returns:
            Document ID
        """
        # Ensure default collection exists
        if not self.get_collection(collection_name):
            self.create_collection(collection_name)

        # Use batch_add with single document
        ids = self.batch_add(
            collection_name=collection_name,
            ids=[doc_id] if doc_id else None,
            documents=[text],
            metadatas=[metadata] if metadata else None
        )
        return ids[0]

    def update(self, collection_name: str, ids: List[str],
               documents: Optional[List[str]] = None,
               metadatas: Optional[List[Dict[str, Any]]] = None,
               embeddings: Optional[List[List[float]]] = None) -> int:
        """
        Update existing documents.

        Args:
            collection_name: Name of the collection
            ids: List of document IDs to update
            documents: Optional new document texts
            metadatas: Optional new metadata
            embeddings: Optional pre-computed embeddings

        Returns:
            Number of documents updated
        """
        coll = self.get_collection(collection_name)
        if not coll:
            raise ValueError(f"Collection '{collection_name}' not found")
        coll_id = coll['id']
        index = self.indexes[coll_id]

        cursor = self.conn.cursor()
        updated_count = 0

        for i, doc_id in enumerate(ids):
            # Check if document exists
            cursor.execute(
                "SELECT hnsw_id, text FROM documents WHERE id = ? AND collection_id = ?",
                (doc_id, coll_id)
            )
            row = cursor.fetchone()

            if not row:
                continue

            old_hnsw_id = row['hnsw_id']
            text_changed = documents is not None and i < len(documents)

            # If text changed, re-embed and update vector
            if text_changed:
                new_text = documents[i]
                if embeddings and i < len(embeddings):
                    new_embedding = np.array(embeddings[i])
                else:
                    new_embedding = self.embedder.embed([new_text])[0]

                # Mark old vector as deleted
                index.mark_deleted(old_hnsw_id)

                # Add new vector
                new_hnsw_id = self.next_hnsw_ids[coll_id]
                self.next_hnsw_ids[coll_id] += 1

                if new_hnsw_id >= index.get_max_elements():
                    new_size = index.get_max_elements() * 2
                    index.resize_index(new_size)

                index.add_items([new_embedding], [new_hnsw_id])

                # Update SQLite
                cursor.execute(
                    "UPDATE documents SET text = ?, hnsw_id = ? WHERE id = ? AND collection_id = ?",
                    (new_text, new_hnsw_id, doc_id, coll_id)
                )

            # Update metadata if provided
            if metadatas and i < len(metadatas):
                metadata_json = json.dumps(metadatas[i])
                cursor.execute(
                    "UPDATE documents SET metadata = ? WHERE id = ? AND collection_id = ?",
                    (metadata_json, doc_id, coll_id)
                )

            updated_count += 1

        self.conn.commit()
        self._save_index(coll_id)
        return updated_count

    def upsert(self, collection_name: str, ids: Optional[List[str]] = None,
               documents: Optional[List[str]] = None,
               metadatas: Optional[List[Dict[str, Any]]] = None,
               embeddings: Optional[List[List[float]]] = None) -> List[str]:
        """
        Insert or update documents.

        Args:
            collection_name: Name of the collection
            ids: List of document IDs
            documents: Optional document texts
            metadatas: Optional metadata
            embeddings: Optional pre-computed embeddings

        Returns:
            List of document IDs
        """
        if ids is None:
            # All new documents
            return self.batch_add(collection_name, None, documents, metadatas, embeddings)

        coll = self.get_collection(collection_name)
        if not coll:
            raise ValueError(f"Collection '{collection_name}' not found")
        coll_id = coll['id']

        cursor = self.conn.cursor()

        # Check which IDs exist
        existing_ids = set()
        for doc_id in ids:
            cursor.execute(
                "SELECT id FROM documents WHERE id = ? AND collection_id = ?",
                (doc_id, coll_id)
            )
            if cursor.fetchone():
                existing_ids.add(doc_id)

        # Split into update and insert
        to_update_ids = []
        to_update_docs = []
        to_update_metas = []
        to_update_embeds = []

        to_insert_ids = []
        to_insert_docs = []
        to_insert_metas = []
        to_insert_embeds = []

        for i, doc_id in enumerate(ids):
            if doc_id in existing_ids:
                to_update_ids.append(doc_id)
                if documents:
                    to_update_docs.append(documents[i])
                if metadatas:
                    to_update_metas.append(metadatas[i])
                if embeddings:
                    to_update_embeds.append(embeddings[i])
            else:
                to_insert_ids.append(doc_id)
                if documents:
                    to_insert_docs.append(documents[i])
                if metadatas:
                    to_insert_metas.append(metadatas[i])
                if embeddings:
                    to_insert_embeds.append(embeddings[i])

        # Update existing
        if to_update_ids:
            self.update(
                collection_name,
                to_update_ids,
                to_update_docs if to_update_docs else None,
                to_update_metas if to_update_metas else None,
                to_update_embeds if to_update_embeds else None
            )

        # Insert new
        if to_insert_ids:
            self.batch_add(
                collection_name,
                to_insert_ids,
                to_insert_docs if to_insert_docs else None,
                to_insert_metas if to_insert_metas else None,
                to_insert_embeds if to_insert_embeds else None
            )

        return ids

    def delete(self, collection_name: str, ids: Optional[List[str]] = None,
               where: Optional[Dict[str, Any]] = None,
               where_document: Optional[Dict[str, Any]] = None) -> int:
        """
        Delete documents from a collection.

        Args:
            collection_name: Name of the collection
            ids: Optional list of document IDs to delete
            where: Optional metadata filter
            where_document: Optional document text filter

        Returns:
            Number of documents deleted
        """
        coll = self.get_collection(collection_name)
        if not coll:
            raise ValueError(f"Collection '{collection_name}' not found")
        coll_id = coll['id']
        index = self.indexes[coll_id]

        cursor = self.conn.cursor()

        # Build query
        query = "SELECT id, hnsw_id FROM documents WHERE collection_id = ?"
        params = [coll_id]

        if ids:
            placeholders = ','.join('?' * len(ids))
            query += f" AND id IN ({placeholders})"
            params.extend(ids)

        # Apply filters
        if where:
            filter_ids = self._apply_metadata_filter(coll_id, where)
            if filter_ids is not None:
                if not filter_ids:
                    return 0
                placeholders = ','.join('?' * len(filter_ids))
                query += f" AND id IN ({placeholders})"
                params.extend(filter_ids)

        if where_document:
            filter_ids = self._apply_document_filter(coll_id, where_document)
            if filter_ids is not None:
                if not filter_ids:
                    return 0
                placeholders = ','.join('?' * len(filter_ids))
                query += f" AND id IN ({placeholders})"
                params.extend(filter_ids)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Mark deleted in HNSW and delete from SQLite
        deleted_count = 0
        for row in rows:
            index.mark_deleted(row['hnsw_id'])
            cursor.execute(
                "DELETE FROM documents WHERE id = ? AND collection_id = ?",
                (row['id'], coll_id)
            )
            deleted_count += 1

        self.conn.commit()
        self._save_index(coll_id)
        return deleted_count

    def get(self, collection_name: str, ids: Optional[List[str]] = None,
            where: Optional[Dict[str, Any]] = None,
            where_document: Optional[Dict[str, Any]] = None,
            limit: Optional[int] = None,
            offset: Optional[int] = None,
            include: List[str] = ["documents", "metadatas"]) -> Dict[str, List]:
        """
        Get documents by ID or filter.

        Args:
            collection_name: Name of the collection
            ids: Optional list of document IDs
            where: Optional metadata filter
            where_document: Optional document text filter
            limit: Maximum number of results
            offset: Number of results to skip
            include: Fields to include in response

        Returns:
            Dictionary with requested fields
        """
        coll = self.get_collection(collection_name)
        if not coll:
            raise ValueError(f"Collection '{collection_name}' not found")
        coll_id = coll['id']

        cursor = self.conn.cursor()

        # Build query
        query = "SELECT id, text, metadata, hnsw_id FROM documents WHERE collection_id = ?"
        params = [coll_id]

        if ids:
            placeholders = ','.join('?' * len(ids))
            query += f" AND id IN ({placeholders})"
            params.extend(ids)

        # Apply filters
        if where:
            filter_ids = self._apply_metadata_filter(coll_id, where)
            if filter_ids is not None:
                if not filter_ids:
                    return self._empty_result(include)
                placeholders = ','.join('?' * len(filter_ids))
                query += f" AND id IN ({placeholders})"
                params.extend(filter_ids)

        if where_document:
            filter_ids = self._apply_document_filter(coll_id, where_document)
            if filter_ids is not None:
                if not filter_ids:
                    return self._empty_result(include)
                placeholders = ','.join('?' * len(filter_ids))
                query += f" AND id IN ({placeholders})"
                params.extend(filter_ids)

        # Apply limit and offset
        if limit:
            query += f" LIMIT {limit}"
        if offset:
            query += f" OFFSET {offset}"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Build response (IDs always included)
        result = {"ids": [row['id'] for row in rows]}

        if "documents" in include:
            result["documents"] = [row['text'] for row in rows]
        if "metadatas" in include:
            result["metadatas"] = [
                json.loads(row['metadata']) if row['metadata'] else None
                for row in rows
            ]
        if "embeddings" in include:
            index = self.indexes[coll_id]
            hnsw_ids = [row['hnsw_id'] for row in rows]
            if hnsw_ids:
                embeddings = index.get_items(hnsw_ids)
                result["embeddings"] = embeddings.tolist()
            else:
                result["embeddings"] = []

        return result

    def search(self, query: str, top_k: int = 5, collection_name: str = "default") -> List[Dict[str, Any]]:
        """
        Search for similar documents (legacy method).

        Args:
            query: Query text
            top_k: Number of results to return
            collection_name: Collection to search in

        Returns:
            List of search results with id, text, metadata, and distance
        """
        # Ensure default collection exists
        if not self.get_collection(collection_name):
            return []

        # Use query method
        result = self.query(
            collection_name=collection_name,
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # Convert to legacy format
        if not result['ids'] or not result['ids'][0]:
            return []

        results = []
        for i in range(len(result['ids'][0])):
            results.append({
                'id': result['ids'][0][i],
                'text': result['documents'][0][i],
                'metadata': result['metadatas'][0][i],
                'distance': result['distances'][0][i]
            })

        return results

    def query(self, collection_name: str,
              query_texts: Optional[List[str]] = None,
              query_embeddings: Optional[List[List[float]]] = None,
              n_results: int = 10,
              where: Optional[Dict[str, Any]] = None,
              where_document: Optional[Dict[str, Any]] = None,
              include: List[str] = ["documents", "metadatas", "distances"]) -> Dict[str, List]:
        """
        Query documents by similarity with filtering.

        Args:
            collection_name: Name of the collection
            query_texts: Optional list of query texts
            query_embeddings: Optional pre-computed query embeddings
            n_results: Number of results per query
            where: Optional metadata filter
            where_document: Optional document text filter
            include: Fields to include in response

        Returns:
            Dictionary with lists of results for each query
        """
        coll = self.get_collection(collection_name)
        if not coll:
            raise ValueError(f"Collection '{collection_name}' not found")
        coll_id = coll['id']
        index = self.indexes[coll_id]

        # Generate embeddings if needed
        if query_embeddings is None and query_texts is None:
            raise ValueError("Either query_texts or query_embeddings must be provided")

        if query_embeddings is None:
            query_embeddings = self.embedder.embed(query_texts)
        else:
            query_embeddings = np.array(query_embeddings)

        num_queries = len(query_embeddings)

        # Build filter function if needed
        filter_func = None
        if where or where_document:
            allowed_hnsw_ids = set()

            # Apply metadata filter
            if where:
                filter_ids = self._apply_metadata_filter(coll_id, where)
                if filter_ids is not None:
                    cursor = self.conn.cursor()
                    for doc_id in filter_ids:
                        cursor.execute(
                            "SELECT hnsw_id FROM documents WHERE id = ? AND collection_id = ?",
                            (doc_id, coll_id)
                        )
                        row = cursor.fetchone()
                        if row:
                            allowed_hnsw_ids.add(row['hnsw_id'])

            # Apply document filter
            if where_document:
                filter_ids = self._apply_document_filter(coll_id, where_document)
                if filter_ids is not None:
                    cursor = self.conn.cursor()
                    doc_hnsw_ids = set()
                    for doc_id in filter_ids:
                        cursor.execute(
                            "SELECT hnsw_id FROM documents WHERE id = ? AND collection_id = ?",
                            (doc_id, coll_id)
                        )
                        row = cursor.fetchone()
                        if row:
                            doc_hnsw_ids.add(row['hnsw_id'])

                    # Intersect with metadata filter if both present
                    if where:
                        allowed_hnsw_ids &= doc_hnsw_ids
                    else:
                        allowed_hnsw_ids = doc_hnsw_ids

            if allowed_hnsw_ids:
                filter_func = lambda label: label in allowed_hnsw_ids
            else:
                # No matches
                return self._empty_query_result(num_queries, include)

        # Perform search
        cursor = self.conn.cursor()
        all_ids = []
        all_documents = []
        all_metadatas = []
        all_distances = []
        all_embeddings = []

        for query_emb in query_embeddings:
            try:
                if filter_func:
                    labels, distances = index.knn_query(
                        [query_emb],
                        k=n_results,
                        filter=filter_func
                    )
                else:
                    labels, distances = index.knn_query([query_emb], k=n_results)
            except RuntimeError:
                # No results (possibly all filtered out)
                all_ids.append([])
                all_documents.append([])
                all_metadatas.append([])
                all_distances.append([])
                all_embeddings.append([])
                continue

            ids = []
            documents = []
            metadatas = []
            dists = []
            embeddings = []

            for hnsw_id, distance in zip(labels[0], distances[0]):
                cursor.execute(
                    "SELECT id, text, metadata FROM documents WHERE hnsw_id = ? AND collection_id = ?",
                    (int(hnsw_id), coll_id)
                )
                row = cursor.fetchone()

                if row:
                    ids.append(row['id'])
                    documents.append(row['text'])
                    metadatas.append(json.loads(row['metadata']) if row['metadata'] else None)
                    dists.append(float(distance))

                    if "embeddings" in include:
                        embeddings.append(query_emb.tolist())

            all_ids.append(ids)
            all_documents.append(documents)
            all_metadatas.append(metadatas)
            all_distances.append(dists)
            all_embeddings.append(embeddings)

        # Build response (IDs always included)
        result = {"ids": all_ids}

        if "documents" in include:
            result["documents"] = all_documents
        if "metadatas" in include:
            result["metadatas"] = all_metadatas
        if "distances" in include:
            result["distances"] = all_distances
        if "embeddings" in include:
            result["embeddings"] = all_embeddings

        return result

    def count(self, collection_id: Optional[str] = None) -> int:
        """Return the number of documents."""
        cursor = self.conn.cursor()
        if collection_id:
            cursor.execute("SELECT COUNT(*) as count FROM documents WHERE collection_id = ?", (collection_id,))
        else:
            cursor.execute("SELECT COUNT(*) as count FROM documents")
        return cursor.fetchone()['count']

    def _save_index(self, coll_id: str):
        """Persist HNSW index for a collection to disk."""
        index_path = self.data_dir / f"vectors_{coll_id}.bin"
        self.indexes[coll_id].save_index(str(index_path))

    def _save(self):
        """Persist all HNSW indexes to disk."""
        for coll_id in self.indexes:
            self._save_index(coll_id)

    def _apply_metadata_filter(self, coll_id: str, where: Dict[str, Any]) -> Optional[List[str]]:
        """
        Apply metadata filter and return matching document IDs.

        Supports: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $and, $or
        """
        cursor = self.conn.cursor()

        # Handle logical operators
        if "$and" in where:
            all_ids = None
            for condition in where["$and"]:
                ids = self._apply_metadata_filter(coll_id, condition)
                if ids is None:
                    continue
                if all_ids is None:
                    all_ids = set(ids)
                else:
                    all_ids &= set(ids)
            return list(all_ids) if all_ids else []

        if "$or" in where:
            all_ids = set()
            for condition in where["$or"]:
                ids = self._apply_metadata_filter(coll_id, condition)
                if ids:
                    all_ids.update(ids)
            return list(all_ids)

        # Get all documents with metadata
        cursor.execute(
            "SELECT id, metadata FROM documents WHERE collection_id = ? AND metadata IS NOT NULL",
            (coll_id,)
        )
        rows = cursor.fetchall()

        matching_ids = []
        for row in rows:
            metadata = json.loads(row['metadata'])
            if self._match_metadata(metadata, where):
                matching_ids.append(row['id'])

        return matching_ids

    def _match_metadata(self, metadata: Dict[str, Any], where: Dict[str, Any]) -> bool:
        """Check if metadata matches the where clause."""
        for key, condition in where.items():
            if key.startswith('$'):
                continue  # Skip operators

            if key not in metadata:
                return False

            value = metadata[key]

            if isinstance(condition, dict):
                # Handle operators
                for op, op_value in condition.items():
                    if op == "$eq" and value != op_value:
                        return False
                    elif op == "$ne" and value == op_value:
                        return False
                    elif op == "$gt" and not (value > op_value):
                        return False
                    elif op == "$gte" and not (value >= op_value):
                        return False
                    elif op == "$lt" and not (value < op_value):
                        return False
                    elif op == "$lte" and not (value <= op_value):
                        return False
                    elif op == "$in" and value not in op_value:
                        return False
                    elif op == "$nin" and value in op_value:
                        return False
            else:
                # Direct equality
                if value != condition:
                    return False

        return True

    def _apply_document_filter(self, coll_id: str, where_document: Dict[str, Any]) -> Optional[List[str]]:
        """
        Apply document text filter and return matching document IDs.

        Supports: $contains, $not_contains
        """
        cursor = self.conn.cursor()

        matching_ids = []

        for key, value in where_document.items():
            if key == "$contains":
                cursor.execute(
                    "SELECT id FROM documents WHERE collection_id = ? AND text LIKE ?",
                    (coll_id, f"%{value}%")
                )
                matching_ids = [row['id'] for row in cursor.fetchall()]

            elif key == "$not_contains":
                cursor.execute(
                    "SELECT id FROM documents WHERE collection_id = ? AND text NOT LIKE ?",
                    (coll_id, f"%{value}%")
                )
                matching_ids = [row['id'] for row in cursor.fetchall()]

        return matching_ids

    def _empty_result(self, include: List[str]) -> Dict[str, List]:
        """Return empty result for get operation (IDs always included)."""
        result = {"ids": []}

        if "documents" in include:
            result["documents"] = []
        if "metadatas" in include:
            result["metadatas"] = []
        if "embeddings" in include:
            result["embeddings"] = []
        return result

    def _empty_query_result(self, num_queries: int, include: List[str]) -> Dict[str, List]:
        """Return empty result for query operation (IDs always included)."""
        result = {"ids": [[] for _ in range(num_queries)]}

        if "documents" in include:
            result["documents"] = [[] for _ in range(num_queries)]
        if "metadatas" in include:
            result["metadatas"] = [[] for _ in range(num_queries)]
        if "distances" in include:
            result["distances"] = [[] for _ in range(num_queries)]
        if "embeddings" in include:
            result["embeddings"] = [[] for _ in range(num_queries)]
        return result

    def close(self):
        """Close database connections and save state."""
        self._save()
        self.conn.close()
        print("✓ VectorDB closed and saved")


# ============================================================================
# FastAPI Application
# ============================================================================

# Initialize FastAPI app
app = FastAPI(
    title="Lightweight Vector Database",
    description="ChromaDB-like vector database using ONNX + HNSW + SQLite",
    version="1.0.0"
)

# Initialize VectorDB (singleton)
db: Optional[VectorDB] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the vector database on startup."""
    global db
    print("\n" + "="*70)
    print("Starting Lightweight Vector Database")
    print("="*70 + "\n")

    # Check if models exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not ONNX_MODEL_PATH.exists() or not TOKENIZER_PATH.exists():
        print("\n⚠️  SETUP REQUIRED ⚠️")
        print("\nTo use this vector database, you need to download the model files:")
        print("\n1. Install optimum CLI:")
        print("   pip install optimum[onnxruntime]")
        print("\n2. Export the model to ONNX:")
        print(f"   optimum-cli export onnx --model sentence-transformers/all-MiniLM-L6-v2 {MODEL_DIR}")
        print("\nThis will download and convert all-MiniLM-L6-v2 to ONNX format.\n")
        raise FileNotFoundError("Model files not found. See instructions above.")

    db = VectorDB()
    print("\n✓ Server ready!\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown."""
    global db
    if db:
        db.close()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "documents": db.count() if db else 0,
        "embedding_dim": EMBEDDING_DIM,
        "model": "all-MiniLM-L6-v2 (ONNX)"
    }


@app.post("/add")
async def add_document(request: AddRequest) -> Dict[str, str]:
    """
    Add a document to the vector database.

    Embeds the text using ONNX Runtime and stores it in both HNSW and SQLite.
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        doc_id = db.add(
            text=request.text,
            metadata=request.metadata,
            doc_id=request.id
        )
        return {
            "id": doc_id,
            "status": "added",
            "message": f"Document added successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding document: {str(e)}")


@app.post("/search", response_model=List[SearchResult])
async def search_documents(request: SearchRequest) -> List[SearchResult]:
    """
    Search for similar documents.

    Embeds the query using ONNX Runtime, searches HNSW, and fetches results from SQLite.
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        results = db.search(query=request.query, top_k=request.top_k)
        return [SearchResult(**result) for result in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get database statistics."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    return {
        "total_documents": db.count(),
        "embedding_dim": EMBEDDING_DIM,
        "hnsw_space": HNSW_SPACE,
        "max_seq_length": MAX_SEQ_LENGTH,
        "data_dir": str(DATA_DIR),
        "index_size_bytes": HNSW_INDEX_PATH.stat().st_size if HNSW_INDEX_PATH.exists() else 0,
        "db_size_bytes": SQLITE_DB_PATH.stat().st_size if SQLITE_DB_PATH.exists() else 0
    }


# ============================================================================
# Collection Endpoints
# ============================================================================

@app.post("/collections")
async def create_collection(request: CreateCollectionRequest) -> Dict[str, Any]:
    """Create a new collection."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        coll_id = db.create_collection(
            name=request.name,
            metadata=request.metadata,
            distance_metric=request.distance_metric
        )
        return {
            "id": coll_id,
            "name": request.name,
            "status": "created"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating collection: {str(e)}")


@app.get("/collections")
async def list_collections() -> List[CollectionInfo]:
    """List all collections."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        collections = db.list_collections()
        return [CollectionInfo(**coll) for coll in collections]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing collections: {str(e)}")


@app.get("/collections/{name}")
async def get_collection(name: str) -> CollectionInfo:
    """Get collection by name."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        coll = db.get_collection(name)
        if not coll:
            raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
        return CollectionInfo(**coll)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting collection: {str(e)}")


@app.put("/collections/{name}")
async def update_collection(name: str, request: UpdateCollectionRequest) -> Dict[str, str]:
    """Update collection metadata."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        updated = db.update_collection(name, request.metadata)
        if not updated:
            raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
        return {"status": "updated", "name": name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating collection: {str(e)}")


@app.delete("/collections/{name}")
async def delete_collection(name: str) -> Dict[str, str]:
    """Delete a collection and all its documents."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        deleted = db.delete_collection(name)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
        return {"status": "deleted", "name": name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting collection: {str(e)}")


# ============================================================================
# Collection Document Endpoints
# ============================================================================

@app.post("/collections/{name}/add")
async def add_to_collection(name: str, request: BatchAddRequest) -> Dict[str, Any]:
    """Add documents to a collection."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        ids = db.batch_add(
            collection_name=name,
            ids=request.ids,
            documents=request.documents,
            metadatas=request.metadatas,
            embeddings=request.embeddings
        )
        return {
            "ids": ids,
            "status": "added",
            "count": len(ids)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding documents: {str(e)}")


@app.post("/collections/{name}/update")
async def update_in_collection(name: str, request: UpdateRequest) -> Dict[str, Any]:
    """Update documents in a collection."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        count = db.update(
            collection_name=name,
            ids=request.ids,
            documents=request.documents,
            metadatas=request.metadatas,
            embeddings=request.embeddings
        )
        return {
            "status": "updated",
            "count": count
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating documents: {str(e)}")


@app.post("/collections/{name}/upsert")
async def upsert_in_collection(name: str, request: UpsertRequest) -> Dict[str, Any]:
    """Upsert documents in a collection."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        ids = db.upsert(
            collection_name=name,
            ids=request.ids,
            documents=request.documents,
            metadatas=request.metadatas,
            embeddings=request.embeddings
        )
        return {
            "ids": ids,
            "status": "upserted",
            "count": len(ids)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error upserting documents: {str(e)}")


@app.post("/collections/{name}/delete")
async def delete_from_collection(name: str, request: DeleteRequest) -> Dict[str, Any]:
    """Delete documents from a collection."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        count = db.delete(
            collection_name=name,
            ids=request.ids,
            where=request.where,
            where_document=request.where_document
        )
        return {
            "status": "deleted",
            "count": count
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting documents: {str(e)}")


@app.post("/collections/{name}/get")
async def get_from_collection(name: str, request: GetRequest) -> Dict[str, List]:
    """Get documents from a collection by ID or filter."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        result = db.get(
            collection_name=name,
            ids=request.ids,
            where=request.where,
            where_document=request.where_document,
            limit=request.limit,
            offset=request.offset,
            include=request.include
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting documents: {str(e)}")


@app.post("/collections/{name}/query")
async def query_collection(name: str, request: QueryRequest) -> Dict[str, List]:
    """Query documents in a collection by similarity."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        result = db.query(
            collection_name=name,
            query_texts=request.query_texts,
            query_embeddings=request.query_embeddings,
            n_results=request.n_results,
            where=request.where,
            where_document=request.where_document,
            include=request.include
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Lightweight Vector Database Server")
    print("ChromaDB-like architecture with ONNX + HNSW + SQLite")
    print("="*70 + "\n")

    # Run the FastAPI server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
