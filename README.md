# Lightweight Vector Database

A from-scratch implementation of a local vector database mimicking ChromaDB's lightweight architecture. **Zero PyTorch dependencies** - uses ONNX Runtime for embeddings.

## Architecture

This implementation recreates ChromaDB's exact lightweight architecture:

1. **Embedder**: ONNX Runtime + tokenizers library
   - Runs the `all-MiniLM-L6-v2` model in ONNX format
   - Manual tokenization and mean pooling
   - No sentence-transformers or PyTorch

2. **Vector Index**: hnswlib
   - Approximate Nearest Neighbor (ANN) search
   - Fast cosine similarity search
   - Persistent binary index

3. **Document Store**: SQLite
   - Stores original text and metadata
   - Maps documents to HNSW vector IDs
   - Transactional and durable

4. **API**: FastAPI
   - `POST /add` - Add documents with automatic embedding
   - `POST /search` - Semantic search with vector similarity
   - Full persistence across restarts

## Installation

### 1. Install Dependencies

```bash
uv venv
uv pip install -r requirements.txt
```

### 2. Download and Convert Model to ONNX

```bash
uv run setup_model.py
```

This will:
- Install `optimum[onnxruntime]`
- Download `all-MiniLM-L6-v2` from Hugging Face
- Convert it to ONNX format (~90MB)
- Save to `./models/` directory

## Usage

### Start the Server

```bash
uv run vector_db.py
```

Server runs on `http://localhost:8000`

### Add Documents

```bash
curl -X POST "http://localhost:8000/add" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The quick brown fox jumps over the lazy dog",
    "metadata": {"source": "example", "category": "animals"}
  }'
```

Response:
```json
{
  "id": "a1b2c3d4-...",
  "status": "added",
  "message": "Document added successfully"
}
```

### Search Documents

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "fast animals",
    "top_k": 5
  }'
```

Response:
```json
[
  {
    "id": "a1b2c3d4-...",
    "text": "The quick brown fox jumps over the lazy dog",
    "metadata": {"source": "example", "category": "animals"},
    "distance": 0.234
  }
]
```

### Get Statistics

```bash
curl "http://localhost:8000/stats"
```

## Running Tests

To test all Chroma-compatible features:

```bash
# Start the server
uv run vector_db.py

# Run comprehensive test suite
uv run test_collections.py
```

The test suite covers:
- Collections CRUD operations
- Batch add/update/upsert/delete
- Metadata and document filtering
- Query with filters
- Include/exclude fields
- Multiple queries in one request

## Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000"

# Add documents
documents = [
    "Python is a high-level programming language",
    "Machine learning uses statistical techniques",
    "Neural networks are inspired by the brain",
    "FastAPI is a modern web framework for Python"
]

for doc in documents:
    response = requests.post(
        f"{BASE_URL}/add",
        json={"text": doc, "metadata": {"source": "docs"}}
    )
    print(f"Added: {response.json()['id']}")

# Search
response = requests.post(
    f"{BASE_URL}/search",
    json={"query": "web development with Python", "top_k": 3}
)

for result in response.json():
    print(f"\nDistance: {result['distance']:.3f}")
    print(f"Text: {result['text']}")
```

## Persistence

All data is automatically persisted to disk:

- **Vector Index**: `./data/vectors.bin` (HNSW binary format)
- **Documents**: `./data/documents.db` (SQLite database)

Data survives server restarts. Delete these files to reset the database.

## Features

✅ **No PyTorch** - Uses ONNX Runtime for embeddings
✅ **Lightweight** - Minimal dependencies
✅ **Fast** - HNSW for sub-millisecond ANN search
✅ **Persistent** - SQLite + binary HNSW index
✅ **Scalable** - Auto-resizing HNSW index
✅ **Production-ready** - FastAPI with proper error handling

## API Endpoints

### `GET /`
Health check and server status

### `POST /add`
Add a document to the database

**Request Body:**
```json
{
  "text": "Document text to embed",
  "metadata": {"key": "value"},  // optional
  "id": "custom-id"  // optional, auto-generated if not provided
}
```

### `POST /search`
Search for similar documents

**Request Body:**
```json
{
  "query": "Search query text",
  "top_k": 5  // optional, default 5
}
```

### `GET /stats`
Get database statistics (document count, index size, etc.)

## Configuration

Edit these constants in `vector_db.py`:

```python
DATA_DIR = Path("./data")              # Data directory
EMBEDDING_DIM = 384                    # Model embedding dimension
MAX_SEQ_LENGTH = 256                   # Maximum token length
HNSW_SPACE = 'cosine'                  # Distance metric
HNSW_EF_CONSTRUCTION = 200             # Index construction quality
HNSW_M = 16                            # Index connectivity
```

## How It Works

### Adding Documents

1. Text is tokenized using the HuggingFace tokenizer
2. Tokens are fed into the ONNX model for inference
3. Token embeddings are mean-pooled and L2-normalized
4. Vector is added to HNSW index with auto-generated ID
5. Document text and metadata stored in SQLite
6. Both HNSW and SQLite persisted to disk

### Searching

1. Query text is embedded using the same process
2. HNSW performs fast ANN search (cosine similarity)
3. Top-K vector IDs retrieved
4. Documents fetched from SQLite by ID
5. Results returned with distance scores

## Chroma-Compatible Features

This implementation now includes full Chroma-compatible features:

✅ **Collections**: Multiple isolated collections with different distance metrics
✅ **Batch Operations**: Efficient batch add/update/upsert/delete
✅ **Pre-computed Embeddings**: Pass your own embeddings to skip generation
✅ **Metadata Filtering**: Query with complex filters ($eq, $ne, $gt, $in, $and, $or)
✅ **Document Filtering**: Filter by document text ($contains, $not_contains)
✅ **Get by ID**: Retrieve specific documents without search
✅ **Update/Upsert**: Modify existing documents or create new ones
✅ **Delete with Filters**: Remove documents by ID or metadata
✅ **Include/Exclude Fields**: Control which fields are returned
✅ **Multiple Distance Metrics**: Cosine, L2, IP per collection

### Collections API

```bash
# Create a collection
curl -X POST "http://localhost:8000/collections" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_documents",
    "metadata": {"description": "My document collection"},
    "distance_metric": "cosine"
  }'

# List all collections
curl "http://localhost:8000/collections"

# Get collection info
curl "http://localhost:8000/collections/my_documents"

# Update collection metadata
curl -X PUT "http://localhost:8000/collections/my_documents" \
  -H "Content-Type: application/json" \
  -d '{"metadata": {"description": "Updated description"}}'

# Delete a collection
curl -X DELETE "http://localhost:8000/collections/my_documents"
```

### Batch Add with Filters

```bash
# Add multiple documents
curl -X POST "http://localhost:8000/collections/my_documents/add" \
  -H "Content-Type: application/json" \
  -d '{
    "ids": ["doc1", "doc2", "doc3"],
    "documents": ["Document 1 text", "Document 2 text", "Document 3 text"],
    "metadatas": [
      {"category": "A", "year": 2024},
      {"category": "B", "year": 2023},
      {"category": "A", "year": 2024}
    ]
  }'
```

### Query with Metadata Filtering

```bash
# Query with metadata filter
curl -X POST "http://localhost:8000/collections/my_documents/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query_texts": ["search query"],
    "n_results": 5,
    "where": {"category": "A", "year": {"$gte": 2024}},
    "include": ["documents", "metadatas", "distances"]
  }'

# Complex filter with $and/$or
curl -X POST "http://localhost:8000/collections/my_documents/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query_texts": ["search query"],
    "where": {
      "$or": [
        {"category": "A"},
        {"$and": [{"category": "B"}, {"year": {"$gt": 2023}}]}
      ]
    }
  }'
```

### Get Documents by ID

```bash
# Get specific documents
curl -X POST "http://localhost:8000/collections/my_documents/get" \
  -H "Content-Type: application/json" \
  -d '{
    "ids": ["doc1", "doc2"],
    "include": ["documents", "metadatas", "embeddings"]
  }'

# Get with metadata filter
curl -X POST "http://localhost:8000/collections/my_documents/get" \
  -H "Content-Type: application/json" \
  -d '{
    "where": {"category": "A"},
    "limit": 10,
    "offset": 0
  }'
```

### Update/Upsert/Delete

```bash
# Update existing documents
curl -X POST "http://localhost:8000/collections/my_documents/update" \
  -H "Content-Type: application/json" \
  -d '{
    "ids": ["doc1"],
    "documents": ["Updated text"],
    "metadatas": [{"category": "A", "updated": true}]
  }'

# Upsert (insert or update)
curl -X POST "http://localhost:8000/collections/my_documents/upsert" \
  -H "Content-Type: application/json" \
  -d '{
    "ids": ["doc1", "doc4"],
    "documents": ["Updated doc1", "New doc4"]
  }'

# Delete by ID
curl -X POST "http://localhost:8000/collections/my_documents/delete" \
  -H "Content-Type: application/json" \
  -d '{"ids": ["doc1", "doc2"]}'

# Delete by metadata filter
curl -X POST "http://localhost:8000/collections/my_documents/delete" \
  -H "Content-Type: application/json" \
  -d '{"where": {"category": "B"}}'
```

## Differences from ChromaDB

This implementation uses the **exact same architecture** as local ChromaDB:
- ONNX-based embedding generation (no PyTorch)
- HNSW for vector indexing (one per collection)
- SQLite for document storage
- RESTful API with full Chroma-compatible features

Key features now included:
- ✅ Collections with isolated namespaces
- ✅ Batch operations for efficiency
- ✅ Metadata and document filtering
- ✅ Update/upsert/delete operations
- ✅ Get by ID without search
- ✅ Include/exclude field control
- ✅ Pre-computed embeddings support

## Troubleshooting

**Model not found error:**
```bash
uv run setup_model.py
```

**Port 8000 already in use:**
```bash
uv run uvicorn vector_db:app --port 8001
```

**Database corruption:**
```bash
rm -rf ./data
# Restart server to recreate
```

## License

MIT
