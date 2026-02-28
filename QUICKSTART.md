# Quick Start Guide

Get your vector database running in 3 minutes!

## Step 1: Install Dependencies

```bash
uv venv
uv pip install -r requirements.txt
```

This installs:
- ✅ `onnxruntime` - For running ONNX models (no PyTorch!)
- ✅ `tokenizers` - For text tokenization
- ✅ `hnswlib` - For fast vector search
- ✅ `fastapi` + `uvicorn` - For the REST API
- ✅ `sqlite3` - Built into Python (document storage)

## Step 2: Download the Model

```bash
uv run setup_model.py
```

This will:
1. Install `optimum[onnxruntime]`
2. Download `all-MiniLM-L6-v2` from Hugging Face
3. Convert it to ONNX format (~90MB)
4. Save to `./models/` directory

**Expected output:**
```
Step 1: Installing optimum for ONNX export...
✓ Optimum installed

Step 2: Downloading and converting all-MiniLM-L6-v2 to ONNX...
This will take a few minutes...

✓ Model successfully exported to ONNX format!
✓ Verified model files:
  - models/model.onnx (90.3 MB)
  - models/tokenizer.json (466 KB)

Setup complete! You can now run: python vector_db.py
```

## Step 3: Start the Server

```bash
uv run vector_db.py
```

**Expected output:**
```
======================================================================
Starting Lightweight Vector Database
======================================================================

✓ ONNX Embedder initialized with model: models/model.onnx
✓ SQLite DB ready at data/documents.db
✓ HNSW index created (capacity: 1000)
✓ VectorDB initialized (documents: 0)

✓ Server ready!

INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Leave this terminal running!

## Step 4: Test It!

Open a **new terminal** and run:

```bash
uv run test_db.py
```

This will:
1. Check server status
2. Add 8 sample documents about programming, AI, and databases
3. Run 4 semantic search queries
4. Display database statistics

**Sample output:**
```
======================================================================
 3. Running Search Queries
======================================================================

Query: 'web development and APIs'
----------------------------------------------------------------------

1. Distance: 0.2847
   Text: FastAPI is a modern, fast web framework for building APIs...
   Metadata: {'category': 'programming', 'framework': 'FastAPI'}

2. Distance: 0.4521
   Text: Python is a high-level, interpreted programming language...
   Metadata: {'category': 'programming', 'language': 'python'}
```

## Step 5: Try Your Own Queries!

### Add a document:

```bash
curl -X POST "http://localhost:8000/add" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Docker is a platform for developing and running containerized applications.",
    "metadata": {"category": "devops", "tool": "docker"}
  }'
```

### Search:

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "container technology",
    "top_k": 3
  }'
```

### Get stats:

```bash
curl "http://localhost:8000/stats"
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Server                         │
│                     (Port 8000)                             │
└───────────────┬─────────────────────────┬───────────────────┘
                │                         │
       ┌────────▼────────┐       ┌───────▼────────┐
       │  ONNX Embedder  │       │   Vector DB    │
       │   (tokenizers   │       │                │
       │  + onnxruntime) │       │                │
       └────────┬────────┘       └───┬────────┬───┘
                │                    │        │
                │ 384-dim vectors    │        │
                │                    │        │
                └────────────────────┘        │
                         │                    │
                   ┌─────▼──────┐      ┌─────▼─────┐
                   │   HNSW     │      │  SQLite   │
                   │  (hnswlib) │      │   (docs)  │
                   │            │      │           │
                   │ vectors.bin│      │documents.db
                   └────────────┘      └───────────┘
```

## Next Steps

1. **Read the full README.md** for detailed documentation
2. **Explore vector_db.py** to understand the implementation
3. **Modify the code** to add features like:
   - Batch operations
   - Metadata filtering
   - Multiple collections
   - Authentication

## Troubleshooting

### "Model files not found"
Run: `python setup_model.py`

### "Port 8000 already in use"
Change the port in `vector_db.py` or:
```bash
uv run uvicorn vector_db:app --port 8001
```

### "Server not running" in test_db.py
Make sure `uv run vector_db.py` is running in another terminal

### Start fresh
```bash
rm -rf data/ models/
uv run setup_model.py
uv run vector_db.py
```

## Performance Notes

- **Embedding speed**: ~50-100 docs/sec on CPU
- **Search speed**: Sub-millisecond for 10K docs
- **Memory usage**: ~200MB base + ~1.5KB per document
- **Disk usage**: ~2KB per document (vectors + text)

## What Makes This Special?

✨ **No PyTorch dependency** - Uses ONNX Runtime instead
✨ **Lightweight** - Only 7 dependencies in requirements.txt
✨ **Fast** - HNSW for approximate nearest neighbor search
✨ **Persistent** - All data saved to disk automatically
✨ **Simple** - Single file, ~500 lines of well-commented code
✨ **Production-ready** - FastAPI with proper error handling

Enjoy your vector database! 🚀
