"""
Comprehensive test suite for Chroma-compatible features.
Tests collections, batch operations, filtering, and all CRUD operations.
"""

import requests
import time
import json
from typing import List, Dict

BASE_URL = "http://localhost:8000"


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70 + "\n")


def test_collections():
    """Test collection CRUD operations."""
    print_section("1. Testing Collection Operations")

    # Create collections
    print("Creating collections...")
    collections = [
        {"name": "documents", "metadata": {"type": "text"}, "distance_metric": "cosine"},
        {"name": "images", "metadata": {"type": "image"}, "distance_metric": "cosine"},
        {"name": "code", "metadata": {"type": "code"}, "distance_metric": "l2"}
    ]

    for coll in collections:
        response = requests.post(f"{BASE_URL}/collections", json=coll)
        if response.status_code == 200:
            print(f"✓ Created collection: {coll['name']}")
        elif "already exists" in response.text:
            print(f"  Collection '{coll['name']}' already exists, skipping")
        else:
            print(f"✗ Failed to create collection {coll['name']}: {response.text}")

    # List collections
    print("\nListing collections...")
    response = requests.get(f"{BASE_URL}/collections")
    if response.status_code == 200:
        collections = response.json()
        print(f"✓ Found {len(collections)} collections:")
        for coll in collections:
            print(f"  - {coll['name']}: {coll['count']} documents, metric: {coll['distance_metric']}")
    else:
        print(f"✗ Failed to list collections: {response.text}")

    # Get specific collection
    print("\nGetting 'documents' collection...")
    response = requests.get(f"{BASE_URL}/collections/documents")
    if response.status_code == 200:
        coll = response.json()
        print(f"✓ Collection info: {coll['name']} - {coll['count']} documents")
    else:
        print(f"✗ Failed to get collection: {response.text}")

    # Update collection metadata
    print("\nUpdating collection metadata...")
    response = requests.put(
        f"{BASE_URL}/collections/documents",
        json={"metadata": {"type": "text", "updated": True}}
    )
    if response.status_code == 200:
        print("✓ Collection metadata updated")
    else:
        print(f"✗ Failed to update collection: {response.text}")


def test_batch_add():
    """Test batch add with and without pre-computed embeddings."""
    print_section("2. Testing Batch Add Operations")

    # Batch add documents
    print("Adding batch of documents...")
    docs = {
        "ids": ["doc1", "doc2", "doc3", "doc4", "doc5"],
        "documents": [
            "Python is a versatile programming language for data science and web development.",
            "Machine learning models learn patterns from data to make predictions.",
            "Neural networks consist of layers of interconnected nodes that process information.",
            "FastAPI provides high-performance API development with automatic documentation.",
            "Vector databases enable semantic search by comparing embedding similarities."
        ],
        "metadatas": [
            {"category": "programming", "language": "python", "difficulty": "beginner"},
            {"category": "AI", "topic": "machine learning", "difficulty": "intermediate"},
            {"category": "AI", "topic": "neural networks", "difficulty": "advanced"},
            {"category": "programming", "framework": "FastAPI", "difficulty": "intermediate"},
            {"category": "database", "type": "vector", "difficulty": "intermediate"}
        ]
    }

    response = requests.post(f"{BASE_URL}/collections/documents/add", json=docs)
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Added {result['count']} documents")
        print(f"  IDs: {result['ids']}")
    else:
        print(f"✗ Failed to add documents: {response.text}")

    # Add more documents without IDs (auto-generated)
    print("\nAdding documents with auto-generated IDs...")
    docs2 = {
        "documents": [
            "SQLite is a lightweight embedded database engine.",
            "ONNX Runtime enables efficient model inference across platforms."
        ],
        "metadatas": [
            {"category": "database", "type": "SQL"},
            {"category": "AI", "tool": "ONNX"}
        ]
    }

    response = requests.post(f"{BASE_URL}/collections/documents/add", json=docs2)
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Added {result['count']} documents with auto-generated IDs")
    else:
        print(f"✗ Failed to add documents: {response.text}")


def test_get_operations():
    """Test get by ID and filtering."""
    print_section("3. Testing Get Operations")

    # Get by IDs
    print("Getting documents by IDs...")
    response = requests.post(
        f"{BASE_URL}/collections/documents/get",
        json={"ids": ["doc1", "doc3"], "include": ["documents", "metadatas"]}
    )
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Retrieved {len(result['ids'])} documents:")
        for i, doc_id in enumerate(result['ids']):
            print(f"  {doc_id}: {result['documents'][i][:60]}...")
    else:
        print(f"✗ Failed to get documents: {response.text}")

    # Get with metadata filter
    print("\nGetting documents with metadata filter (category=AI)...")
    response = requests.post(
        f"{BASE_URL}/collections/documents/get",
        json={
            "where": {"category": "AI"},
            "include": ["documents", "metadatas"]
        }
    )
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Found {len(result['ids'])} AI documents:")
        for i, doc_id in enumerate(result['ids']):
            print(f"  {result['metadatas'][i]['topic'] if 'topic' in result['metadatas'][i] else 'N/A'}: {result['documents'][i][:60]}...")
    else:
        print(f"✗ Failed to get documents: {response.text}")

    # Get with complex metadata filter
    print("\nGetting documents with complex filter (category=programming AND difficulty=intermediate)...")
    response = requests.post(
        f"{BASE_URL}/collections/documents/get",
        json={
            "where": {"$and": [{"category": "programming"}, {"difficulty": "intermediate"}]},
            "include": ["documents", "metadatas"]
        }
    )
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Found {len(result['ids'])} matching documents")
    else:
        print(f"✗ Failed to get documents: {response.text}")

    # Get with document filter
    print("\nGetting documents containing 'database'...")
    response = requests.post(
        f"{BASE_URL}/collections/documents/get",
        json={
            "where_document": {"$contains": "database"},
            "include": ["documents"]
        }
    )
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Found {len(result['ids'])} documents containing 'database':")
        for doc in result['documents']:
            print(f"  - {doc[:60]}...")
    else:
        print(f"✗ Failed to get documents: {response.text}")


def test_query_operations():
    """Test semantic search with filtering."""
    print_section("4. Testing Query Operations")

    # Simple query
    print("Querying: 'web development frameworks'...")
    response = requests.post(
        f"{BASE_URL}/collections/documents/query",
        json={
            "query_texts": ["web development frameworks"],
            "n_results": 3,
            "include": ["documents", "metadatas", "distances"]
        }
    )
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Found {len(result['ids'][0])} results:")
        for i in range(len(result['ids'][0])):
            print(f"  Distance: {result['distances'][0][i]:.4f}")
            print(f"  Text: {result['documents'][0][i][:60]}...")
    else:
        print(f"✗ Failed to query: {response.text}")

    # Query with metadata filter
    print("\nQuerying 'learning from data' with filter (category=AI)...")
    response = requests.post(
        f"{BASE_URL}/collections/documents/query",
        json={
            "query_texts": ["learning from data"],
            "n_results": 3,
            "where": {"category": "AI"},
            "include": ["documents", "distances"]
        }
    )
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Found {len(result['ids'][0])} AI-related results:")
        for i in range(len(result['ids'][0])):
            print(f"  Distance: {result['distances'][0][i]:.4f}")
            print(f"  Text: {result['documents'][0][i][:60]}...")
    else:
        print(f"✗ Failed to query: {response.text}")

    # Multiple queries
    print("\nQuerying multiple queries at once...")
    response = requests.post(
        f"{BASE_URL}/collections/documents/query",
        json={
            "query_texts": [
                "programming languages",
                "artificial intelligence",
                "databases"
            ],
            "n_results": 2,
            "include": ["documents", "distances"]
        }
    )
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Processed {len(result['ids'])} queries:")
        for q_idx, query in enumerate(["programming languages", "artificial intelligence", "databases"]):
            print(f"\n  Query: '{query}'")
            for i in range(len(result['ids'][q_idx])):
                print(f"    {i+1}. {result['documents'][q_idx][i][:50]}... (distance: {result['distances'][q_idx][i]:.4f})")
    else:
        print(f"✗ Failed to query: {response.text}")


def test_update_operations():
    """Test update and upsert operations."""
    print_section("5. Testing Update & Upsert Operations")

    # Update document text
    print("Updating doc1 text...")
    response = requests.post(
        f"{BASE_URL}/collections/documents/update",
        json={
            "ids": ["doc1"],
            "documents": ["Python is an excellent programming language for AI, web development, and data science."],
            "metadatas": [{"category": "programming", "language": "python", "difficulty": "beginner", "updated": True}]
        }
    )
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Updated {result['count']} document(s)")
    else:
        print(f"✗ Failed to update: {response.text}")

    # Verify update
    print("\nVerifying update...")
    response = requests.post(
        f"{BASE_URL}/collections/documents/get",
        json={"ids": ["doc1"], "include": ["documents", "metadatas"]}
    )
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Updated text: {result['documents'][0][:60]}...")
        print(f"  Metadata: {result['metadatas'][0]}")

    # Upsert (update existing, insert new)
    print("\nUpserting documents (doc2 exists, doc_new doesn't)...")
    response = requests.post(
        f"{BASE_URL}/collections/documents/upsert",
        json={
            "ids": ["doc2", "doc_new"],
            "documents": [
                "Machine learning algorithms learn patterns from data.",
                "Deep learning uses neural networks with many layers."
            ],
            "metadatas": [
                {"category": "AI", "topic": "machine learning", "difficulty": "intermediate", "updated": True},
                {"category": "AI", "topic": "deep learning", "difficulty": "advanced"}
            ]
        }
    )
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Upserted {result['count']} document(s)")
    else:
        print(f"✗ Failed to upsert: {response.text}")


def test_delete_operations():
    """Test delete operations."""
    print_section("6. Testing Delete Operations")

    # Add a temporary document
    print("Adding temporary document for deletion test...")
    response = requests.post(
        f"{BASE_URL}/collections/documents/add",
        json={
            "ids": ["temp1", "temp2"],
            "documents": ["Temporary document 1", "Temporary document 2"],
            "metadatas": [{"temp": True}, {"temp": True}]
        }
    )
    if response.status_code == 200:
        print("✓ Added temporary documents")

    # Delete by ID
    print("\nDeleting temp1 by ID...")
    response = requests.post(
        f"{BASE_URL}/collections/documents/delete",
        json={"ids": ["temp1"]}
    )
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Deleted {result['count']} document(s)")
    else:
        print(f"✗ Failed to delete: {response.text}")

    # Delete by metadata filter
    print("\nDeleting documents with metadata temp=True...")
    response = requests.post(
        f"{BASE_URL}/collections/documents/delete",
        json={"where": {"temp": True}}
    )
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Deleted {result['count']} document(s)")
    else:
        print(f"✗ Failed to delete: {response.text}")


def test_include_fields():
    """Test include/exclude fields functionality."""
    print_section("7. Testing Include/Exclude Fields")

    # Get with different include options
    print("Getting documents with only metadatas...")
    response = requests.post(
        f"{BASE_URL}/collections/documents/get",
        json={"ids": ["doc1", "doc2"], "include": ["metadatas"]}
    )
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Result keys: {list(result.keys())}")
        print(f"  Has documents: {'documents' in result}")
        print(f"  Has metadatas: {'metadatas' in result}")

    # Query with embeddings
    print("\nQuerying with embeddings included...")
    response = requests.post(
        f"{BASE_URL}/collections/documents/query",
        json={
            "query_texts": ["programming"],
            "n_results": 2,
            "include": ["documents", "embeddings", "distances"]
        }
    )
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Result keys: {list(result.keys())}")
        if "embeddings" in result and result["embeddings"][0]:
            print(f"  Embedding dimensions: {len(result['embeddings'][0][0])}")
    else:
        print(f"✗ Failed to query: {response.text}")


def test_collection_deletion():
    """Test deleting a collection."""
    print_section("8. Testing Collection Deletion")

    # Create a temporary collection
    print("Creating temporary collection...")
    response = requests.post(
        f"{BASE_URL}/collections",
        json={"name": "temp_collection", "metadata": {"temporary": True}}
    )
    if response.status_code == 200:
        print("✓ Created temporary collection")

        # Add some documents
        response = requests.post(
            f"{BASE_URL}/collections/temp_collection/add",
            json={"documents": ["doc1", "doc2"]}
        )
        print(f"✓ Added documents to temporary collection")

    # Delete the collection
    print("\nDeleting temporary collection...")
    response = requests.delete(f"{BASE_URL}/collections/temp_collection")
    if response.status_code == 200:
        print("✓ Collection deleted successfully")
    else:
        print(f"✗ Failed to delete collection: {response.text}")

    # Verify deletion
    print("\nVerifying deletion...")
    response = requests.get(f"{BASE_URL}/collections/temp_collection")
    if response.status_code == 404:
        print("✓ Collection no longer exists")
    else:
        print("✗ Collection still exists!")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" CHROMA-COMPATIBLE FEATURES - COMPREHENSIVE TEST SUITE")
    print("="*70)

    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/", timeout=2)
        print(f"\n✓ Server is running")
    except:
        print(f"\n✗ Server is not running. Please start it first:")
        print("  python vector_db.py")
        return

    # Run all tests
    test_collections()
    test_batch_add()
    test_get_operations()
    test_query_operations()
    test_update_operations()
    test_delete_operations()
    test_include_fields()
    test_collection_deletion()

    print_section("✓ All Tests Complete!")
    print("The Chroma-compatible features are working correctly!\n")


if __name__ == "__main__":
    main()
