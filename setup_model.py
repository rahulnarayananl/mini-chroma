"""
Setup script to download and convert all-MiniLM-L6-v2 to ONNX format.
Run this once before starting the vector database server.
"""

import subprocess
import sys
from pathlib import Path


def main():
    print("\n" + "="*70)
    print("Model Setup for Lightweight Vector Database")
    print("="*70 + "\n")

    model_dir = Path("./models")
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Installing optimum for ONNX export...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "optimum[onnxruntime]", "-q"
        ])
        print("✓ Optimum installed\n")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install optimum: {e}")
        return

    print("Step 2: Downloading and converting all-MiniLM-L6-v2 to ONNX...")
    print("This will take a few minutes...\n")

    try:
        subprocess.check_call([
            "optimum-cli", "export", "onnx",
            "--model", "sentence-transformers/all-MiniLM-L6-v2",
            str(model_dir)
        ])
        print("\n✓ Model successfully exported to ONNX format!\n")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed to export model: {e}")
        print("\nTry manually running:")
        print(f"  optimum-cli export onnx --model sentence-transformers/all-MiniLM-L6-v2 {model_dir}")
        return

    # Verify files exist
    model_path = model_dir / "model.onnx"
    tokenizer_path = model_dir / "tokenizer.json"

    if model_path.exists() and tokenizer_path.exists():
        print("✓ Verified model files:")
        print(f"  - {model_path} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"  - {tokenizer_path} ({tokenizer_path.stat().st_size / 1024:.1f} KB)")
        print("\n" + "="*70)
        print("Setup complete! You can now run: python vector_db.py")
        print("="*70 + "\n")
    else:
        print("✗ Model files not found after export. Please check for errors above.")


if __name__ == "__main__":
    main()
