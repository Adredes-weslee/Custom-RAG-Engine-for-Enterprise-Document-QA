#!/usr/bin/env python3
"""
Model Setup Script - Downloads required Ollama models based on environment.
Automatically detects local vs cloud deployment and downloads appropriate models.
"""

import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from utils.model_config import ModelConfig


def check_ollama_installed() -> bool:
    """Check if Ollama is installed and accessible."""
    try:
        result = subprocess.run(
            ["ollama", "--version"], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_ollama_server() -> bool:
    """Check if Ollama server is running."""
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_installed_models() -> list:
    """Get list of currently installed Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")[1:]  # Skip header
            models = []
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    models.append(model_name)
            return models
        return []
    except Exception:
        return []


def pull_model(model_name: str) -> bool:
    """
    Pull a specific Ollama model.

    Args:
        model_name (str): Name of the model to pull

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"📥 Downloading {model_name}...")
    try:
        # Show progress by not capturing output
        result = subprocess.run(
            ["ollama", "pull", model_name], timeout=1800
        )  # 30 minute timeout
        if result.returncode == 0:
            print(f"✅ Successfully downloaded {model_name}")
            return True
        else:
            print(f"❌ Failed to download {model_name}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout while downloading {model_name}")
        return False
    except Exception as e:
        print(f"❌ Error downloading {model_name}: {e}")
        return False


def setup_models(environment: str = None, force: bool = False) -> bool:
    """
    Set up all required models for the environment.

    Args:
        environment (str, optional): Force specific environment. Defaults to auto-detect.
        force (bool): Force re-download even if models exist

    Returns:
        bool: True if all models successfully set up
    """
    print("🚀 OLLAMA MODEL SETUP")
    print("=" * 50)

    # Check Ollama installation
    if not check_ollama_installed():
        print("❌ Ollama is not installed or not in PATH")
        print("📋 Please install Ollama from: https://ollama.com/download")
        return False

    print("✅ Ollama is installed")

    # Check Ollama server
    if not check_ollama_server():
        print("❌ Ollama server is not running")
        print("📋 Please start Ollama server: ollama serve")
        return False

    print("✅ Ollama server is running")

    # Detect environment and get required models
    if environment is None:
        environment = ModelConfig.detect_environment()

    config = ModelConfig.get_model_config(environment)
    required_models = ModelConfig.list_required_models(environment)

    print(f"\n🎯 Environment: {environment.upper()}")
    print(f"📋 Required models: {required_models}")

    # Check existing models
    installed_models = get_installed_models()
    print(f"📦 Currently installed: {installed_models}")

    # Determine which models to download
    models_to_download = []
    for model in required_models:
        if force or model not in installed_models:
            models_to_download.append(model)
        else:
            print(f"✅ {model} already installed")

    if not models_to_download:
        print("\n🎉 All required models are already installed!")
        return True

    print(f"\n📥 Models to download: {models_to_download}")

    # Estimate download sizes
    model_sizes = {
        "llama3.2:1b": "1.3GB",
        "llama3.2:3b": "2.0GB",
        "llama3.1:8b": "4.7GB",
        "gemma2:2b": "1.6GB",
        "phi3:mini": "2.3GB",
    }

    total_size = sum(
        [float(model_sizes.get(model, "2.0GB")[:-2]) for model in models_to_download]
    )
    print(f"📊 Estimated total download: ~{total_size:.1f}GB")

    # Confirm download
    if not force:
        response = input("\n🤔 Proceed with download? (y/N): ").lower().strip()
        if response not in ["y", "yes"]:
            print("❌ Download cancelled")
            return False

    # Download models
    print(f"\n📥 Starting download of {len(models_to_download)} models...")
    success_count = 0

    for i, model in enumerate(models_to_download, 1):
        print(f"\n[{i}/{len(models_to_download)}] {model}")
        size = model_sizes.get(model, "Unknown size")
        print(f"📊 Estimated size: {size}")

        if pull_model(model):
            success_count += 1
        else:
            print(f"⚠️ Failed to download {model}")

    # Summary
    print("\n🏁 SETUP COMPLETE")
    print("=" * 30)
    print(
        f"✅ Successfully downloaded: {success_count}/{len(models_to_download)} models"
    )

    if success_count == len(models_to_download):
        print("🎉 All models downloaded successfully!")
        print("\n📋 You can now run:")
        print("   python src/main.py")
        print("   # OR")
        print("   streamlit run src/main.py")
        return True
    else:
        print("⚠️ Some models failed to download")
        print("🔧 You can retry with: python setup_models.py --force")
        return False


def main():
    """Main setup function with command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Set up Ollama models for RAG engine")
    parser.add_argument(
        "--environment",
        choices=["local", "cloud"],
        help="Force specific environment (auto-detect if not specified)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-download even if models exist"
    )
    parser.add_argument(
        "--list", action="store_true", help="List currently installed models"
    )

    args = parser.parse_args()

    if args.list:
        print("📦 INSTALLED OLLAMA MODELS")
        print("=" * 30)
        models = get_installed_models()
        if models:
            for model in models:
                print(f"✅ {model}")
        else:
            print("❌ No models installed")
        return

    success = setup_models(args.environment, args.force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
