#!/usr/bin/env python3
"""
Model Configuration Manager - Automatic environment detection and model selection.
Handles local development vs Streamlit Cloud deployment scenarios.
"""

import os
import platform
import sys
from pathlib import Path
from typing import Dict

import psutil

# ✅ ROBUST FIX: Always use absolute path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from logging_setup import setup_logging

logger = setup_logging()


class ModelConfig:
    """Configuration manager for environment-aware model selection."""

    # Model configurations
    LOCAL_MODELS = {
        "primary": "llama3.2:3b",  # Better quality for local dev
        "judge": "llama3.1:8b",  # Thorough evaluation locally
        "fallback": "llama3.2:1b",  # Backup if larger models fail
    }

    CLOUD_MODELS = {
        "primary": "gemma2:2b",  # Streamlit-compatible size
        "judge": "phi3:mini",  # Alternative small model for evaluation
        "fallback": "gemma2:2b",  # Same as primary for simplicity
    }

    # Environment detection thresholds
    MEMORY_THRESHOLD_GB = 4  # If less than 4GB RAM, assume cloud

    @staticmethod
    def detect_environment() -> str:
        """
        Detect the current environment (local, cloud, or testing).

        Returns:
            str: 'local', 'cloud', or 'testing'
        """
        # Check for Streamlit Cloud indicators
        streamlit_indicators = [
            "STREAMLIT_SHARING_MODE" in os.environ,
            "streamlit.io" in os.getenv("HOME", "").lower(),
            "streamlit" in os.getenv("USER", "").lower(),
            "app" in os.getenv("HOME", "").lower() and "streamlit" in str(Path.cwd()),
        ]

        if any(streamlit_indicators):
            logger.info("🌐 Detected Streamlit Cloud environment")
            return "cloud"

        # Check available system memory
        try:
            memory_gb = psutil.virtual_memory().total / (1024**3)
            if memory_gb < ModelConfig.MEMORY_THRESHOLD_GB:
                logger.info(
                    f"💾 Low memory detected ({memory_gb:.1f}GB) - using cloud config"
                )
                return "cloud"
        except Exception:
            logger.warning("⚠️ Could not detect memory - defaulting to cloud config")
            return "cloud"

        # Check for local development indicators
        local_indicators = [
            "localhost" in os.getenv("HOSTNAME", "").lower(),
            platform.system() in ["Windows", "Darwin"],  # Windows/Mac likely local
            "CONDA_DEFAULT_ENV" in os.environ,  # Conda environment
            Path.home().name in ["tcmk_", "user", "admin"],  # Common local usernames
        ]

        if any(local_indicators):
            logger.info("🏠 Detected local development environment")
            return "local"

        # Default to cloud for safety
        logger.info("🤷 Environment unclear - defaulting to cloud configuration")
        return "cloud"

    @staticmethod
    def get_model_config(environment: str = None) -> Dict[str, str]:
        """
        Get model configuration for the specified or detected environment.

        Args:
            environment (str, optional): Force specific environment. Defaults to auto-detect.

        Returns:
            Dict[str, str]: Model configuration dict with 'primary', 'judge', 'fallback' keys
        """
        if environment is None:
            environment = ModelConfig.detect_environment()

        if environment == "local":
            config = ModelConfig.LOCAL_MODELS.copy()
            logger.info(f"🔥 Using LOCAL models: {config}")
        else:  # cloud or unknown
            config = ModelConfig.CLOUD_MODELS.copy()
            logger.info(f"☁️ Using CLOUD models: {config}")

        return config

    @staticmethod
    def get_primary_model(environment: str = None) -> str:
        """Get the primary RAG model for the environment."""
        config = ModelConfig.get_model_config(environment)
        return config["primary"]

    @staticmethod
    def get_judge_model(environment: str = None) -> str:
        """Get the evaluation/judge model for the environment."""
        config = ModelConfig.get_model_config(environment)
        return config["judge"]

    @staticmethod
    def get_fallback_model(environment: str = None) -> str:
        """Get the fallback model for the environment."""
        config = ModelConfig.get_model_config(environment)
        return config["fallback"]

    @staticmethod
    def list_required_models(environment: str = None) -> list:
        """Get list of all models required for the environment."""
        config = ModelConfig.get_model_config(environment)
        # Remove duplicates while preserving order
        models = []
        for model in config.values():
            if model not in models:
                models.append(model)
        return models


def get_optimal_model_config() -> Dict[str, str]:
    """Convenience function to get optimal model configuration."""
    return ModelConfig.get_model_config()


def get_primary_model() -> str:
    """Convenience function to get primary model."""
    return ModelConfig.get_primary_model()


def get_judge_model() -> str:
    """Convenience function to get judge model."""
    return ModelConfig.get_judge_model()


def get_fallback_model() -> str:
    """Convenience function to get fallback model."""
    return ModelConfig.get_fallback_model()


if __name__ == "__main__":
    # Test the configuration
    print("🧪 MODEL CONFIGURATION TEST")
    print("=" * 40)

    env = ModelConfig.detect_environment()
    config = ModelConfig.get_model_config(env)

    print(f"Environment: {env}")
    print(f"Primary Model: {config['primary']}")
    print(f"Judge Model: {config['judge']}")
    print(f"Fallback Model: {config['fallback']}")
    print(f"Required Models: {ModelConfig.list_required_models(env)}")
