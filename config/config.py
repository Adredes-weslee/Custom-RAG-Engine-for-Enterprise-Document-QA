import os
from dotenv import load_dotenv

load_dotenv()

# Fetch credentials from environment variables
endpoint: str = os.getenv("ENDPOINT")
api_key: str = os.getenv("API_KEY")
model: str = os.getenv("MODEL")
api_version: str = os.getenv("API_VERSION")
model_version: str = os.getenv("MODEL_VERSION")