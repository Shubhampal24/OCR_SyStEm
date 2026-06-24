import os
from dotenv import load_dotenv

# Load environment variables from the .env file into Python
load_dotenv()

class Config:
    """
    Centralized configuration class.
    This reads from the .env file. If a value isn't found in .env, it uses the default provided here.
    """
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "en")
    
    # Convert string "True"/"False" from .env into actual Python boolean
    DEBUG_MODE = os.getenv("DEBUG_MODE", "True").lower() in ("true", "1", "t", "yes")

# We create a single instance of this config to be imported across the whole app
settings = Config()
