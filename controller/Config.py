# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    HF_TOKEN = os.getenv("HF_ACCESS_TOKEN")

    
    # Model configuration
    CHATTING_MODEL_NAME = os.getenv("CHATTING_MODEL_NAME")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

    