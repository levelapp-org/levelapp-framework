""" levelapp/setup/download_models.py"""
import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

from levelapp.aspects.logger import logger

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".models_cache")


def run():
    logger.info(f"[Setup] Pre-downloading baseline models ...")

    os.makedirs(CACHE_DIR, exist_ok=True)

    # Ensure all libraries use our cache dir
    os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
    os.environ["HF_HOME"] = CACHE_DIR
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = CACHE_DIR

    # RAG profiles required models
    SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    AutoTokenizer.from_pretrained("google/flan-t5-base")
    AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    logger.info(f"[Setup] Baseline model installation completed.")
