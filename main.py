from controller.generator.RAGPipeline import RAGPipeline
from .controller.Logger import setup_logger
import json

logger = setup_logger()
logger.info("Starting application...")

pipeline = RAGPipeline("example.pdf", logger)
questions = pipeline.run("query")
with open('sample output.json', 'w', encoding='utf-8') as f:
    json.dump(questions, f, ensure_ascii=False, indent=4)
