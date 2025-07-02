from controller.generator.RAGPipeline import RAGPipeline
from .controller.Logger import setup_logger

logger = setup_logger()
logger.info("Starting application...")

pipeline = RAGPipeline("example.pdf", logger)
questions = pipeline.run("الذكاء الاصطناعي في التعليم")
print(questions)
