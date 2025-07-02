from ..data.DataLoader import ArabicPDFLoader
from ..data.Chunking import Chunker
from ...model.VectorStoreManager import VectorStoreManager
from QuestionGenerator import ModelQuestionGenerator
import time
import logging
from Config import Config

class RAGPipeline:
    def __init__(self, file_path: str, logger:logging):
        self.logger = logger
        self.file_path = file_path
        self.loader = ArabicPDFLoader(file_path)
        self.chunker = Chunker()
        self.vector_store = VectorStoreManager(model_name=Config.EMBEDDING_MODEL_NAME, logger=self.logger)
        self.generator = ModelQuestionGenerator(model_name=Config.CHATTING_MODEL_NAME, logger=self.logger)

    def create_question_prompt(self, context: str, query: str = None) -> str:
        """Construct the prompt for question generation"""
        base_prompt = """
        Generate 10 questions of type Multiple Choice with 4 choices from the given text only in Arabic.
        Questions should be distributed as:
        1- Difficulty: (easy:20%, average:50%, difficult:30%)
        2- Bloom Taxonomy: (Remember:40%, Evaluate:30%, Apply:30%)
        Output STRICTLY as JSON array with each question having:
        (question_text, question_answers, correct_answer, difficulty, bloom_taxonomy).
        """
        
        if query:
            prompt = f"{base_prompt}\nFocus on aspects related to: {query}\n\nText: {context}"
        else:
            prompt = f"{base_prompt}\n\nText: {context}"
            
        return prompt.strip()


    def run(self, query: str) -> list:
        t0 = time.time()
        docs = self.loader.load()
        self.logger.info(f"Loading took {time.time() - t0:.2f}s")

        t1 = time.time()
        chunks = self.chunker.chunk(docs)
        self.logger.info(f"Chunking took {time.time() - t1:.2f}s")

        t2 = time.time()
        self.vector_store.create_store(chunks)
        self.logger.info(f"Embedding & FAISS build took {time.time() - t2:.2f}s")

        t3 = time.time()
        retriever = self.vector_store.get_retriever()
        relevant_docs = retriever.get_relevant_documents(query or "")
        self.logger.info(f"Retrieval took {time.time() - t3:.2f}s")

        context = "\n".join(d.page_content for d in relevant_docs)
        prompt = self.create_question_prompt(context, query)

        t4 = time.time()
        questions = self.generator.generate_questions(prompt)
        self.logger.info(f"Generation took {time.time() - t4:.2f}s")

        self.logger.info(f"Total runtime: {time.time() - t0:.2f}s")
        return questions