from langchain.schema import Document
from PdfFilesHybridExtractor import PdfFilesHybridExtractor

class ArabicPDFLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.extractor = PdfFilesHybridExtractor()  # Use hybrid approach

    def load(self) -> list[Document]:
        text = self.extractor.extract_text(self.file_path)
        return [Document(page_content=text)]