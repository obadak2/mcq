import fitz
from BaseExtractor import TextExtractor


class PdfFilesTextBasedExtractor(TextExtractor):
    def extract_text(self, file_path: str) -> str:
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text.strip()