from PIL import Image
import fitz
import pytesseract
from BaseExtractor import TextExtractor

class PdfFilesImagesBasedExtractor(TextExtractor):
    def extract_text(self, file_path: str) -> str:
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text += pytesseract.image_to_string(img, lang='ara') + "\n"
        return text.strip()