import fitz
from .PdfFilesExtractor import TextExtractor
import pytesseract
from PIL import Image

class PdfFilesHybridExtractor(TextExtractor):
    def extract_text(self, file_path: str) -> str:
        pages = []
        with fitz.open(file_path) as doc:
            for page in doc:
                # 1) Extract text once
                page_text = page.get_text().strip()
                
                # 2) If very little text, then do OCR
                if self._needs_ocr(page_text):
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = pytesseract.image_to_string(img, lang='ara').strip()
                    # Only replace if OCR gave more content
                    if len(ocr_text) > len(page_text):
                        page_text = ocr_text
                
                pages.append(page_text)
        
        # 3) Join pages with a newline
        return "\n".join(pages).strip()