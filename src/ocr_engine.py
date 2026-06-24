import easyocr
import numpy as np
from src.core.logger import get_logger
from src.core.config import settings
from src.core.exceptions import OCRExtractionError

logger = get_logger(__name__)

class OCREngine:
    """
    Industrial OCR Engine Wrapper.
    Uses EasyOCR to extract text from preprocessed images. 
    Includes confidence filtering to drop garbage characters.
    """
    def __init__(self):
        logger.info(f"Initializing EasyOCR Engine with language: '{settings.OCR_LANGUAGE}'")
        try:
            # gpu=True attempts to use NVIDIA CUDA. If no GPU is found, it safely falls back to CPU.
            self.reader = easyocr.Reader([settings.OCR_LANGUAGE], gpu=True)
            logger.info("EasyOCR Engine initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {str(e)}")
            raise OCRExtractionError(f"Initialization failed: {str(e)}")

    def extract_text(self, preprocessed_image: np.ndarray, confidence_threshold: float = 0.25) -> str:
        """
        Takes a cleaned numpy image array and returns a single combined string of text.
        Filters out text with a confidence score lower than the threshold.
        """
        try:
            logger.info("Starting text extraction from image.")
            
            # readtext() returns a list of tuples: (bounding_box, text, confidence_score)
            results = self.reader.readtext(preprocessed_image)
            
            extracted_blocks = []
            for (bbox, text, confidence) in results:
                # Edge Case Handling: Filter out highly uncertain reads (e.g., dust interpreted as commas)
                if confidence >= confidence_threshold:
                    extracted_blocks.append(text)
                else:
                    logger.debug(f"Dropped low-confidence text: '{text}' (Score: {confidence:.2f})")
            
            # Combine all valid text blocks into a single string document
            full_text = " ".join(extracted_blocks)
            
            if not full_text.strip():
                logger.warning("OCR completed, but no high-confidence text was found in the image.")
            else:
                logger.info(f"Extraction complete. Aggregated {len(extracted_blocks)} text blocks.")
            
            return full_text
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {str(e)}")
            raise OCRExtractionError(f"Extraction failed: {str(e)}")
