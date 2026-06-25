import numpy as np
from rapidocr_onnxruntime import RapidOCR
from src.core.logger import get_logger
from src.core.exceptions import OCRExtractionError

logger = get_logger(__name__)

# Minimum confidence score for accepting a text block.
# RapidOCR returns scores from 0.0 to 1.0.
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

class OCREngine:
    """
    Industrial OCR Engine using RapidOCR (ONNX Runtime).
    
    Why RapidOCR?
    - Built on PaddleOCR's superior ONNX models but works on any Python version
    - Includes built-in text detection, recognition, and angle classification
    - Significantly outperforms EasyOCR on dense text (invoices, ID cards)
    - Pure CPU inference via ONNX - no GPU required, no PaddlePaddle dependency
    
    The engine runs 3 sub-models internally:
    1. Text Detection (DB++) - finds where text is on the image
    2. Angle Classification - determines if text is upside-down (0° or 180°)
    3. Text Recognition (CRNN) - reads each detected text region
    """
    def __init__(self):
        logger.info("Initializing RapidOCR Engine...")
        try:
            # use_angle_cls=True enables automatic orientation detection
            # This replaces our old buggy OpenCV deskewing code completely
            self.engine = RapidOCR()
            logger.info("RapidOCR Engine initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize RapidOCR: {str(e)}")
            raise OCRExtractionError(f"RapidOCR initialization failed: {str(e)}")

    def extract_text(self, preprocessed_image: np.ndarray, 
                     confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> str:
        """
        Runs the full OCR pipeline on a preprocessed numpy image array.
        
        Returns a single string of all high-confidence text blocks,
        joined in reading order (top-to-bottom, left-to-right).
        
        Edge Cases Handled:
        - Empty image / all white: returns empty string without crashing
        - Very low confidence results: filtered out to reduce garbage
        - None results from engine: safely returns empty string
        - Single character noise (e.g., random dots): filtered out
        """
        try:
            logger.info("Starting text extraction from image.")
            
            result, elapse = self.engine(preprocessed_image)
            
            # Edge case: Engine returned None (completely blank image)
            if result is None:
                logger.warning("RapidOCR returned no results. Image may be blank or unreadable.")
                return ""
            
            extracted_blocks = []
            total_dropped = 0
            
            for item in result:
                # RapidOCR result format: [bounding_box, text, confidence_score]
                if len(item) < 3:
                    continue
                    
                bbox, text, confidence_raw = item[0], item[1], item[2]
                
                # RapidOCR might return confidence as a string in some builds, cast to float
                try:
                    confidence = float(confidence_raw)
                except (ValueError, TypeError):
                    confidence = 0.0
                
                # Edge Case: Filter single-character noise (random dots, specs)
                if len(text.strip()) <= 1:
                    logger.debug(f"Dropped single-char noise: '{text}' (Score: {confidence:.2f})")
                    total_dropped += 1
                    continue
                
                # Edge Case: Filter low-confidence reads
                if confidence < confidence_threshold:
                    logger.debug(f"Dropped low-confidence text: '{text}' (Score: {confidence:.2f})")
                    total_dropped += 1
                    continue
                    
                extracted_blocks.append(text.strip())
            
            full_text = " ".join(extracted_blocks)
            
            if not full_text.strip():
                logger.warning(
                    f"OCR completed but no high-confidence text found. "
                    f"Dropped {total_dropped} low-quality blocks. "
                    f"Try lowering the confidence threshold."
                )
            else:
                logger.info(
                    f"Extraction complete. Kept {len(extracted_blocks)} blocks, "
                    f"dropped {total_dropped} low-quality blocks. "
                    f"Inference took {sum(elapse):.2f}s"
                )
            
            return full_text
            
        except Exception as e:
            logger.error(f"RapidOCR extraction failed: {str(e)}")
            raise OCRExtractionError(f"Extraction failed: {str(e)}")
