"""
Custom Exception definitions.
In an industrial application, we don't just use standard generic 'Exception'.
We create specific error types so the application knows exactly what failed 
(e.g., did the OCR fail, or did the LLM fail?) and can recover gracefully.
"""

class ImageProcessingError(Exception):
    """Raised when OpenCV fails to process an image (e.g., corrupted file)."""
    pass

class OCRExtractionError(Exception):
    """Raised when EasyOCR fails to read text from an image."""
    pass

class LLMExtractionError(Exception):
    """Raised when the Hugging Face model fails or hallucinates invalid JSON."""
    pass

class DataValidationError(Exception):
    """Raised when the extracted data fails Pydantic business logic rules."""
    pass
