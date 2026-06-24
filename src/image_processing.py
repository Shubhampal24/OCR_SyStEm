import cv2
import numpy as np
from PIL import Image
from src.core.logger import get_logger
from src.core.exceptions import ImageProcessingError

logger = get_logger(__name__)

class ImageProcessor:
    """
    Industrial-grade Image Preprocessing Pipeline.
    Real-world document photos are messy. This class applies computer vision 
    techniques (OpenCV) to clean, normalize, and prepare images so that the 
    OCR engine can read them with maximum accuracy.
    """
    def __init__(self):
        logger.debug("Initializing ImageProcessor pipeline.")
        
    def process_image(self, pil_image: Image.Image) -> np.ndarray:
        """
        Main pipeline: takes a raw uploaded image and returns a cleaned numpy array.
        """
        try:
            logger.info("Starting image preprocessing pipeline.")
            
            # Convert PIL image (which Streamlit uses) to OpenCV format (numpy array)
            cv_image = np.array(pil_image)
            
            # Convert RGB to BGR (OpenCV standard format)
            if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                
            # Step 1: Grayscale
            gray = self._to_grayscale(cv_image)
            
            # Step 2: Noise Removal (Smoothing)
            denoised = self._remove_noise(gray)
            
            # Step 3: Adaptive Thresholding (Binarization)
            thresh = self._apply_threshold(denoised)
            
            logger.info("Image preprocessing completed successfully.")
            return thresh
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise ImageProcessingError(f"Failed to process image: {str(e)}")

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Converts the image to black and white, dropping unnecessary color data."""
        logger.debug("Applying grayscale conversion.")
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    def _remove_noise(self, gray_image: np.ndarray) -> np.ndarray:
        """Applies a Gaussian Blur to remove tiny speckles (noise) from poor cameras."""
        logger.debug("Applying noise removal (Gaussian Blur).")
        return cv2.GaussianBlur(gray_image, (5, 5), 0)
        
    def _apply_threshold(self, denoised_image: np.ndarray) -> np.ndarray:
        """
        Uses Adaptive Thresholding. Unlike simple thresholding, this calculates 
        different thresholds for different regions of the image. This is CRITICAL 
        for photos where a shadow falls across half the document.
        """
        logger.debug("Applying adaptive thresholding.")
        return cv2.adaptiveThreshold(
            denoised_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
