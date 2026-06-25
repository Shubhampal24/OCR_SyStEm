import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from src.core.logger import get_logger
from src.core.exceptions import ImageProcessingError

logger = get_logger(__name__)

# Maximum dimension (px) for an image before we resize it down.
# Oversized images slow down OCR without improving accuracy.
MAX_DIMENSION = 2000

class ImageProcessor:
    """
    Lightweight, safe Image Preprocessing Pipeline.
    
    Philosophy: Do NOT destroy information. Every operation here is designed
    to enhance legibility (removing camera noise, improving contrast) without 
    risking text deletion (no aggressive thresholding or manual deskewing).
    
    RapidOCR's internal ONNX models handle text detection and orientation 
    correction natively — we just need to provide a clean, well-lit image.
    """
    def __init__(self):
        logger.info("ImageProcessor initialized (Lightweight Mode).")

    def process_image(self, pil_image: Image.Image) -> np.ndarray:
        """
        Main pipeline: Takes a raw uploaded PIL image, cleans it up,
        and returns a numpy array ready for the OCR engine.
        
        Cases handled:
        - RGBA/palette images (e.g., PNG with transparency): converted to RGB
        - Oversized images: resized to MAX_DIMENSION to prevent OCR timeout
        - Dark/underexposed images: contrast-enhanced with CLAHE
        - Noisy camera images: light Gaussian denoising
        """
        try:
            logger.info("Starting image preprocessing pipeline.")
            
            # --- Step 0: Normalize image mode ---
            # Convert RGBA (PNG with alpha channel) to RGB
            if pil_image.mode in ("RGBA", "P", "LA"):
                logger.debug(f"Converting image from {pil_image.mode} to RGB.")
                pil_image = pil_image.convert("RGB")
            elif pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            
            # --- Step 1: Resize if oversized ---
            pil_image = self._normalize_size(pil_image)
            
            # --- Step 2: Convert PIL to OpenCV numpy array ---
            cv_image = np.array(pil_image)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            
            # --- Step 3: Convert to Grayscale for analysis ---
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # --- Step 4: Check brightness and enhance contrast if needed ---
            mean_brightness = np.mean(gray)
            logger.debug(f"Image mean brightness: {mean_brightness:.1f}/255")
            
            # Use CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # This is gentle and locally adaptive - it won't wash out bright text
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # --- Step 5: Light Gaussian Denoising (remove camera speckle) ---
            # Kernel size 3x3 is gentle - doesn't blur text strokes
            denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            logger.info("Image preprocessing completed successfully.")
            return denoised
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise ImageProcessingError(f"Failed to process image: {str(e)}")

    def _normalize_size(self, pil_image: Image.Image) -> Image.Image:
        """
        Resizes oversized images while maintaining aspect ratio.
        Prevents OCR from timing out on 4K phone camera photos.
        """
        w, h = pil_image.size
        if max(w, h) > MAX_DIMENSION:
            ratio = MAX_DIMENSION / max(w, h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            logger.debug(f"Resizing image from {w}x{h} to {new_w}x{new_h}.")
            return pil_image.resize((new_w, new_h), Image.LANCZOS)
        return pil_image
