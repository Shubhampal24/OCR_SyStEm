import traceback
import sys
from PIL import Image
import numpy as np

from src.image_processing import ImageProcessor
from src.ocr_engine import OCREngine

try:
    print("Loading image processor...")
    proc = ImageProcessor()
    
    print("Loading OCR Engine...")
    ocr = OCREngine()
    
    print("Opening sample image...")
    # Load the uploaded image
    img = Image.open('C:/Users/DELL/.gemini/antigravity-ide/brain/91315416-784c-444c-bd32-4f47de17c1ac/uploaded_media_1782369286715.png')
    
    print("Processing image...")
    arr = proc.process_image(img)
    
    print("Extracting text...")
    res = ocr.extract_text(arr)
    
    print("Success:", res)
    
except Exception as e:
    print("ERROR CAUGHT!")
    traceback.print_exc()
