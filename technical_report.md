# Technical Report: Intelligent OCR System

## 1. Executive Summary
This report details the architectural decisions, workflows, and challenges faced during the development of an industrial-grade Intelligent OCR System. The system strictly adheres to the requirement of utilizing only open-source tools (EasyOCR, Hugging Face LLMs, OpenCV, Pydantic, Streamlit).

## 2. System Architecture & Workflow
*(To be updated as the system is built)*
The high-level workflow consists of Image Preprocessing -> OCR Text Extraction -> LLM Processing (Correction, Extraction, Classification) -> Data Validation -> UI Presentation.

## 3. Image Preprocessing & OCR Workflow
**Phase 2: Image Preprocessing Pipeline**
Real-world document inputs often suffer from shadows, blur, and noise. To maximize OCR accuracy, an `ImageProcessor` class was built using OpenCV (`cv2`). The pipeline applies three sequential transformations:
1.  **Grayscale Conversion:** Strips unnecessary color channels (`cv2.cvtColor`).
2.  **Noise Removal:** Applies Gaussian Blur (`cv2.GaussianBlur`) to eliminate high-frequency noise common in mobile camera photos.
3.  **Adaptive Thresholding:** Uses `cv2.adaptiveThreshold` (Gaussian C) to handle uneven illumination. This ensures text remains legible even if shadows fall across sections of the document.

**Phase 3: OCR Extraction Engine**
Once the image is preprocessed, it is passed to the `OCREngine` which wraps the `EasyOCR` library. EasyOCR was chosen over Tesseract due to its native Python/PyTorch support and robust out-of-the-box accuracy for multiple languages. 
- **Confidence Filtering:** To prevent garbage data (e.g., speckles of dust interpreted as commas) from polluting the LLM prompt, the engine implements a strict confidence threshold (default 0.25). Any text bounding box scoring below this threshold is dropped entirely.
- **Aggregation:** Valid text blocks are joined into a single coherent text string to be passed down the pipeline.

## 4. LLM Integration & Information Extraction Approach
**Phase 4: Intelligent LLM Engine**
To satisfy the requirement of using open-source LLMs without relying on external APIs, an `LLMEngine` class was developed utilizing the Hugging Face `transformers` pipeline. 
- **Prompt Engineering:** The engine constructs a rigid system prompt instructing the model to act as an expert data extractor. It commands the model to correct OCR spelling mistakes, classify the document type, and strictly output JSON. The `temperature` is set to 0.1 to maximize factual determinism and prevent creative hallucinations.
- **Hardware Agnosticism:** The engine dynamically checks for `torch.cuda.is_available()`. If an NVIDIA GPU is present, it loads the model in memory-efficient `float16` precision to maximize speed. If not, it safely falls back to CPU processing.
- **Resilient JSON Parsing:** LLMs are notorious for disobeying formatting constraints (e.g., wrapping JSON in markdown backticks). A fallback parser utilizing Regular Expressions (`re.search(r'\{.*\}')`) was implemented to salvage JSON data even if the LLM includes conversational filler text.

## 5. Validation Logic
*(To be updated with details on Pydantic schemas and regex logic used for data integrity)*

## 6. Challenges Faced & Solutions
*(This section will track our challenges step-by-step)*
- **Challenge 1:** Establishing an industrial-grade foundation without relying on quick, messy scripts.
  - **Solution:** Designed a granular, 7-phase implementation plan emphasizing configuration management, strict validation, and error handling from day one.
- **Challenge 2:** Ensuring the repository remains clean and secure during early development phases.
  - **Solution:** Implemented a robust `.gitignore` to prevent virtual environments, local caches, and `.env` secrets from being accidentally committed. Created an `.env.example` to safely communicate required environment variables.
- **Challenge 3:** Bridging environment configurations to Python without hardcoding, while ensuring strict error tracking.
  - **Solution:** Developed a `src/core/` module. Created `config.py` using `dotenv` to load settings, `logger.py` to replace standard print statements with industrial logging, and `exceptions.py` to enforce strict custom error boundaries (e.g., `OCRExtractionError`, `LLMExtractionError`).
