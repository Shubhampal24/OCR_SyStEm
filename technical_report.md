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

## 4. LLM Integration & Information Extraction Approach
*(To be updated with details on the chosen Hugging Face model, prompt engineering techniques, and JSON extraction strategy)*

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
