# Intelligent OCR System Using Open-Source LLM

## Overview
This project is an industrial-grade Intelligent Document Processing (IDP) application. It accepts document images or PDFs, performs optical character recognition (OCR), and leverages open-source Large Language Models (LLMs) to intelligently extract, classify, and validate structured information without relying on paid APIs.

## Key Features
* **Robust Image Preprocessing:** Cleans and normalizes noisy images before OCR.
* **Open-Source OCR Extraction:** Utilizes EasyOCR for high-accuracy text extraction.
* **Intelligent LLM Parsing:** Uses Hugging Face open-source models to correct OCR errors and extract key-value pairs.
* **Document Classification:** Automatically categorizes uploaded documents (e.g., Invoices, Receipts, ID Cards).
* **Strict Data Validation:** Validates extracted fields (Email, Phone, PAN, Aadhaar) against predefined rules.
* **Interactive Dashboard:** Built with Streamlit for a seamless user experience.

## Installation & Setup
*(This section will be updated as we finalize the setup process)*

1. Clone the repository.
2. Set up the virtual environment.
3. Install dependencies from `requirements.txt`.
4. Run the Streamlit app.

## Project Structure
```text
OCR_System/
├── src/
│   ├── core/
│   │   ├── config.py          # Links .env to Python variables
│   │   ├── exceptions.py      # Custom error handling
│   │   └── logger.py          # Standardized event logging
│   ├── image_processing.py    # OpenCV preprocessing pipeline
│   └── ocr_engine.py          # EasyOCR extraction wrapper
├── .env.example       # Template for environment variables
├── .gitignore         # Ignores venv, caches, and secrets
├── requirements.txt   # Core Python dependencies
├── README.md          # Project documentation
└── technical_report.md# Detailed technical interview report
```
## Usage
*(Instructions on how to upload and interact with the system will be added here)*
