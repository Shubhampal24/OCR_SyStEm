# Intelligent OCR & Document Understanding System

An industrial-grade Optical Character Recognition (OCR) pipeline that extracts, understands, and validates document data using 100% open-source, privacy-first Local LLMs and Computer Vision models.

![Streamlit UI Concept](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Vision-PaddleOCR_ONNX-0073bc?style=for-the-badge)
![LLM](https://img.shields.io/badge/LLM-Phi_3.5_Mini-00A4EF?style=for-the-badge&logo=Microsoft)

## 📌 Project Overview
This system is designed to ingest document images (Invoices, PAN Cards, Aadhaar Cards, etc.) and PDFs, and output highly structured JSON data. It uses **RapidOCR** (PaddleOCR ONNX) for vision detection and **Microsoft Phi-3.5** (via Hugging Face Transformers) for text correction, semantic extraction, and conversational querying. 

**Zero Cloud APIs are used. 100% of the inference happens locally.**

## 🚀 Features
- **Non-Destructive Image Preprocessing**: Automatic resizing, CLAHE contrast enhancement, and noise reduction.
- **Robust OCR Extraction**: Detects and recognizes dense text, including automatic orientation correction.
- **LLM Contextual Extraction**: Few-shot prompted SLMs correct OCR typos and format data into strict JSON.
- **Business Logic Validation**: Pydantic models automatically validate Indian phone numbers, PAN cards, Aadhaar cards, and GST formats.
- **Persistent Chatbot**: Multi-turn RAG chatbot memory allows you to ask conversational questions about the uploaded document.
- **PDF Support**: Automatically rips the first page of PDF documents for OCR.

## 🛠️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/OCR_System.git
cd OCR_System
```

**2. Create a Virtual Environment**
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Environment Setup**
Create a `.env` file in the root directory based on the provided `.env.example`:
```env
LLM_MODEL_NAME="microsoft/Phi-3.5-mini-instruct"
OCR_LANGUAGE="en"
DEBUG_MODE=True
```

## 🏃 Execution Steps
Start the Streamlit web application:
```bash
streamlit run app.py
```
1. Open your browser to `http://localhost:8501`.
2. Wait for the engine initialization sequence to complete. (Note: The first time you run this, Hugging Face will download the open-source LLM weights to your local machine).
3. Upload an Image or PDF.
4. Click **Run Extraction Sequence**.

## 🏗️ Architecture
*Please see [ARCHITECTURE.md](ARCHITECTURE.md) for the full system pipeline diagram.*

## 📄 Technical Report
*Please see [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for a deep dive into the engineering decisions, OCR workflow, and LLM anti-hallucination techniques.*
