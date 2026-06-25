# Technical Report: Intelligent OCR & Document Understanding System

## 1. Introduction & Problem Statement
The objective of this project was to construct an intelligent Optical Character Recognition (OCR) pipeline capable of ingesting raw document images (invoices, ID cards, receipts), extracting their text, and converting that text into highly structured JSON payloads using exclusively open-source, local AI models. The use of paid APIs (OpenAI, Gemini) was strictly prohibited to guarantee a privacy-first, air-gapped solution.

## 2. OCR Workflow (Computer Vision Layer)

### Image Preprocessing
Before passing an image to the neural networks, we must ensure the data is as clean as possible. Destructive preprocessing methods (like strict binary adaptive thresholding) were avoided because they often wash out faint text on poorly lit photos. 
Instead, our `ImageProcessor` applies:
1. **Size Normalization:** Massively high-resolution images are scaled down to prevent memory timeouts.
2. **CLAHE (Contrast Limited Adaptive Histogram Equalization):** This algorithm locally enhances the contrast of text against its background without blowing out the highlights of the entire image.
3. **Gaussian Blur:** A mild 3x3 kernel blur is applied to remove camera sensor noise.

### RapidOCR (PaddleOCR ONNX)
For the extraction layer, we selected **RapidOCR** over EasyOCR and Tesseract. RapidOCR is a CPU-optimized ONNX runtime wrapper for Baidu’s state-of-the-art PaddleOCR models. 
It operates using three distinct neural networks:
1. **DB++ (Text Detection):** Accurately finds bounding boxes around text, heavily outperforming EasyOCR on dense documents like invoices.
2. **Angle Classification:** Replaces the need for manual OpenCV deskewing. It detects if the text is upside down (180 degrees) and corrects it internally.
3. **CRNN (Text Recognition):** Reads the actual characters inside the bounding boxes. We implemented a strict confidence threshold filter (`0.5`) to automatically drop hallucinated characters and camera noise.

## 3. LLM Integration & Information Extraction

Extracting text is only the first step; the text must be semantically understood. We implemented Hugging Face Transformers to run **Microsoft Phi-3.5-mini-instruct**, an incredibly capable 3.8-billion-parameter Small Language Model (SLM) running purely on CPU.

### Few-Shot Prompt Engineering
SLMs have a tendency to hallucinate formatting if not strictly guided. To ensure the output was perfect JSON, we utilized **Few-Shot Prompting**. We injected a perfect example of a PAN Card extraction directly into the system prompt. By showing the model exactly what a successful input and output looks like, we eliminated JSON structure hallucinations.

### Fallback Regex Parsing
Because SLMs are trained on markdown, they occasionally wrap their JSON output in markdown blocks (e.g., ` ```json {...} ``` `). We built a robust Regex parser inside `llm_engine.py` that strips conversational garbage and markdown artifacts, extracting only the raw dictionary.

### RAG Chatbot Integration
By utilizing Streamlit's `st.session_state`, we stored the user's conversational history. The extracted document text is injected as "DOCUMENT CONTEXT", and the chat history is appended to the prompt. This creates a multi-turn, Retrieval-Augmented Generation (RAG) chatbot that allows users to seamlessly interrogate their documents.

## 4. Validation Logic
AI outputs cannot be fully trusted in a production environment. We implemented a strict business-logic layer using **Pydantic**.
The `validation.py` script applies custom RegEx rules to the LLM's JSON output to verify format compliance for:
- **Indian PAN Cards:** (`^[A-Z]{5}[0-9]{4}[A-Z]{1}$`)
- **Aadhaar Numbers:** 12 numeric digits
- **GST Numbers:** Validates state codes and checksum placement
- **Phone & Date Formats**

To prevent system crashes when the AI hallucinates a bad date format, we implemented **Soft Validation**. If a rule is violated, Pydantic catches the error, prevents a Python crash, and elegantly displays a yellow warning box to the user on the Streamlit frontend.

## 5. Challenges Faced & Resolutions

**Challenge 1: Environment & Dependency Conflicts**
*Issue:* The standard `paddlepaddle` library lacks pre-compiled binaries for newer Python versions (3.14), resulting in C++ compiler errors during installation.
*Resolution:* Pivoted architecture to use `rapidocr-onnxruntime`. This bypassed the PaddlePaddle ecosystem entirely, running the exact same models natively via Microsoft's ONNX runtime, ensuring 100% cross-platform compatibility.

**Challenge 2: LLM Hallucination on Blank Images**
*Issue:* If a user uploads an image with just a logo and no text, the LLM attempts to fulfill its prompt by completely inventing fake names and invoice numbers out of thin air.
*Resolution:* Implemented an "Anti-Hallucination Guard" in the code. If RapidOCR returns fewer than 10 total characters of text, the system completely skips the LLM generation phase and instantly returns an "Unreadable Document" flag.

**Challenge 3: Streamlit File Lock Exceptions**
*Issue:* Streamlit's cache holds onto `.pyd` files (like OpenCV binaries), causing permissions errors (`[WinError 5] Access is denied`) when trying to update backend dependencies.
*Resolution:* Implemented manual task killing and cache-clearing mechanisms, ensuring the development environment remained stable during hot-reloads.
