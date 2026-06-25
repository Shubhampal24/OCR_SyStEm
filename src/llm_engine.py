import torch
from transformers import pipeline
import json
import re
from src.core.logger import get_logger
from src.core.config import settings
from src.core.exceptions import LLMExtractionError

logger = get_logger(__name__)

class LLMEngine:
    """
    Industrial LLM Wrapper using Hugging Face Transformers.
    This class loads an open-source model, uses Prompt Engineering to 
    correct OCR mistakes, and forces the model to output structured JSON data.
    """
    def __init__(self):
        logger.info(f"Loading Hugging Face LLM: {settings.LLM_MODEL_NAME}")
        try:
            # Auto-detect if the user has an NVIDIA GPU (CUDA)
            device = 0 if torch.cuda.is_available() else -1
            if device == 0:
                logger.info("CUDA GPU detected. Loading model on GPU for fast inference.")
            else:
                logger.warning("No GPU detected. Loading model on CPU. Inference may take a few minutes per document.")

            # Load the model via the high-level Pipeline API
            self.generator = pipeline(
                "text-generation", 
                model=settings.LLM_MODEL_NAME, 
                device=device,
                # Use float16 on GPU to save VRAM, float32 on CPU
                torch_dtype=torch.float16 if device == 0 else torch.float32,
            )
            logger.info("LLM loaded successfully.")
            
        except Exception as e:
            logger.error(f"Failed to load LLM: {str(e)}")
            raise LLMExtractionError(f"Model loading failed: {str(e)}")

    def extract_information(self, ocr_text: str) -> dict:
        """
        Takes raw OCR text, builds a strict prompt, and generates JSON.
        """
        logger.info("Starting LLM information extraction and classification.")
        
        prompt = self._build_prompt(ocr_text)
        
        try:
            # Generate response
            # temperature=0.1 makes the model highly factual and less creative (perfect for data extraction)
            response = self.generator(
                prompt, 
                max_new_tokens=512, 
                return_full_text=False,
                temperature=0.1, 
                do_sample=True
            )
            
            generated_text = response[0]['generated_text']
            logger.debug(f"Raw LLM Output:\n{generated_text}")
            
            # Parse the JSON string into a Python dictionary
            json_data = self._parse_json(generated_text)
            logger.info("LLM extraction and JSON parsing successful.")
            return json_data

        except Exception as e:
            logger.error(f"LLM extraction failed: {str(e)}")
            raise LLMExtractionError(f"Failed to extract information: {str(e)}")

    def _build_prompt(self, ocr_text: str) -> str:
        """
        Industrial Prompt Engineering.
        We establish a strict persona and explicitly define the required output.
        """
        system_prompt = (
            "You are an expert data extraction AI. "
            "Your job is to read messy OCR text from a document and extract key information into a strict JSON format. "
            "Rule 1: Always include a 'document_type' key classifying the document strictly into one of these: 'Invoice', 'Receipt', 'Resume', 'Aadhaar Card', 'PAN Card', 'Driving License', 'Passport', 'Bank Statement', 'Other'. "
            "Rule 2: Always include a 'corrected_text' key containing a grammatically fixed version of the raw OCR text. "
            "Rule 3: Dynamically extract ALL relevant fields based on the document type (e.g., 'name', 'date', 'total_amount', 'invoice_number', 'gst_number', 'pan_number', 'aadhaar_number'). "
            "CRITICAL RULE: If the OCR text is garbage or empty, do NOT make up fake data like 'John Doe'. Output an empty JSON {} if no real data is found. "
            "Do NOT include any explanations, introductory text, or markdown formatting like ```json. Output ONLY the raw JSON dictionary."
        )
        
        user_prompt = f"OCR TEXT:\n{ocr_text}\n\nEXTRACTED JSON:"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            return self.generator.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}. Falling back to basic prompt.")
            return f"{system_prompt}\n\n{user_prompt}"

    def _parse_json(self, text: str) -> dict:
        """
        Fallback parser (Edge Case Handling).
        LLMs frequently disobey instructions and wrap JSON in markdown blocks like ```json ... ```.
        This uses Regular Expressions (Regex) to safely hunt down and extract the actual dictionary.
        """
        try:
            # Best case scenario: The LLM listened and output clean JSON
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Direct JSON parsing failed. LLM likely included markdown. Attempting regex extraction.")
            
            # Search for the first { and the last } in the string
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError as e:
                    raise LLMExtractionError(f"Regex extracted text is still not valid JSON: {str(e)}")
            else:
                raise LLMExtractionError("No JSON structure could be found in the LLM output.")

    def answer_question(self, question: str, context: str) -> str:
        """
        Takes the user's question and the raw OCR text as context and answers the question.
        """
        logger.info(f"Answering user query: {question}")
        system_prompt = "You are a helpful assistant reading a document."
        user_prompt = f"DOCUMENT CONTEXT:\n{context}\n\nUSER QUESTION: {question}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            prompt = self.generator.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = f"{system_prompt}\n\n{user_prompt}\n\nANSWER:"
            
        try:
            response = self.generator(
                prompt,
                max_new_tokens=256,
                return_full_text=False,
                temperature=0.3,
                do_sample=True
            )
            return response[0]['generated_text'].strip()
        except Exception as e:
            logger.error(f"Chatbot failed: {str(e)}")
            return "I am sorry, I am having trouble answering that question right now."
