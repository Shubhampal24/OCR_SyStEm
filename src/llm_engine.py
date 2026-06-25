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
    Upgraded for v2.0: 
    - Zero temperature (Deterministic output)
    - Few-Shot Prompting (Anti-hallucination)
    - Regex fallback parser
    - Conversational History for Chatbot
    """
    def __init__(self):
        logger.info(f"Loading Hugging Face LLM: {settings.LLM_MODEL_NAME}")
        try:
            device = 0 if torch.cuda.is_available() else -1
            if device == 0:
                logger.info("CUDA GPU detected. Loading model on GPU.")
            else:
                logger.warning("No GPU detected. Loading model on CPU.")

            self.generator = pipeline(
                "text-generation", 
                model=settings.LLM_MODEL_NAME, 
                device=device,
                torch_dtype=torch.float16 if device == 0 else torch.float32,
            )
            logger.info("LLM loaded successfully.")
            
        except Exception as e:
            logger.error(f"Failed to load LLM: {str(e)}")
            raise LLMExtractionError(f"Model loading failed: {str(e)}")

    def extract_information(self, ocr_text: str) -> dict:
        """
        Takes raw OCR text, builds a strict few-shot prompt, and generates JSON.
        """
        logger.info("Starting LLM information extraction and classification.")
        
        # Anti-Hallucination Guard: If OCR text is too short, don't even ask the LLM
        if len(ocr_text.strip()) < 10:
            logger.warning("OCR text is too short. Bypassing LLM to prevent hallucination.")
            return {
                "document_type": "Unreadable",
                "corrected_text": ocr_text,
                "error": "Image did not contain enough readable text for extraction."
            }
            
        prompt = self._build_extraction_prompt(ocr_text)
        
        try:
            # temperature=0.01 makes the model highly deterministic
            response = self.generator(
                prompt, 
                max_new_tokens=512, 
                return_full_text=False,
                temperature=0.01, 
                do_sample=True
            )
            
            generated_text = response[0]['generated_text']
            logger.debug(f"Raw LLM Output:\n{generated_text}")
            
            json_data = self._parse_json(generated_text)
            logger.info("LLM extraction and JSON parsing successful.")
            return json_data

        except Exception as e:
            logger.error(f"LLM extraction failed: {str(e)}")
            raise LLMExtractionError(f"Failed to extract information: {str(e)}")

    def _build_extraction_prompt(self, ocr_text: str) -> str:
        """
        Few-Shot Prompt Engineering.
        We provide examples of EXACTLY what we want so the LLM doesn't guess.
        """
        system_prompt = (
            "You are an expert data extraction AI. "
            "Your job is to read messy OCR text from a document and extract key information into a strict JSON format. "
            "Rule 1: Always include a 'document_type' key. Valid options: 'Invoice', 'Receipt', 'Aadhaar Card', 'PAN Card', 'Passport', 'Other'. "
            "Rule 2: Always include a 'corrected_text' key fixing obvious OCR typos. "
            "Rule 3: Dynamically extract fields (e.g., 'name', 'date', 'total_amount', 'pan_number'). "
            "Rule 4: Output ONLY a valid JSON dictionary. Do NOT include markdown blocks like ```json."
        )
        
        # Example 1: PAN Card
        ex1_user = "OCR TEXT:\nINCOME TAX DEPARTMENT GOVT OF INDIA\nJOH N DOE\n01/01/1990\nABCDE1234F"
        ex1_asst = '{"document_type": "PAN Card", "corrected_text": "INCOME TAX DEPARTMENT GOVT OF INDIA JOHN DOE 01/01/1990 ABCDE1234F", "name": "John Doe", "date": "01/01/1990", "pan_number": "ABCDE1234F"}'
        
        # Actual User Query
        user_prompt = f"OCR TEXT:\n{ocr_text}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ex1_user},
            {"role": "assistant", "content": ex1_asst},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            return self.generator.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback if tokenizer doesn't support chat templates
            return f"{system_prompt}\n\nExample:\n{ex1_user}\nOutput:\n{ex1_asst}\n\nNow do this:\n{user_prompt}\nOutput:\n"

    def _parse_json(self, text: str) -> dict:
        """
        Robust Regex parsing to extract JSON even if the LLM wraps it in markdown
        or adds conversational garbage at the end.
        """
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Direct parsing failed. Attempting robust Regex extraction.")
            
            # Attempt 1: Look for markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
                    
            # Attempt 2: Non-greedy search for the first valid JSON block
            # This ignores trailing conversation like "Sure! Here is the JSON: {}"
            match_non_greedy = re.search(r'\{.*?\}', text, re.DOTALL)
            if match_non_greedy:
                try:
                    return json.loads(match_non_greedy.group(0))
                except json.JSONDecodeError as e:
                    raise LLMExtractionError(f"Regex extracted text is not valid JSON: {str(e)}")
            
            raise LLMExtractionError("No JSON structure could be found in the LLM output.")

    def answer_question_with_history(self, chat_history: list, context: str) -> str:
        """
        Multi-turn Chatbot Engine.
        Accepts the full chat history array to maintain conversational memory.
        chat_history format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        """
        system_prompt = (
            "You are a helpful AI assistant analyzing a document for a user. "
            "Answer questions based ONLY on the provided DOCUMENT CONTEXT. "
            "If the answer is not in the context, politely say so."
        )
        
        # Inject the OCR context into the first system message
        messages = [
            {"role": "system", "content": f"{system_prompt}\n\nDOCUMENT CONTEXT:\n{context}"}
        ]
        
        # Append all historical messages
        messages.extend(chat_history)
        
        try:
            prompt = self.generator.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback
            prompt = f"{system_prompt}\n\nDOCUMENT CONTEXT:\n{context}\n\n"
            for msg in chat_history:
                prompt += f"{msg['role'].upper()}: {msg['content']}\n"
            prompt += "ASSISTANT:"
            
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
            return "I'm sorry, I'm having trouble analyzing the document right now."
