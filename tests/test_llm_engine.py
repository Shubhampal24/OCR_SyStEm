import pytest
from unittest.mock import patch
from src.llm_engine import LLMEngine
from src.core.exceptions import LLMExtractionError

# We mock the Hugging Face 'pipeline' function so we don't load 3GB of weights during a unit test!
@patch('src.llm_engine.pipeline')
def test_parse_json_perfect_output(mock_pipeline):
    """Test when the LLM successfully obeys instructions and outputs pure JSON."""
    engine = LLMEngine() # Uses mocked pipeline
    text = '{"name": "test", "amount": 100}'
    result = engine._parse_json(text)
    assert result["name"] == "test"
    assert result["amount"] == 100

@patch('src.llm_engine.pipeline')
def test_parse_json_with_markdown_ticks(mock_pipeline):
    """Test Edge Case: LLM disobeys and wraps JSON in markdown blocks."""
    engine = LLMEngine()
    # Simulate an LLM hallucinating extra conversational text
    text = '''Here is the data you requested:
```json
{"name": "test", "document_type": "Invoice"}
```
Hope this helps!'''
    
    result = engine._parse_json(text)
    # The regex fallback should successfully hunt down the JSON inside the text
    assert result["name"] == "test"
    assert result["document_type"] == "Invoice"

@patch('src.llm_engine.pipeline')
def test_parse_json_failure_raises_exception(mock_pipeline):
    """Test Edge Case: LLM outputs zero JSON, just garbage text."""
    engine = LLMEngine()
    text = "I am sorry, I couldn't find any data in this document because it is too blurry."
    
    with pytest.raises(LLMExtractionError) as exc_info:
        engine._parse_json(text)
        
    assert "No JSON structure could be found" in str(exc_info.value)
