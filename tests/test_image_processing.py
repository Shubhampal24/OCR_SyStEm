import numpy as np
from PIL import Image
from src.image_processing import ImageProcessor

def test_process_image_converts_pil_to_grayscale_numpy():
    """Test that a web-uploaded PIL image is properly transformed into an OpenCV format."""
    # Create a 100x100 white RGB image (dummy data)
    dummy_img = Image.new('RGB', (100, 100), color='white')
    
    processor = ImageProcessor()
    result = processor.process_image(dummy_img)
    
    # Check that OpenCV output is a numpy array
    assert isinstance(result, np.ndarray)
    
    # Check that it converted to grayscale (2D array, no color channels)
    assert len(result.shape) == 2
    assert result.shape == (100, 100)
