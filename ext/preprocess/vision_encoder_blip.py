"""
Vision encoder using original BLIP model (lighter than BLIP-2).
"""
import os
from typing import List, Optional
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch


class VisionEncoderBLIP:
    """Encode images to text captions using BLIP model (lighter alternative)."""
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base", device: str = "cpu"):
        """Initialize vision encoder.
        
        Args:
            model_name: HuggingFace model name for BLIP
            device: Device to run the model on ("cpu", "cuda", or "mps")
        """
        # Apple Silicon対応: multithreading問題を回避
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # Apple Silicon MPS自動検出
        if device == "cpu" and torch.backends.mps.is_available():
            device = "mps"
            print("Apple Silicon detected: Using MPS for acceleration")
        elif device == "cpu":
            torch.set_num_threads(4)  # Apple Silicon用に最適化
            
        self.device = device
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
        
        # Apple Silicon MPS最適化
        if device == "mps":
            self.model.eval()
            print("BLIP model loaded on MPS device for acceleration")
    
    def generate_caption(self, image: Image.Image, max_length: int = 50) -> str:
        """Generate caption for a single image.
        
        Args:
            image: PIL Image object
            max_length: Maximum length of generated caption
            
        Returns:
            Generated caption text
        """
        try:
            # 画像を前処理
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # キャプション生成
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_length=max_length,
                    min_length=10,
                    num_beams=3,
                    early_stopping=True
                )
            
            # テキストにデコード
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            return caption.strip() if caption.strip() else "Unable to generate caption"
            
        except Exception as e:
            print(f"Warning: Failed to generate caption for image: {e}")
            return "Image caption generation failed"
    
    def generate_captions_batch(self, images: List[Image.Image], max_length: int = 50) -> List[str]:
        """Generate captions for multiple images.
        
        Args:
            images: List of PIL Image objects
            max_length: Maximum length of generated captions
            
        Returns:
            List of generated caption texts
        """
        captions = []
        for i, image in enumerate(images):
            print(f"Generating caption {i+1}/{len(images)}...")
            caption = self.generate_caption(image, max_length)
            captions.append(caption)
        return captions


def load_image_from_bytes(image_bytes: bytes) -> Optional[Image.Image]:
    """Load PIL Image from bytes data.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        PIL Image object or None if loading fails
    """
    try:
        from io import BytesIO
        return Image.open(BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        print(f"Warning: Failed to load image from bytes: {e}")
        return None