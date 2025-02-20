"""
This module provides a service for encoding text and images using OpenAI's CLIP model.
CLIP (Contrastive Language-Image Pre-Training) can create embeddings for both images
and text in the same vector space, enabling cross-modal similarity comparisons.
"""

# Import Libraries
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class ClipService:
    def __init__(self):
        # Initialize the CLIP model and processor.
        # Automatically selects GPU if available, otherwise uses CPU.
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(self.device)

    def encode_text(self, text):
        # Encode a text string into a feature vector using CLIP.
        # The input text to be encoded
        # Returns an array containing the text embeddings
        
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs) # Unpacking Dictionary Elements
        
        return text_features.cpu().numpy()

    def encode_image(self, image):
        
        # Encode a single image into a feature vector using CLIP.
        # The input PIL image object to be encoded
        # Returns an array containing the image embeddings

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
        return image_features.cpu().numpy()

    def encode_images(self, images):
        
        # Encode multiple images into feature vectors using CLIP.
        # A list of PIL Image objects to encode
        # Returns An array containing the image embeddings for all input images
        
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
        return image_features.cpu().numpy()