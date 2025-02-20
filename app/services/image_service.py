"""
This module provides a service for image similarity search using FAISS indexing.
It works in conjunction with the CLIP service to enable efficient text-to-image
search capabilities by maintaining a searchable index of image embeddings.
"""

# Import Libraries
import os
import faiss
import shutil
import random
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict
from .clip_service import ClipService


class ImageService:

    def __init__(self, clip_service: ClipService):
        # Initialize the Image Service.
        # An input instance of ClipService for encoding images and text
        
        self.clip_service = clip_service
        self.image_paths = []
        self.index = None
        # Use relative path from the current file location
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.dataset_path = os.path.join(current_dir, 'archive', 'sampled_test_data')
        
    def initialize_index(self):
        # Initialize FAISS index with image embeddings.
        # Processes the first 500 images from the dataset in batches of 32
        # Generating their embeddings using CLIP, and builds a FAISS index
        # The embeddings are L2-normalized before being added to the index.
        # Get first 500 images from the dataset
        
        all_images = os.listdir(self.dataset_path)[:500]
        self.image_paths = [os.path.join(self.dataset_path, img) for img in all_images]
        
        # Process images in batches
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(self.image_paths), batch_size):
            batch_paths = self.image_paths[i:i + batch_size]
            images = [Image.open(path).convert('RGB') for path in batch_paths]
            batch_embeddings = self.clip_service.encode_images(images)
            embeddings.append(batch_embeddings)
            
            # Clean up
            for img in images:
                img.close()
        
        # Concatenate all embeddings
        all_embeddings = np.vstack(embeddings)
        
        # Normalize embeddings
        faiss.normalize_L2(all_embeddings)
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(all_embeddings.shape[1])
        self.index.add(all_embeddings)
    
    def search_images(self, query: str, top_k: int = 5):
        # Search for similar images given a text query.
        # Input text query to search for similar images.
        # Input number of results to return. Defaults to 5.
            
        # Returns list of dictionaries containing search results, each with:
            # image_path: Path to the matched image
            # similarity_score: Cosine similarity score
            # filename: Base name of the image file
                
        if not self.index:
            raise ValueError("Index not initialized")
            
        # Encode query
        query_embedding = self.clip_service.encode_text(query)
        faiss.normalize_L2(query_embedding)
        
        # Search index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "image_path": self.image_paths[idx],
                "similarity_score": float(distances[0][i]),
                "filename": os.path.basename(self.image_paths[idx])
            })
            
        return results

    @staticmethod
    def create_sample_dataset(source_dir, target_dir, sample_size: int = 500):
        # Randomly samples images from source directory and copies them to target directory
        # input source_dir: Path to source directory containing images
        # input target_dir: Path to target directory where sampled images will be copied
        # input sample_size: Number of images to sample (default: 500)
       
       
        # Create target directory if it doesn't exist 
        os.makedirs(target_dir, exist_ok=True)
        
        # Get list of all jpg files in source directory
        all_images = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
        
        # Randomly sample specified number of images
        sampled_images = random.sample(all_images, min(sample_size, len(all_images)))
        
        # Copy sampled images to target directory
        for image_name in sampled_images:
            source_path = os.path.join(source_dir, image_name)
            target_path = os.path.join(target_dir, image_name)
            shutil.copy2(source_path, target_path)