# Import Libraries
import os
from flask import Flask, request, jsonify
from .services.clip_service import ClipService
from .services.image_service import ImageService

# Initialize Flask application
app = Flask(__name__)

# Global service variables
image_service = None
clip_service = None
is_initialized = False

# Initialize CLIP and Image services if not already initialized
def initialize_services():
    global image_service, clip_service, is_initialized
    if not is_initialized:
        clip_service = ClipService()
        image_service = ImageService(clip_service)
        image_service.initialize_index()
        is_initialized = True

# Middleware to ensure services are initialized before any request
@app.before_request
def before_request():
    initialize_services()

# Root endpoint - returns API information
@app.route("/")
def root():
    return jsonify({"message": "Multi-Modal Image Retrieval API"})

# Search endpoint - processes image search requests
@app.route("/search", methods=["POST"])
def search_images():
    # Get query parameters from the request
    query = request.args.get("query")
    top_k = request.args.get("top_k", default=5, type=int)
    
    try:
        # Perform image search using the image service
        results = image_service.search_images(query, top_k)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check endpoint - for monitoring service status
@app.route("/health")
def health_check():
    return jsonify({"status": "healthy"})

# Run the Flask application in debug mode if executed directly
if __name__ == "__main__":
    app.run(debug=True, port=8000)