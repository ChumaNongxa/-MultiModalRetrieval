# Multi-Modal Image Retrieval System

A system that retrieves images based on natural language text descriptions using deep learning models.

![Streamlit UI Screenshot](archive/images/Streamlit%20UI%20Screenshot.png)

## Features

- Text-to-image retrieval using CLIP model
- Fast similarity search using FAISS
- REST API backend using Flask
- Streamlit frontend with accessibility features
- Support for 500 sample images from AIvS dataset

## System Architecture

![Image Retrieval System Flow](archive/images/Image%20Retrieval%20Flow.png)

## Setup Instructions

### Prerequisites

- Anaconda
- Git

### Installation

1. Create and activate Conda environment using the provided environment.yml:
```bash
conda env create -f environment.yml
conda activate multimodal
```

Alternatively, you can create the environment manually:
```bash
conda create -n multimodal python=3.8
conda activate multimodal
pip install -r requirements.txt
```

### Running the Application

1. Activate the Conda environment (if not already activated):
```bash
conda activate multimodal
```

2. Start the Flask backend server:
```bash
python -m flask --app app.main run
```

3. In a new terminal, activate Conda environment and start the Streamlit frontend:
```bash
conda activate multimodal
streamlit run frontend.py
```

The application will open in your default web browser.
Submit your query to see top k image results.

## Project Structure

```
├── app/                   # Backend code
│   ├── main.py            # Flask application
│   └── services/          # Process logic
│       ├── clip_service.py    # CLIP model integration
│       └── image_service.py   # Image processing service
├── archive/               # Sample image dataset
│   └── sampled_test_data/    # Test images
├── frontend.py            # Streamlit frontend
├── styles/               # Frontend styling
│   └── main.css         # CSS styles
├── environment.yml       # Conda environment config
└── requirements.txt      # Python dependencies
```

## Assumptions

1. Images are pre-processed and stored locally
2. System runs on a machine with sufficient RAM for CLIP model
3. Frontend and backend run on the same machine in development
4. Top-K is configurable but defaults to 5 results