import os
import requests
from PIL import Image
import streamlit as st
from pathlib import Path


st.set_page_config(
    page_title="Image Search",
    page_icon="🔍",
    layout="wide"
)

# Load external CSS
with open("styles/main.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Multi-Modal Image Retrieval")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)

# Main content
query = st.text_input(
    "Enter your image description",
    placeholder="e.g., 'A dog playing in the park'"
)

if query:
    try:
        # Make API request to Flask backend
        response = requests.post(
            "http://localhost:8000/search",
            params={"query": query, "top_k": top_k}
        )
        results = response.json()
        
        # Display results in a grid
        cols = st.columns(3)
        for idx, result in enumerate(results):
            col_idx = idx % 3
            with cols[col_idx]:
                if os.path.exists(result["image_path"]):
                    img = Image.open(result["image_path"])
                    st.image(
                        img,
                        caption=f"Score: {result['similarity_score']:.2f}",
                        use_column_width=True
                    )
                else:
                    st.error(f"Image not found: {result['image_path']}")
                    
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend server. Please make sure it's running.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")