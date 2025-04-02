import streamlit as st
import chromadb

import chromadb

# Initialize ChromaDB with persistent storage
client = chromadb.PersistentClient(path="vectordb")  # Saves embeddings to 'vectordb' directory
# Retrieve or create a collection for subtitles
collection = client.get_or_create_collection(name="subtitle_sem")

# Streamlit Page Config
st.set_page_config(
    page_title="ShowFinder 🔍",
    page_icon=":tv:",
    layout="wide"
)

# Set Custom CSS for Styling
def set_style():
    st.markdown("""
    <style>
    .title { font-size: 36px !important; text-align: center !important; margin-bottom: 30px !important; }
    .subtitle { font-size: 24px !important; text-align: center !important; margin-bottom: 20px !important; }
    .text { font-size: 18px !important; margin-bottom: 10px !important; }
    .stTextInput>div>div>input { font-size: 18px; }
    </style>
    """, unsafe_allow_html=True)

set_style()

# Page Header
st.title('ShowFinder 🔍')
st.subheader('Say goodbye to endless scrolling and decision fatigue 😫')

# Search Input
query_text = st.text_input('Enter your query:', key="search_query")

# Function to fetch similar titles
def similar_title(query_text):
    if not query_text.strip():
        return []  # Return empty list if query is blank
    
    try:
        result = collection.query(
            query_texts=[query_text],
            include=["metadatas", "distances"],
            n_results=10
        )
        ids = result.get('ids', [[]])[0]
        distances = result.get('distances', [[]])[0]
        metadatas = result.get('metadatas', [[]])[0]

        sorted_data = sorted(zip(metadatas, ids, distances), key=lambda x: x[2], reverse=True)
        return sorted_data

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return []

# Automatically search when user types (no button needed)
if query_text:
    result_data = similar_title(query_text)
    
    if result_data:
        st.success('Here are the most relevant subtitle names:')
        for metadata, ids, distance in result_data:
            subtitle_name = metadata.get('subtitle_name', 'Unknown')
            subtitle_id = metadata.get('subtitle_id', '')
            subtitle_link = f"https://www.opensubtitles.org/en/subtitles/{subtitle_id}" if subtitle_id else "#"

            st.markdown(f"🎬 **[{subtitle_name}]({subtitle_link})**")
    else:
        st.warning("No relevant subtitles found. Try a different query!")

  
