import streamlit as st
import sqlite3
import pandas as pd
import zipfile
import io
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


bert_model = SentenceTransformer('all-MiniLM-L12-v2')

# Initialize ChromaDB client
similarity_search_database = chromadb.PersistentClient(path="chroma_db")
collection = similarity_search_database.get_or_create_collection(name="subtitles")

# --- Function to extract subtitle content from zip files ---
def extract_subtitle(binary_data):
    with io.BytesIO(binary_data) as f:
        with zipfile.ZipFile(f, 'r') as zip_file:
            subtitle_content = zip_file.read(zip_file.namelist()[0])
    return subtitle_content.decode("latin1")

# --- Function to chunk documents ---
def chunk_document(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# --- Function to embed text using BERT ---
def embed_text(text):
    return bert_model.encode(text).tolist()

# --- Streamlit App ---
def main(db_path='eng_subtitles_database.db'):
    st.title("Subtitle Search Engine")

    # --- Database Connection ---
    st.sidebar.header("Database Configuration")
    db_path = st.sidebar.text_input("Enter database path:", db_path)

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Load data from database
        df = pd.read_sql_query("SELECT * FROM zipfiles", conn)

        # Extract and process subtitles
        df['subtitle_text'] = df['content'].apply(extract_subtitle)

        # Chunk documents and create embeddings
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        df['split_text'] = df['subtitle_text'].apply(lambda x: text_splitter.split_text(x))

        all_chunks = df.explode('split_text')  # Flatten the list
        all_chunks = all_chunks.dropna(subset=['split_text'])  # Remove empty entries

        # Store embeddings in ChromaDB
        for idx, row in all_chunks.iterrows():
            collection.add(
                ids=[str(idx)],
                embeddings=[embed_text(row['split_text'])],
                metadatas=[{"name": row['name'], "text": row['split_text']}] 
            )
        
        # User input for text query
        st.subheader("Enter a text query")
        query_text = st.text_input("Search subtitles:")

        if query_text:
            query_embedding = embed_text(query_text)
            results = collection.query(query_embeddings=[query_embedding], n_results=2)

            # Display search results
            st.subheader("Semantic Search Results:")
            for result in results["metadatas"][0]:
                st.write(f"Subtitle Name: {result['name']}")
                st.write(f"Matching Text: {result['text']}")
                st.write("---")

        conn.close()
    
    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")

if __name__ == "__main__":
    main()
