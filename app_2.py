import streamlit as st
from sentence_transformers import SentenceTransformer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import chromadb

# Download NLTK resources if not available
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the SentenceTransformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Connect to ChromaDB client
@st.cache_resource
def get_chroma_collection():
    try:
        client = chromadb.PersistentClient(path="vectordb")
        return client.get_collection("searchengine1")
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

collection = get_chroma_collection()

# Function to clean and preprocess text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(clean_tokens).strip()

# Streamlit UI
st.title("🔍 Semantic Search Engine")
st.write("Enter your query below to search relevant documents.")

query = st.text_input("Search Query:", "")

if query and collection:
    with st.spinner("Searching..."):
        cleaned_query = clean_text(query)
        query_embedding = model.encode([cleaned_query])

        # Perform the search query
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=5,
            include=['documents']
        )

        documents = results.get('documents', [])

        # Display results
        if documents:
            st.subheader("🔹 Search Results:")
            for i, query_documents in enumerate(documents):
                for j, document in enumerate(query_documents):
                    st.markdown(f"**{j+1}.** {document}")
        else:
            st.warning("No results found. Try a different query.")
