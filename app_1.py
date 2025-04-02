import streamlit as st
from sentence_transformers import SentenceTransformer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import chromadb

# Download NLTK stopwords if not already available
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to the ChromaDB client and collection
client = chromadb.PersistentClient(path="vectordb")
collection = client.get_collection("searchengine1")

def clean_text(text):
    """Preprocess the input text: remove special characters, tokenize, remove stopwords, and lemmatize."""
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stop_words]
    return ' '.join(clean_tokens).strip()

# Streamlit UI
st.title("Semantic Search Engine")

query = st.text_input("Enter your search query:")

if query:
    # Preprocess and encode the query
    cleaned_query = clean_text(query)
    query_embedding = model.encode([cleaned_query])

    # Perform the query on ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5,
        include=['documents']
    )

    documents = results['documents']

    if documents:
        st.subheader("Search Results:")
        for i, query_documents in enumerate(documents):
            for j, document in enumerate(query_documents):
                st.write(f"**Result {j+1}:** {document}")
    else:
        st.write("No results found.")
