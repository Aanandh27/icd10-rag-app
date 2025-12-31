import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from google import generativeai as genai
import os



# -----------------------------
# App Title
# -----------------------------
st.title("ICD-10 Code Suggester")
st.write("Enter a clinical description to explore relevant ICD-10 codes.")

# -----------------------------
# Load ICD-10 Data
# -----------------------------
df = pd.read_csv("icd10_codes.csv")
st.write(f"Total ICD-10 codes loaded: {len(df)}")

# -----------------------------
# Load Embedding Model 
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# -----------------------------
# Create ICD-10 Embeddings 
# -----------------------------
@st.cache_resource
def create_icd10_embeddings(descriptions):
    embeddings = embedding_model.encode(descriptions)
    return embeddings

icd10_embeddings = create_icd10_embeddings(df["description"].tolist())

# -----------------------------
# FAISS
# -----------------------------

@st.cache_resource
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

faiss_index = build_faiss_index(icd10_embeddings)
# -----------------------------
# Search Function
# -----------------------------
def search_icd10_codes(query, top_k=5):
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(
        np.array(query_embedding), top_k
    )

    results = []
    for idx in indices[0]:
        code = df.iloc[idx]["code"]
        description = df.iloc[idx]["description"]
        results.append((code, description))

    return results


# -----------------------------
# User Input (STAGE 2)
# -----------------------------
st.subheader("Clinical Description")
user_input = st.text_area(
    "Describe the patient's condition:",
    placeholder="e.g., acute chest pain after exercise",
    height=120
)

if user_input:
    st.subheader("You entered:")
    st.write(user_input)
# -----------------------------
# FAISS TEST
# -----------------------------
if user_input:
    st.subheader("Top Retrieved ICD-10 Codes (Vector Search)")
    results = search_icd10_codes(user_input)
    for code, desc in results:
        st.write(f"**{code}**: {desc}")

# -----------------------------
# API 
# -----------------------------

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Gemini API key not found.")
    st.stop()

client = genai.Client(api_key=api_key)

# -----------------------------
# Gemini Prompt
# -----------------------------

def build_gemini_prompt(user_query, retrieved_codes):
    context = "\n".join([f"{code}: {desc}" for code, desc in retrieved_codes])
    
    prompt = f"""
You are a medical coding assistant.

Clinical description:
{user_query}

Candidate ICD-10-CM codes:
{context}

Task:
Select the most relevant ICD-10-CM codes.
Explain each in 1-2 sentences.
Do NOT add new codes.
Provide a bullet-point list.
"""
    return prompt

# -----------------------------
# reasoning
# -----------------------------

def generate_reasoning_gemini(user_query, retrieved_codes):
    prompt = build_gemini_prompt(user_query, retrieved_codes)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

if user_input:
    st.subheader("Top Retrieved ICD-10 Codes (Vector Search)")
    retrieved = search_icd10_codes(user_input)
    for code, desc in retrieved:
        st.write(f"**{code}**: {desc}")

    st.subheader("Gemini Explanation")
    explanation = generate_reasoning_gemini(user_input, retrieved)
    st.write(explanation)



# -----------------------------
# Preview ICD-10 Data
# -----------------------------
st.subheader("Sample ICD-10 Codes")
st.dataframe(df.head(10))
