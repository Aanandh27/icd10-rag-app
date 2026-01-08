import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
from google import genai


# -----------------------------
# App Title
# -----------------------------
st.title("ICD-10 Code Suggester")
st.write(
    "Enter a clinical description below and click **Search ICD-10 Codes** "
    "to retrieve relevant ICD-10-CM codes with AI-generated explanations.")

st.warning("⚠️ This tool is for educational purposes only and does not provide medical or billing advice.")


# -----------------------------
# Gemini API Setup 
# -----------------------------
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Gemini API key not found. Please add it in Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)


# -----------------------------
# Load ICD-10 Data
# -----------------------------
df = pd.read_csv("icd10_codes.csv")
st.caption(f"Using {len(df)} ICD-10-CM codes (CMS dataset)")


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
def search_icd10_codes(query, top_k=3):
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
# Gemini Prompt Builder
# -----------------------------
def build_gemini_prompt(user_query, retrieved_codes):
    context = "\n".join(
        [f"{i+1}. {code}: {desc}" for i, (code, desc) in enumerate(retrieved_codes)]
    )

    prompt = f"""
You are a medical coding assistant.

Clinical description:
{user_query}

Below are EXACTLY 3 ICD-10-CM codes retrieved by semantic search:

{context}

Task:
- Explain EACH of the 3 codes separately
- Use the same numbering as above (1, 2, 3)
- Explain why EACH code is relevant to the clinical description
- Use 1–2 sentences per code
- Do NOT skip any code
- Do NOT add new codes
- Do NOT reorder the codes

Output format:
1. CODE – explanation
2. CODE – explanation
3. CODE – explanation
"""
    return prompt


# -----------------------------
# Gemini Reasoning
# -----------------------------
def generate_reasoning_gemini(user_query, retrieved_codes):
    prompt = build_gemini_prompt(user_query, retrieved_codes)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

# -----------------------------
# User Input
# -----------------------------
st.subheader("Clinical Description")

user_input = st.text_area(
    "Describe the patient's condition:",
    placeholder="e.g., acute chest pain after exercise",
    height=120
)

search_clicked = st.button("🔍 Search ICD-10 Codes")

if search_clicked:
    if not user_input.strip():
        st.error("Please enter a clinical description before searching.")
        st.stop()

    st.subheader("Top Retrieved ICD-10 Codes (Vector Search)")
    retrieved = search_icd10_codes(user_input,top_k=top_k)

    for code, desc in retrieved:
        st.write(f"**{code}**: {desc}")

    st.subheader("AI Explanation")
    with st.spinner("Analyzing with Gemini..."):
        explanation = generate_reasoning_gemini(user_input, retrieved)

    st.markdown(explanation)




