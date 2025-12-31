import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer

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
# Load Embedding Model (STEP 3.2)
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# -----------------------------
# Create ICD-10 Embeddings (STEP 3.3)
# -----------------------------
@st.cache_resource
def create_icd10_embeddings(descriptions):
    embeddings = embedding_model.encode(descriptions)
    return embeddings

icd10_embeddings = create_icd10_embeddings(df["description"].tolist())

# -----------------------------
# Embedding Test (STEP 3.4)
# -----------------------------
st.subheader("Embedding Test")
st.write("Embedding vector length:")
st.write(len(icd10_embeddings[0]))

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
# Preview ICD-10 Data
# -----------------------------
st.subheader("Sample ICD-10 Codes")
st.dataframe(df.head(10))
