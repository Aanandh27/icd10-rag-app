import streamlit as st
import pandas as pd

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
# User Input
# -----------------------------
st.subheader("Clinical Description")
user_input = st.text_area(
    "Describe the patient's condition:",
    placeholder="e.g., acute chest pain after exercise",
    height=120
)

# -----------------------------
# Display Input (Temporary)
# -----------------------------
if user_input:
    st.subheader("You entered:")
    st.write(user_input)

# -----------------------------
# Preview ICD-10 Data
# -----------------------------
st.subheader("Sample ICD-10 Codes")
st.dataframe(df.head(10))
