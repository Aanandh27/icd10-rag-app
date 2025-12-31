import streamlit as st
import pandas as pd

st.title("ICD-10 Code Suggester")

st.write("Loading ICD-10 data...")

df = pd.read_csv("icd10_codes.csv")

st.write("Number of ICD-10 codes loaded:")
st.write(len(df))

st.subheader("Sample ICD-10 Codes")
st.dataframe(df.head(10))
