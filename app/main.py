"""Connection check for the development API."""
import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="BrainScan", page_icon="🧠")
st.title("BrainScan")
st.write("Brain MRI classification and tumor detection")
st.info("Environment setup complete. Model training and prediction workflows are pending.")
api_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
if st.button("Check API connection"):
    try:
        response = requests.get(f"{api_url.rstrip('/')}/health", timeout=10)
        response.raise_for_status()
        st.json(response.json())
    except requests.RequestException as exc:
        st.error(f"Cannot reach the API: {exc}")
