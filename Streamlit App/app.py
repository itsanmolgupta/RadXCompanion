import streamlit as st
from classification import classification_page
from report_generation import report_generation_page

st.set_page_config(page_title="RadX: Chest X-ray Companion", page_icon="🩺")
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Classification", "Report Generation"])

    if page == "Classification":
        classification_page()
    elif page == "Report Generation":
        report_generation_page()

if __name__ == "__main__":
    main()