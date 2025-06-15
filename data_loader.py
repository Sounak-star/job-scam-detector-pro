import streamlit as st  # This was missing - causing the errors
import pandas as pd

def parse_jobs_csv(uploaded_file):
    """
    Processes uploaded CSV file containing job listings
    Args:
        uploaded_file: Streamlit file uploader object
    Returns:
        DataFrame if valid, None if errors occur
    """
    try:
        df = pd.read_csv(uploaded_file)
        required_cols = ["title", "description", "location"]
        
        # Check for missing columns
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            return None
            
        return df
        
    except pd.errors.EmptyDataError:
        st.error("Uploaded file is empty")
        return None
    except Exception as e:
        st.error(f"Failed to process file: {str(e)}")
        return None