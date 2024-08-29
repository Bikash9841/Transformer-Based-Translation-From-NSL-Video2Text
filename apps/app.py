import streamlit as st
import os
import requests
from model import run_inference, v2t_model, target_tokenizer, get_video_tensor
from stick_figure import conv_to_stick_fig
import torch


'''
A streamlit application to infer the model.
'''

result = None

# Initialize session state variables if they don't exist
if "video_o" not in st.session_state:
    st.session_state.video_o = None
if "video_c" not in st.session_state:
    st.session_state.video_c = None


# Directory to save the uploaded videos
UPLOAD_DIR = "uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.title("Nepali Sign Language Translation")


uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save the uploaded video locally
    video_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    video_o, video_c = conv_to_stick_fig(video_path, UPLOAD_DIR)

    # Store the videos in session state
    st.session_state.video_o = video_o
    st.session_state.video_c = video_c

# Create two columns for displaying videos
col1, col2 = st.columns(2)

if st.session_state.video_o and st.session_state.video_c:
    with col1:
        st.write("Original Video")
        st.video(st.session_state.video_o)

    with col2:
        st.write("Converted Video")
        st.video(st.session_state.video_c)

    with col1:
        if st.button("Translate"):

            src_video = get_video_tensor(st.session_state.video_c)
            result = run_inference(
                v2t_model, {"video": src_video}, target_tokenizer, 10, torch.device('cpu'))

        with col2:
            if result:
                st.markdown(
                    f"<h3>{result['translation']}</h3>", unsafe_allow_html=True)
