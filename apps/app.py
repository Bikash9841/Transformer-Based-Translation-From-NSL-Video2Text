import streamlit as st
import os
import requests
from model import run_inference, v2t_model, target_tokenizer, get_video_tensor
import torch

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

    st.video(video_path)

    src_video = get_video_tensor(video_path)
    print(src_video)

    if st.button("Translate"):

        st.write('Pass')
        # res = run_inference(
        #     v2t_model, {"video": src_video}, target_tokenizer, 10, torch.device('cpu'))

        # st.write(res['translation'])
