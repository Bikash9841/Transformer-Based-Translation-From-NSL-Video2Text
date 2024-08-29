# Transformer Based Translation From Nepali Sign Language Video to Text

## Installation:

`pip install -q torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118 xformers==0.0.21`
`pip install -r requirements.txt`

## Run:

`cd apps`
`streamlit run app.py`

## Overview

The project focuses on developing a system to interpret Nepali Sign Language gestures from video into Nepali sentence. It involves using a video vision transformer model to efficiently encode video frames and a decoder to translate it into corresponding nepali sentences. The project will utilize a dataset of videos, each depicting specific Nepali sign language sentences, to train and evaluate the system. For preprocessing, the Mediapipe framework will be employed to convert raw video into a stick-figure representation by extracting key landmarks for the face, hands, and body. Multiple frames will be extracted from the video and fed into the transformer model. Ultimately, The end to end transformer network will provide the nepali text sequences based on the video depicting nepali sentences in sign language.

## Model Used:

Encoder: [ViViT] (https://arxiv.org/pdf/2103.15691)
Decoder: [Custom_Decoder] (https://i.sstatic.net/nQ2f5.png)
