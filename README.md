# Seq2Seq English-Arabic Translation Model

This repository contains a Seq2Seq model for English-Arabic translation, implemented in TensorFlow/Keras.
The model uses an encoder-decoder LSTM architecture and can translate sentences from English to Arabic.
This setup includes weights and tokenizers for easy local testing.

Requirements:
To run the model, you'll need the following packages:
- Python 3.x
- TensorFlow
- Streamlit (for a web-based interface)
- Other dependencies listed in requirements.txt

# Install dependencies
pip install -r requirements.txt

Repository Files:
main.py                 : Streamlit application for testing the translation model in a web interface.
# model_weights.h5        : Pretrained model weights. Download with:
# wget https://drive.google.com/uc?id=1Pd0xf17Fs6J5gZaSzgHBEdTm0OlUBWEd -O model_weights.h5
# english_tokenizer.pickle: Tokenizer for English input. Download with:
# wget https://drive.google.com/uc?id=1uQ19mOKEJxL5BGwD1mQxUxfbg2duH4Mi -O english_tokenizer.pickle
# arabic_tokenizer.pickle : Tokenizer for Arabic output. Download with:
# wget https://drive.google.com/uc?id=1YCtwMdjJ_46ihaZS9RPqtQrALyGFdp3L -O arabic_tokenizer.pickle

# Usage:
You can test the model by running the Streamlit application.
Run this command to start the app:
streamlit run main.py
