# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 16:09:35 2025

@author: mmmkh
"""

!pip install easyocr opencv-python-headless matplotlib pillow ipywidgets pdf2image gradio torch torchvision torchaudio

import easyocr
import cv2
import numpy as np
from google.colab import files
from google.colab.patches import cv2_imshow
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import ipywidgets as widgets
from pdf2image import convert_from_path
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple CNN for text recognition
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # For example, 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to enhance the image
def enhance_image(input_img, contrast_factor=1.5, sharpen_amount=1.5):
    pil_img = Image.fromarray(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Contrast(pil_img)
    enhanced_img = enhancer.enhance(contrast_factor)
    sharpener = ImageEnhance.Sharpness(enhanced_img)
    enhanced_img = sharpener.enhance(sharpen_amount)
    enhanced_img_np = cv2.cvtColor(np.array(enhanced_img), cv2.COLOR_RGB2BGR)
    enhanced_img_gray = cv2.cvtColor(enhanced_img_np, cv2.COLOR_BGR2GRAY)
    denoised_img = cv2.fastNlMeansDenoising(enhanced_img_gray, None, 30, 7, 21)
    blurred_img = cv2.GaussianBlur(denoised_img, (5, 5), 0)
    enhanced_img = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return enhanced_img

# Function to extract text using EasyOCR
def extract_text_from_image(enhanced_img, languages=['ar', 'en']):
    reader = easyocr.Reader(languages, gpu=torch.cuda.is_available())
    result = reader.readtext(enhanced_img)
    extracted_text = "\n".join([text for (_, text, _) in result])
    return extracted_text

# Function to process PDF files
def process_pdf(pdf_path, languages=['ar', 'en']):
    images = convert_from_path(pdf_path)
    extracted_text = ""
    for img in images:
        img_np = np.array(img)
        enhanced_img = enhance_image(img_np)
        extracted_text += extract_text_from_image(enhanced_img, languages) + "\n"
    return extracted_text

# Function to process a single image
def process_image(img, selected_languages=['ar', 'en']):
    enhanced_img = enhance_image(np.array(img))
    extracted_text = extract_text_from_image(enhanced_img, languages=selected_languages)
    return enhanced_img, extracted_text

# Interactive interface using Gradio
iface = gr.Interface(
    fn=process_image,
    inputs=["image", gr.CheckboxGroup(['ar', 'en', 'fr', 'es', 'de'], label="Select Languages", value=['ar', 'en'])],
    outputs=["image", "text"],
    title="Text Recognition in Images Using OCR and CNN",
    description="Upload an image with text, and we will enhance it and use EasyOCR and CNN to extract the text."
)
iface.launch(share=True)