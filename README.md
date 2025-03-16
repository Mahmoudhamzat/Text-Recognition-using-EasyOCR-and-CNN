# Text-Recognition-using-EasyOCR-and-CNN
Text Recognition using EasyOCR and CNN
This project uses EasyOCR and a simple Convolutional Neural Network (CNN) to extract text from images and PDFs. It involves image enhancement techniques like contrast adjustment, sharpening, and denoising, followed by text recognition in multiple languages.

Requirements
To install the necessary dependencies, run the following command:

bash
نسخ
تحرير
pip install easyocr opencv-python-headless matplotlib pillow ipywidgets pdf2image gradio torch torchvision torchaudio
Libraries Used
EasyOCR: Optical Character Recognition (OCR) to extract text from images.
OpenCV: For image processing like contrast enhancement, sharpening, denoising, etc.
PyTorch: For building and running a simple CNN for text recognition.
Gradio: To create an interactive interface for users to upload images and get text extracted from them.
PDF2Image: To convert PDF pages into images.
Matplotlib, Pillow: For image visualization and enhancement.
ipywidgets: For interactive widget creation.
Project Structure
1. SimpleCNN class
A basic CNN model with two convolutional layers and two fully connected layers for text recognition. It serves as a prototype and can be extended for more advanced models.

2. enhance_image function
Enhances the input image by adjusting the contrast, sharpening the image, denoising, and applying adaptive thresholding.

3. extract_text_from_image function
Uses EasyOCR to extract text from the enhanced image in the selected languages.

4. process_pdf function
Processes a PDF file, converting each page to an image and extracting text from it.

5. process_image function
Handles the processing of a single image, including enhancement and text extraction.

6. gr.Interface
An interactive interface built using Gradio that allows users to upload an image and select languages for OCR. It outputs the enhanced image and the extracted text.

Usage
Running the Interface
To run the interface, just execute the script. You can upload an image and select which languages to use for OCR. The supported languages are:

Arabic (ar)
English (en)
French (fr)
Spanish (es)
German (de)
The interface will show the enhanced image and extracted text.

Example
Upload an image with text.
Select languages (e.g., Arabic and English).
The system will process the image and display the extracted text.
