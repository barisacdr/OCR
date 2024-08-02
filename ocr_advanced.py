import os
import pytesseract
import easyocr
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import torch
import paddle
from PIL import Image

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe' # It can be at Program Files or Program Files (x86), check for the correct path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize OCR engines
reader_easyocr = easyocr.Reader(['en'], gpu=True if device == 'cuda' else False)
paddle.set_device('gpu' if paddle.is_compiled_with_cuda() else 'cpu')
reader_paddleocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=paddle.is_compiled_with_cuda())

# Print device information
print(f"\nPyTorch device: {device}")
print(f"PaddlePaddle device: {'GPU' if paddle.is_compiled_with_cuda() else 'CPU'}")
print(f"EasyOCR using GPU: {device == 'cuda'}")
print(f"PaddleOCR using GPU: {paddle.is_compiled_with_cuda()}")

def clean_text(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return ' '.join(lines)

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    # Dilation and erosion to remove noise
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(denoised, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    return eroded

def process_image(image, use_tesseract=True, use_easyocr=True, use_paddleocr=True):
    image_np = np.array(image)
    processed_image = preprocess_image(image_np)
    
    # Find contours in the processed image
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    extracted_data = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Expand the bounding box slightly
        x, y, w, h = max(0, x-5), max(0, y-5), min(w+10, image_np.shape[1]-x), min(h+10, image_np.shape[0]-y)
        table_image = image_np[y:y+h, x:x+w]

        result = {'image': table_image, 'texts': {}}

        if use_tesseract:
            try:
                tesseract_text = pytesseract.image_to_string(table_image)
                result['texts']['PyTesseract'] = clean_text(tesseract_text)
            except Exception as e:
                print(f"PyTesseract error: {str(e)}")
                result['texts']['PyTesseract'] = "Error occurred during processing"

        if use_easyocr:
            try:
                easyocr_result = reader_easyocr.readtext(table_image)
                result['texts']['EasyOCR'] = ' '.join([text[1] for text in easyocr_result])
            except Exception as e:
                print(f"EasyOCR error: {str(e)}")
                result['texts']['EasyOCR'] = "Error occurred during processing"

        if use_paddleocr:
            try:
                paddleocr_result = reader_paddleocr.ocr(table_image, cls=True)
                if paddleocr_result and paddleocr_result[0]:
                    result['texts']['PaddleOCR'] = ' '.join([
                        line[1][0] for line in paddleocr_result[0] if line is not None and isinstance(line, list) and len(line) > 1
                    ])
                else:
                    result['texts']['PaddleOCR'] = "No text detected"
            except Exception as e:
                print(f"PaddleOCR error: {str(e)}")
                result['texts']['PaddleOCR'] = "Error occurred during processing"

        extracted_data.append(result)

    return extracted_data

def extract_content(file_path, use_tesseract=True, use_easyocr=True, use_paddleocr=True):
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.pdf':
        images = convert_from_path(file_path)
        extracted_data = []
        for image in images:
            extracted_data.extend(process_image(image, use_tesseract, use_easyocr, use_paddleocr))
    elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
        image = Image.open(file_path)
        extracted_data = process_image(image, use_tesseract, use_easyocr, use_paddleocr)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    return extracted_data

def display_results(extracted_data, plot=True):
    for i, data in enumerate(extracted_data):
        print(f"\nTable {i+1}:")
        
        table_data = []
        for method, text in data['texts'].items():
            table_data.append([method, text])
        
        print(tabulate(table_data, headers=['Method', 'Extracted Text'], tablefmt='grid'))

        if plot:
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(data['image'], cv2.COLOR_BGR2RGB))
            plt.title(f'Table {i+1} Image')
            plt.axis('off')
            plt.tight_layout()
            plt.show()

file_path = r'path/to/your/file.pdf'  # Can be PDF or image file that in supported format
extracted_data = extract_content(file_path)
display_results(extracted_data, plot=True)