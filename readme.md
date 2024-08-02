# Multi-Engine OCR for PDF Tables and Images

This project implements a multi-engine Optical Character Recognition (OCR) system for extracting text from tables in PDF documents and images. It utilizes three popular OCR engines: Tesseract, EasyOCR, and PaddleOCR, to provide comprehensive text extraction capabilities.

## Features

- Extracts text from PDF documents and images (PNG, JPG, JPEG, TIFF, BMP)
- Supports text detection on various background colors
- Advanced image preprocessing for improved OCR accuracy:
  - Grayscale conversion
  - Adaptive thresholding
  - Denoising
  - Dilation and erosion for noise removal
- Utilizes multiple OCR engines:
  - Tesseract (via pytesseract)
  - EasyOCR
  - PaddleOCR
- Supports GPU acceleration for EasyOCR and PaddleOCR
- Displays extracted text and original image regions

## Requirements

- Python 3.x
- pytesseract
- easyocr
- paddleocr
- pdf2image
- opencv-python (cv2)
- numpy
- matplotlib
- tabulate
- torch
- paddle
- Pillow (PIL)

You can install the required packages using pip:

```
pip install pytesseract easyocr paddleocr pdf2image opencv-python numpy matplotlib tabulate torch paddle Pillow
```

Note: You'll need to install Tesseract OCR separately and ensure it's in your system PATH.

## Setup

1. Clone this repository
2. Install the required packages
3. Download and install Tesseract OCR
4. Update the Tesseract path in the script if necessary:

```python
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
```

## Usage

1. Update the `file_path` variable with the path to your PDF or image file:

```python
file_path = r'path/to/your/file.pdf'  # or .png, .jpg, .jpeg, .tiff, .bmp
```

2. Run the script:

```
python ocr_pdf.py
```

The script will extract content from the PDF or image file, process it using all three OCR engines, and display the results.

## Supported File Formats

The script supports the following file formats:
- PDF (.pdf)
- PNG (.png)
- JPEG (.jpg, .jpeg)
- TIFF (.tiff)
- BMP (.bmp)

## Image Preprocessing

The script now includes advanced image preprocessing steps to improve OCR accuracy:

1. Grayscale conversion
2. Adaptive thresholding
3. Denoising using fastNlMeansDenoising
4. Dilation and erosion to remove noise

These steps help in detecting text on various background colors and improve overall text recognition.

## Output

The script outputs:
- Device information (CPU/GPU usage for each OCR engine)
- Extracted text from each detected region, separated by OCR engine
- Visual display of each extracted image region

## Customization

You can customize the OCR process by modifying the `extract_content` function arguments:

```python
extracted_data = extract_content(file_path, use_tesseract=True, use_easyocr=True, use_paddleocr=True)
```

Set any of the `use_*` parameters to `False` to disable a specific OCR engine.

## Error Handling

The script now includes error handling for each OCR engine. If an error occurs during processing with any engine, it will be caught and reported, allowing the script to continue with the other engines.

## Scripts

This repository contains the following scripts, showing the progression of the OCR capabilities:

1. `ocr_pdf.py`: 
   - Performs OCR on PDF files only.
   - Uses multiple OCR engines (Tesseract, EasyOCR, PaddleOCR).

2. `ocr_pdf_and_images.py`: 
   - Extends OCR capabilities to both PDF and image files (PNG, JPG, JPEG, TIFF, BMP).
   - Uses multiple OCR engines (Tesseract, EasyOCR, PaddleOCR).
   - Supports basic image processing.

3. `ocr_advanced.py`: 
   - The most advanced version of the OCR script.
   - Performs OCR on both PDF and image files.
   - Implements advanced image preprocessing techniques:
     - Grayscale conversion
     - Adaptive thresholding
     - Denoising
     - Dilation and erosion
   - Improves text detection on various background colors.
   - Uses multiple OCR engines with error handling.
   - This is the recommended script for most use cases.

To use a specific script, run:

```
python <script_name>.py
```

## Acknowledgements

This project uses the following open-source libraries:
- Tesseract OCR
- EasyOCR
- PaddleOCR
- PyTesseract
- pdf2image
- OpenCV
- NumPy
- Matplotlib
- Tabulate
- PyTorch
- PaddlePaddle
- Pillow (PIL)