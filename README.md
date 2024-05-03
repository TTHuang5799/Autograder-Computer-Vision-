# Exam Paper Analysis Tool

This tool automates the process of analyzing scanned images of exam papers. It utilizes image processing techniques to detect, read, and analyze answers written on the exam. This project leverages optical character recognition (OCR) to convert images into editable and searchable text, which is then used for further analysis.

## Features

- Background removal from images.
- Image preprocessing for enhanced OCR accuracy.
- Harris Corner Detection to identify the corners of the exam paper.
- Perspective transformation for image alignment.
- Answer box detection and analysis.
- Handwriting recognition and interpretation.

## Prerequisites

Before you can run this project, you need to have Python installed on your system. The project is tested with Python 3.9 and higher. You also need `pip` for installing Python packages.

## Setup and Installation

Follow these steps to set up the project on your local machine:

### 1. Clone the Repository
```bash
git clone https://github.com/TTHuang5799/Autograder-Computer-Vision-.git
cd Autograder-Computer-Vision-
```

### 2.  Install Required Python Libraries
For macOS, you can install the necessary libraries using pip and brew. If you are using another OS, adjust the installation commands accordingly.
```bash
pip install rembg
pip install opencv-python
pip install numpy
pip install pytesseract
brew install tesseract
```
After installing, configure the path to the Tesseract executable in your script if necessary:
pytesseract.pytesseract.tesseract_cmd = r'/path/to/tesseract'

### 3. Run the Script
Navigate to the script's directory and run:
``` bash
python 766Project.py
```
### 4. Usage
Once the installation is complete, you can run the tool by executing the Python script provided in the repository. Ensure that the images of the exam papers are placed in the correct directory as expected by the script.

## Methodology Overview

1. **Background Removal**: 
   - Using `rembg`, the tool first removes the background from the scanned images of exam papers. This helps in reducing noise and focusing the analysis on the relevant content.

2. **Image Preprocessing**:
   - Converts the image to grayscale to reduce complexity.
   - Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement, making details clearer and more distinct.
   - Applies Gaussian Blur to smooth the image, helping in reducing noise and improving the effectiveness of subsequent image processing steps.

3. **Harris Corner Detection**:
   - Utilizes Harris Corner Detection to identify the corners of the exam paper. This is crucial for the accurate perspective transformation of the image.
   - Marks detected corners with a specific color to visually confirm the locations of these corners.

4. **Perspective Transformation**:
   - Calculates a perspective transformation matrix using the corners detected. This matrix helps in warping the image to a standardized orientation and scale, which is essential for consistent OCR results.

5. **Contours Detection**:
   - Applies thresholding methods to convert the preprocessed image into a binary format, facilitating contour detection.
   - Detects contours which are likely candidates for answer boxes based on their geometric characteristics and area.

6. **Answer Box Localization and Analysis**:
   - Identifies and isolates answer boxes from the exam paper using the detected contours.
   - Adjusts the coordinates of these boxes slightly inward to focus on the content, avoiding the edges which might not be relevant.

7. **OCR for Text Recognition**:
   - Applies OCR to each identified answer box to convert the handwritten answers into digital text. This step uses Tesseract, a popular OCR engine, configured to optimize text recognition.

8. **Results Interpretation**:
   - Analyzes the recognized text to determine the correctness of answers based on a provided answer key.
   - Provides a scoring mechanism based on the number of correct answers, which is presented along with the processed image.

## Usage

This tool requires users to follow a few simple steps:
- Prepare the scanned images of exam papers.
- Ensure all dependencies are installed and paths are correctly configured.
- Run the tool to process the images and obtain analyzed results, including scores and highlighted answer boxes.

## Project Website
https://thuang293.wixsite.com/teachers-pet-smart
