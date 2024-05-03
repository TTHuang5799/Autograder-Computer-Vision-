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
