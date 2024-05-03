#Import necessary libraries
from rembg import remove
import json
import os
import cv2
import numpy as np
import pytesseract
from difflib import SequenceMatcher
import time
import re


# Configure Tesseract path for OCR, Optical Character Recognition (OCR) tool, which can read and recognize text in images.
pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract' 

#Specify paths for storing extracted answers and answer keys
extracted_answers_path = 'outputs/extracted_answers' 
extracted_answerkey_path = 'outputs/extracted_answerkey'


# Load the exam image and remove its background
def LoadImage(image_path):
    exam_img = cv2.imread(image_path)
    rmvbg_exam_img = remove(exam_img)
    
    return rmvbg_exam_img
    
# Function to perform preliminary image processing
def Preprocess(rmvbg_img):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(rmvbg_img, cv2.COLOR_BGR2GRAY)

    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) 
    gray = clahe.apply(gray)

    #Apply Gaussian Blur to smooth the image
    gray = cv2.GaussianBlur(gray,(5, 5), 0) 
    return gray

# Function to detect corners of the exam paper using Harris Corner Detection
def HarrisCornerDetection(exam_img, gray_exam_img):
    
    alpha_channel = np.zeros(exam_img.shape[:2], dtype=exam_img.dtype)
    exam_img = cv2.merge((exam_img, alpha_channel))
    
    #Apply Harris COrner Detection
    dst = cv2.cornerHarris(gray_exam_img, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    threshold = 10e-3 * dst.max()
    #corners = exam_img.copy()
    #corners[dst > threshold] = [0, 0, 255] 
    for y, x in zip(*np.where(dst > threshold)):
        exam_img[y, x, :3] = [0, 0, 255]  # BGR color
        exam_img[y, x, 3] = 255  # Alpha channel set to opaque
    
    for y, x in zip(*np.where(dst > threshold)):
        cv2.circle(exam_img, (x, y), 5, (0, 255, 0, 255), -1)  # Add alpha value to color tuple
    return dst, threshold

# Function to rescale the exam paper for alignment and size normalization
def ReScale_ExamPaper(exam_img, dst, threshold, rmvbg_img):
    # Find extreme points and perform perspective transformation
    
    # Identify all points where the Harris Corner Detection response is greater than the threshold.
    # 'dst' is the output of the Harris Corner Detector.
    y, x = np.where(dst > threshold)
    
    # Determine the top-left corner of the exam paper by finding the minimum x and y coordinates.
    # This point has the smallest x and y values, representing the top-left corner visually.
    top_left = (min(x), min(y))
    
    # Determine the top-right corner by finding the maximum x value and the minimum y value.
    # This represents the corner at the top-right, having the highest x coordinate but the lowest y, making it visually top-right.
    top_right = (max(x), min(y))
    
    # Determine the bottom-left corner by finding the minimum x value and the maximum y value.
    # This point is visually at the bottom-left of the paper, having the smallest x but the highest y coordinate.
    bottom_left = (min(x), max(y))
    
    # Determine the bottom-right corner by finding the maximum x and y values.
    # This is the visually bottom-right corner, having the highest values of x and y coordinates.
    bottom_right = (max(x), max(y))
    
    # Array of source points derived from the detected corners. 
    # These points represent the corners of the detected exam paper in the image.
    src_pts = np.array([top_left, top_right, bottom_left, bottom_right], dtype=np.float32)
    
    # Destination points are defined for the perspective transformation. 
    # These points represent where we want the source points to be aligned in the output image.
    # This effectively maps the corners of the exam paper to the corners of the image frame,
    # assuming a rectangular shape of the exam paper.
    dst_pts = np.array([
        [0, 0],  # Top-left corner to (0,0)
        [rmvbg_img.shape[1], 0],  # Top-right corner to the top-right of the image frame
        [0, rmvbg_img.shape[0]],  # Bottom-left corner to the bottom-left of the image frame
        [rmvbg_img.shape[1], rmvbg_img.shape[0]]  # Bottom-right corner to the bottom-right of the image frame
    ], dtype=np.float32)
    
    # Compute the perspective transformation matrix.
    # This matrix is calculated using the source and destination points,
    # allowing us to perform the transformation that aligns the exam paper corners
    # with the corners of the image frame.
    perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Apply the perspective transformation to the image with no background.
    # This warps the image, correcting its perspective, so the exam paper appears perfectly rectangular,
    # aligned with the image frame. The size of the output image is specified to match that of the input.
    corrected_image = cv2.warpPerspective(rmvbg_img, perspective_matrix, (rmvbg_img.shape[1], rmvbg_img.shape[0]))
    return corrected_image


# Function to process the corrected paper for analysis by identifying contours of interest
def Inverse_corrected_paper(corrected_image):
    # Convert the corrected image to grayscale. This simplification helps in the subsequent thresholding step,
    # making it easier to distinguish between the foreground (text or drawings) and the background.
    gray_corrected = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
    
    # Apply binary thresholding with Otsu's method to convert the grayscale image to a binary image (black and white).
    # Otsu's method automatically determines the optimal threshold value to distinguish foreground from background.
    # The THRESH_BINARY_INV flag inverts the thresholding, making the foreground white and the background black,
    # which is necessary for the contour detection step.
    _, binary_corrected = cv2.threshold(gray_corrected, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours in the binary image. The function retrieves external contours (RETR_EXTERNAL) only, 
    # which is suitable for identifying distinct answer boxes or marked areas in the corrected paper.
    # The CHAIN_APPROX_NONE mode stores all of the contour points, providing precise outlines.
    contours, _ = cv2.findContours(binary_corrected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

# Function to locate the answer boxes on the exam paper.
def findContours(contours, extracted_answers_path):
    
    # Initialize a list to hold the position information (coordinates and dimensions) of detected answer boxes.
    answers_positions = []

    # Iterate over each contour found in the previous processing step.
    for contour in contours:
        # Calculate the area of the contour to filter out noise or irrelevant marks.
        area = cv2.contourArea(contour)
        if 1500 < area:  # Only consider contours with an area greater than 1500 to ensure they are significant enough to be answer boxes.
            # Calculate the contour's perimeter, which helps in approximating the shape.
            perimeter = cv2.arcLength(contour, True)
            # Approximate the contour shape to simpler polygons with fewer points. This is helpful in identifying rectangular shapes.
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4:  # Ensure the approximated shape has 4 sides, indicating it's likely a rectangle.
                # Obtain the bounding rectangle for the contour, which gives us the position and size.
                x, y, w, h = cv2.boundingRect(approx)
                # Calculate the aspect ratio to further ensure the contour is rectangular, resembling an answer box.
                aspect_ratio = float(w) / h
                if 0.9 < aspect_ratio:  # An aspect ratio check to filter out non-rectangular shapes.
                    # Add the detected answer box's position and size to the list.
                    answers_positions.append({'x': x, 'y': y, 'width': w, 'height': h})

    return answers_positions

# Function to store images of handwriting inside answer boxes on the exam paper and their coordinates.
def storeAnsBoxesCoordinates(corrected_image, answers_positions: np.ndarray, extracted_answers_path):

    # Check if the directory for storing extracted answers exists; if not, create it.
    # This ensures there's a place to save images of the handwriting found in each answer box.
    if not os.path.exists(extracted_answers_path):
        os.makedirs(extracted_answers_path)
        
    # Initialize a list to keep track of the minimum x and y coordinates of the answer boxes.
    # These coordinates will be used to identify and isolate each answer box in the corrected image.
    min_coordinates = []
    min_coor = []

    # Iterate through each identified answer box position.
    for i, pos in enumerate(answers_positions):
        # Adjust each box's coordinates inward by a small margin to focus on the content inside,
        # helping to avoid capturing parts of the box border.
        inner_x = pos['x'] + 6  # Adjust the x-coordinate inward.
        inner_y = pos['y'] + 5  # Adjust the y-coordinate inward.
        inner_w = pos['width'] - 15  # Adjust the width to exclude edges.
        inner_h = pos['height'] - 16  # Adjust the height to exclude edges.
        
        # Extract the adjusted area from the corrected image, which should contain only the handwriting.
        answer_img = corrected_image[inner_y:inner_y+inner_h, inner_x:inner_x+inner_w]
        # Save the extracted image of handwriting to a file for further processing or analysis.
        cv2.imwrite(f'{extracted_answers_path}/answer_{i+1}.png', answer_img)
        # Store the adjusted coordinates for future reference, potentially useful for mapping answers back to their locations.
        min_coordinates.append({'x': inner_x, 'y': inner_y})
        
    for i in range(1, len(min_coordinates)):
        min_coor.append(min_coordinates[i])

    # Save the coordinates of all extracted answer boxes to a JSON file,
    # allowing easy access to this information for subsequent steps in the processing pipeline.
    with open('outputs/answers_positions.json', 'w') as f:
        json.dump(min_coor, f)
        
    return min_coor


# Function to highlight answer boxes on the corrected exam paper image.
def Highlight_AnsBoxes(corrected_image, answers_positions: np.ndarray):
    # Check if the corrected image is grayscale. If so, convert it to BGR color to draw colored rectangles.
    # This step ensures that the highlighting can be visually recognized in color.
    if len(corrected_image.shape) == 2 or corrected_image.shape[2] == 1:
        corrected_image_color = cv2.cvtColor(corrected_image, cv2.COLOR_GRAY2BGR)
    else:
        # If the image is already in color, make a copy to avoid modifying the original image directly.
        corrected_image_color = corrected_image.copy()

    # Iterate through each position in the list of answer box positions.
    for pos in answers_positions:
        # Extract the x, y coordinates and the width and height for each answer box.
        x, y, width, height = pos['x'], pos['y'], pos['width'], pos['height']
        # Draw a green rectangle (with color code (0, 255, 0)) around the answer box.
        # The thickness of the rectangle's border is set to 2 pixels.
        cv2.rectangle(corrected_image_color, (x, y), (x + width, y + height), (0, 255, 0), 2)
    

# Function to recognize handwriting from images of answer boxes.
def RecognizeHandWriting(answers_positions: np.ndarray):
    recognized_answers = []  # Initialize a list to store recognized text from each answer box.
    rec_answers = []
    # Iterate through each answer box position to process the corresponding image.
    for i, pos in enumerate(answers_positions):
        # Construct the path to the image file for the current answer box.
        answer_img_path = f'outputs/extracted_answers/answer_{i+1}.png'

        # Load the image from the specified path.
        answer_img = cv2.imread(answer_img_path)

        # Convert the image to grayscale to simplify the image and focus on text.
        ans_gray = cv2.cvtColor(answer_img, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian Blur to reduce image noise and improve OCR accuracy.
        #ans_blur = cv2.medianBlur(ans_gray, 3)
        ans_blur = cv2.GaussianBlur(ans_gray, (3, 3), 0)

        # Apply thresholding to create a binary image, which enhances text for OCR.
        #ans_thresh = cv2.adaptiveThreshold(ans_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   #cv2.THRESH_BINARY, 11, 2)
        ans_thresh = cv2.threshold(ans_blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        
        # Use a kernel for morphological operations to clean up the binary image,
        # enhancing the separation of text from the background.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        opening = cv2.morphologyEx(ans_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        # Invert the colors of the processed image to match Tesseract's expected input.
        invert = 255 - opening
        
        # Perform OCR on the inverted image using Tesseract with a custom configuration.
        # The configuration is set to use the default OCR engine mode (oem 3) and assume a single block of text (psm 10).
        custom_config = r'--psm 10 --oem 3'
        text = pytesseract.image_to_string(invert, lang='eng', config=custom_config)
        text = text.strip('(),\n').replace(' ', '')
        recognized_answers.append(text)
            
    counter = 0
    for i in range(len(recognized_answers)):
        if recognized_answers[i] in ['T', 'F']:
            counter += 1
        
    if (counter / len(recognized_answers)) >= 0.5:
        recognized_answers[0] = '1'
     
    is_true_false = recognized_answers[0] == '1'
    is_multiple = recognized_answers[0] == '2'
    is_fillout = recognized_answers[0] == '3'
    is_fill = recognized_answers[0] == '4'
    is_math = recognized_answers[0] == '5'
    
    print(f'{recognized_answers[0]}')
    
    if is_true_false:
        # Adjust the following answers based on True-False logic
        print("Question Type: True/False")
        
        for i in range(1, len(recognized_answers)):
                
            if (len(recognized_answers[i]) > 1):
                recognized_answers[i] = '?'
                
            elif (not recognized_answers[i].isalpha()):
                recognized_answers[i] = 'X'
                
            elif recognized_answers[i] != 'T':
                recognized_answers[i] = 'F'
        
            #print(f"Answer Box {box_counter}: {recognized_answers[i]}")
            rec_answers.append(recognized_answers[i])
            
        box_counter = 0
        for count in range(len(rec_answers), 0, -1):
            print(f"Answer Box {box_counter + 1}: {recognized_answers[count]}")
            box_counter += 1

    elif is_multiple:
        print("Question Type: Multiple Choice")

        for i in range(1, len(recognized_answers)):
                
            if not recognized_answers[i].isalpha():
                recognized_answers[i] = 'X'
            elif recognized_answers[i] in ['H', '4']:
                recognized_answers[i] = 'A'
                
            elif recognized_answers[i] in ['8']:
                recognized_answers[i] = 'B'
                
            elif recognized_answers[i] in ['L', '(', 'e']:
                recognized_answers[i] = 'C'
                
            elif recognized_answers[i] in ['0', 'p', 'P']:
                recognized_answers[i] = 'D'
                
            elif recognized_answers[i] in ['F']:
                recognized_answers[i] = 'E'
        
            rec_answers.append(recognized_answers[i])
            
        box_counter = 0
        for count in range(len(rec_answers), 0, -1):
            print(f"Answer Box {box_counter + 1}: {recognized_answers[count]}")
            box_counter += 1
            
            if len(recognized_answers[i]) > 1:
                print(f"\nAnswer Box {i} needs double check.\n")
                
    elif is_fillout:
        print("Question Type: Simple Answer")
        
        for i in range(1, len(recognized_answers)):
                
            if recognized_answers[i] == '':
                recognized_answers[i] = 'X'

            recognized_answers[i] = recognized_answers[i].replace('O', '0').replace('o', '0').replace('|', '1').replace('$', '5')
        
            rec_answers.append(recognized_answers[i])
            
        box_counter = 0
        for count in range(len(rec_answers), 0, -1):
            print(f"Answer Box {box_counter + 1}: {recognized_answers[count]}")
            box_counter += 1
            
    elif is_fill:
        print("Question Type: Fill in the Blanks")

        for i in range(1, len(recognized_answers)):
                
            if recognized_answers[i] == '':
                recognized_answers[i] = 'X'        
        
            rec_answers.append(recognized_answers[i])
            
        box_counter = 0
        for count in range(len(rec_answers), 0, -1):
            print(f"Answer Box {box_counter + 1}: {recognized_answers[count]}")
            box_counter += 1

    elif is_math:
        print("Question Type: Mathematical Problem")
        
        for i in range(1, len(recognized_answers)):
                
            if recognized_answers[i] == '':
                recognized_answers[i] = 'X'        
               
            recognized_answers[i] = recognized_answers[i].replace('o', '0').replace('O', '0').replace('|', '1').replace('$', '5')
            recognized_answers[i] = recognized_answers[i].replace('B', '8').replace('t', '7').replace('T', '7').replace('g', '9')
            recognized_answers[i] = recognized_answers[i].replace('q', '9').replace('a', '2').replace('z', '2').replace('Z', '2')
            recognized_answers[i] = recognized_answers[i].replace('S', '5').replace('s', '5').replace('/', '1').replace('[', '1')
            recognized_answers[i] = recognized_answers[i].replace('¢', '5').replace('G', '6').replace('A', '4').replace('b', '6')
            recognized_answers[i] = recognized_answers[i].replace('I', '1').replace('§', '8').replace('&', '8').replace('Q', '9')
            
            recognized_answers[i] = re.sub(r'\.(?!\d)', '', recognized_answers[i])
            
            rec_answers.append(recognized_answers[i])
            
        box_counter = 0
        for count in range(len(rec_answers), 0, -1):
            print(f"Answer Box {box_counter + 1}: {recognized_answers[count]}")
            box_counter += 1

                    
    with open('outputs/recognized_answers.json', 'w') as f:
        json.dump(rec_answers, f)
    
    return rec_answers


# Function to recognize handwriting from images of answer boxes.
def RecognizeAnsHandWriting(answers: np.ndarray):

    with open('outputs/recognized_answerkey.json', 'w') as f:
        json.dump(answers, f)
    
    print("Answer uploaded successfully.")

# Function to score the exam based on the answers provided and mark them on the corrected image.
def scoredaexam(ans: np.ndarray, corrected_image):
    
    # Assume corrected_image initially has 3 channels (BGR)
    if corrected_image.shape[2] == 3:
        # Add an alpha channel
        alpha_channel = np.zeros(corrected_image.shape[:2], dtype=corrected_image.dtype)  # Create a new alpha channel filled with zeros (fully transparent)
        corrected_image = cv2.merge((corrected_image, alpha_channel))  # Add the alpha channel to the image

    total = len(ans)
    score = 0.0
    ans_list = []

    with open('outputs/answers_positions.json') as f:
        d = json.load(f)
        
    with open('outputs/recognized_answers.json') as f:
        d_ans = json.load(f)
        
    with open('outputs/recognized_answerkey.json') as f:
        d_key = json.load(f)
    
    
    marker_counter = 0
    # Creating a temporary image to draw text on for alpha manipulation
    temp_image = np.zeros_like(corrected_image)

    for answer, key in zip(d_ans, d_key):
        if answer.lower() == key.lower().replace(" ", ""):
            score += (100 / total)
            ans_list.append(1)
        elif (answer.lower() != key.lower()) and (SequenceMatcher(None, answer, key).ratio() > 0.9):
            ans_list.append(2)
        elif answer == '?':
            ans_list.append(2)
        else:
            ans_list.append(0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (0, 0, 255)  # BGR for red
    thickness = 2

    # Process text drawing
    for pos, ans_flag, key in zip(d, ans_list, d_key):
        org_x, org_y = int(pos['x']), int(pos['y'])
        org = (org_x, org_y - 10)
        text = f'+{format(100/total, ".2f")}' if ans_flag == 1 else 'Double check.' if ans_flag == 2 else f'X, Ans: {key.strip()}'
        cv2.putText(temp_image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)  # Draw on temp_image
        cv2.putText(corrected_image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)  # Draw on corrected_image
        marker_counter += 1
        
    score_text = f'Score: {format(score, ".2f")}/100'

    org = (400, 50)
    cv2.putText(temp_image, score_text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(corrected_image, score_text, org, font, fontScale, color, thickness, cv2.LINE_AA)

    # Use the temporary image to update the alpha channel where text has been drawn
    gray_temp_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_temp_image, 1, 255, cv2.THRESH_BINARY)
    corrected_image[:, :, 3][mask == 255] = 255  # Set alpha to 255 where text is visible
    
    print(f'Final Score: {format(score, ".2f")}/100')
    
    box_counter = 0
    for count in range(len(ans_list) - 1, -1, -1):
        if ans_list[count] == 2:
            print(f"Answer Box {box_counter + 1} needs a double check")
            box_counter += 1

    cv2.imwrite('outputs/scored_exam_paper.png', corrected_image)

    
def main():
    tic = time.perf_counter()
    
    answer_key = ["x>2", "7^5", "3582", "30°", "60%", "5","4", "7", "5", "8", "3", 
                  "3", "3", "4", "2", "4", "3", "4.33", "4", "7", "4", "7", "5",
                  "6", "5", "3", "9", "18", "14", "8", "8", "9", "10", "6", "42",
                  "18", "13", "9", "15", "9", "16", "4", "24", "8", "8"]
    
    box_counter = 0
    for count in range(len(answer_key)- 1, -1, -1):
        print(f"Answer Key {box_counter + 1}: {answer_key[count]}")
        box_counter += 1

    # Process flow for exam image
    exam_img = cv2.imread('test_50_3.jpg')
    rmvbg_exam_img = remove(exam_img)
    
    gray_exam_img = Preprocess(rmvbg_exam_img)  
    
    dst, threshold = HarrisCornerDetection(exam_img, gray_exam_img)
    
    corrected_image = ReScale_ExamPaper(exam_img, dst, threshold, rmvbg_exam_img)
    
    contours = Inverse_corrected_paper(corrected_image)
    
    answers_positions = findContours(contours, extracted_answers_path)
    
    ans_coor = storeAnsBoxesCoordinates(corrected_image, answers_positions, extracted_answers_path)
    
    Highlight_AnsBoxes(corrected_image, answers_positions)
    
    ans = RecognizeHandWriting(answers_positions)
            
    # Score the exam based on recognized answers and answer key
    scoredaexam(ans, corrected_image)
    
    toc = time.perf_counter()
    
    print(f"It took {toc - tic:0.2f} seconds for grading")
    
if __name__ == "__main__":
    main()