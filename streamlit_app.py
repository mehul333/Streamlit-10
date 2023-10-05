import os
import cv2
import numpy as np
import streamlit as st
import json
import time

# Constants
MIN_GREEN_HUE = 45
MAX_GREEN_HUE = 77
MIN_GREEN_SAT = 19
MAX_GREEN_SAT = 255
MIN_GREEN_VAL = 164
MAX_GREEN_VAL = 255

KERNEL_SIZE = (2, 2)
ERODE_ITERATIONS = 2
DILATE_ITERATIONS = 4
AREA_THRESHOLD = 500  # Minimum contour area to consider as a plant

st.set_page_config(page_title="Plant Counter", page_icon=":seedling:")

# Add this at the beginning of the code
TEMP_DIR = "temp_dir"

def main():
    st.title("Plant Counting and Analysis App")

    # Create the temporary directory if it doesn't exist
    ensure_temp_dir_exists()

    # Allow the user to choose between uploading an image or using an example
    option = st.radio("Choose an option:", ["Upload an image", "Use an example"])

    if option == "Upload an image":
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "mp4"])
        if uploaded_image is not None:
            # Get the file name
            file_name = uploaded_image.name
            temp_image_path = os.path.join(TEMP_DIR, file_name)
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_image.read())
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            process_image(temp_image_path)
    else:
        # Provide example options for users to choose from
        example_option = st.selectbox("Choose an example:", ["Example 1", "Example 2", "Example 3"])

        example_image_path = ""

        if example_option == "Example 1":
            example_image_path == "D:\XBA\venv\1.jpg"
        elif example_option == "Example 2":
            example_image_path = "D:\XBA\venv\WhatsApp Image 2023-09-21 at 12.18.26.jpg"
        else:
            example_image_path = "example_images/example3.jpg"

        st.image(example_image_path, caption="Example Image", use_column_width=True)
        process_image(example_image_path)

def ensure_temp_dir_exists():
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

def process_image(src_path):
        # Load the image
    src = cv2.imread(src_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale image to HSV color space
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for green color
    lower_green = np.array([MIN_GREEN_HUE, MIN_GREEN_SAT, MIN_GREEN_VAL])
    upper_green = np.array([MAX_GREEN_HUE, MAX_GREEN_SAT, MAX_GREEN_VAL])

    # Create a mask for green color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Create a kernel for erosion and dilation
    kernel = np.ones(KERNEL_SIZE, np.uint8)

    # Apply morphological operations
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    masked = cv2.threshold(masked, 5, 255, cv2.THRESH_BINARY)[1]
    masked = cv2.erode(masked, kernel, iterations=ERODE_ITERATIONS)
    masked = cv2.dilate(masked, kernel, iterations=DILATE_ITERATIONS)

    # Find contours
    contours, _ = cv2.findContours(
        masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process and count the contours
    plants_number = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > AREA_THRESHOLD:
            plants_number += 1
            cv2.drawContours(src, [contour], -1,
                             (0, 255, 0), 2)  # Green contour

    # Print the total number of plants
    st.write(f"Total number of plants: {plants_number}")

    # Display and save results
    st.image(src, caption="Processed Image", use_column_width=True)

    # Clean up the temporary directory
    clean_temp_dir()

def clean_temp_dir():
    temp_dir = "temp_dir"
    if os.path.exists(temp_dir):
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)
        os.rmdir(temp_dir)
    # Rest of your image processing code here...

if __name__ == "__main__":
    main()
