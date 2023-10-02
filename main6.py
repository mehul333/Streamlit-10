import os
import cv2
import numpy as np
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


def process_image(src_path, dst_path):
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

    # Save the masked image
    cv2.imwrite(os.path.join(dst_path, "masked_image.jpg"), masked)

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
    print(f"Total number of plants: {plants_number}")

    # Display and save results
    cv2.imshow('Processed Image', src)
    cv2.imwrite(os.path.join(dst_path, "result.jpg"), src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Create JSON data
    data = {
        "minH": MIN_GREEN_HUE,
        "maxH": MAX_GREEN_HUE,
        "minS": MIN_GREEN_SAT,
        "maxS": MAX_GREEN_SAT,
        "minV": MIN_GREEN_VAL,
        "maxV": MAX_GREEN_VAL,
        "erodeKernel": KERNEL_SIZE,
        "erodeIterations": ERODE_ITERATIONS,
        "dilateKernel": KERNEL_SIZE,
        "dilateIterations": DILATE_ITERATIONS,
        "plantsNumber": plants_number  # Add the total number of plants
    }

    update_json(src_path, dst_path, data)


def update_json(src_path, dst_path, data):
    src_imgname = os.path.basename(src_path)
    json_path = os.path.join(dst_path, "details.json")  # Updated path

    json_data = {
        "source": src_imgname,
        "images": {
            src_imgname: {
                "date": time.strftime("%Y-%m-%d", time.gmtime()),
                "operations": [
                    f"h {data['minH']}-{data['maxH']}",
                    f"s {data['minS']}-{data['maxS']}",
                    f"v {data['minV']}-{data['maxV']}",
                    f"erode {data['erodeKernel']} x{data['erodeIterations']}",
                    f"dilate {data['dilateKernel']} x{data['dilateIterations']}"
                ],
                # Add the total number of plants
                "plantsNumber": data["plantsNumber"]
            }
        }
    }

    if os.path.exists(json_path):
        with open(json_path, 'r') as infile:
            existing_data = json.load(infile)
            json_data.update(existing_data)

    with open(json_path, 'w') as outfile:
        json.dump(json_data, outfile, indent=2)


def main():
    src_path = "Images/6.jpg"
    dst_path = "hsv_results/6"

    ensure_dir(dst_path)
    process_image(src_path, dst_path)


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":
    main()
