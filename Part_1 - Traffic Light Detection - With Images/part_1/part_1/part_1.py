import cv2

from typing import List, Optional, Union, Dict, Tuple
import json
import argparse
from pathlib import Path

import numpy as np
from scipy import signal as sg
import scipy.ndimage as ndimage
from scipy.ndimage import maximum_filter
from PIL import Image
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

# if you wanna iterate over multiple files and json, the default source folder name is this.
DEFAULT_BASE_DIR: str = 'C:/Users/User/Desktop/exellenteam/mobileye-group2/Part_1 - Traffic Light Detection - With Images/part_1/part_1'

# The label we wanna look for in the polygons json file
TFL_LABEL = ['traffic light']

POLYGON_OBJECT = Dict[str, Union[str, List[int]]]
RED_X_COORDINATES = List[int]
RED_Y_COORDINATES = List[int]
GREEN_X_COORDINATES = List[int]
GREEN_Y_COORDINATES = List[int]


def find_tfl_lights1(c_image: np.ndarray,
                    **kargs) -> Tuple[RED_X_COORDINATES, RED_Y_COORDINATES, GREEN_X_COORDINATES, GREEN_Y_COORDINATES]:
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement.

    :param c_image: The image itself as np.uint8, shape of (H, W, 3).
    :param kwargs: Whatever config you want to pass in here.
    :return: 4-tuple of x_red, y_red, x_green, y_green.
    """
    ### WRITE YOUR CODE HERE ###
    ### USE HELPER FUNCTIONS ###

    plt.imshow(c_image)
    plt.show()
#=============================================
    def extract_crop(image, x, y):
        patch_width = 70
        patch_height = 210
        patch_top_left_x = x - 20
        patch_top_left_y = y - 140 if y - 140 > 0 else 0

        extracted_patch = image[patch_top_left_y:patch_top_left_y + patch_height,
                          patch_top_left_x:patch_top_left_x + patch_width, :]

        return extracted_patch

    # Load the image
    # image_path = 'C:/Users/User/Desktop/exellenteam/mobileye-group2/Part_1 - Traffic Light Detection - With Images/part_1/part_1/stuttgart_000109_000019_leftImg8bit.png'
    #image = mpimg.imread(c_image)

    # Specify the pixel coordinates
    pixel_x = 1800  # Example x-coordinate
    pixel_y = 130  # Example y-coordinate

    # Extract the desired RGB patch
    rgb_patch = extract_crop(c_image, pixel_x, pixel_y)

    # Display the extracted RGB patch using matplotlib
    plt.imshow(rgb_patch)

    plt.title('Extracted RGB Patch')
    plt.axis('off')
    plt.show()

#===============================================================
    # Load the image
    image = cv2.cvtColor(rgb_patch, cv2.COLOR_RGB2BGR)

    #image = cv2.imread(rgb_patch, cv2.IMREAD_COLOR)

    # Convert the image to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, threshold1=30, threshold2=70)

    # Apply Hough Circle Transform
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=100, param2=30, minRadius=10,
                               maxRadius=50)

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for circle in circles[0, :]:
            center = (circle[0], circle[1])

            # Retrieve pixel value at the center
            pixel_value = image[circle[1], circle[0]]

            print(f"Center: {center}, Pixel Value: {pixel_value}, Radius: {circle[2]}")
    else:
        print("No circles detected.")

    # Display the image with detected circles
    plt.imshow(image)
    plt.plot(circle[0], circle[1], 'rx', markersize=6)
    plt.show()
    # cv2.imshow('Detected Circles', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #=============================================
    height, width, channels = c_image.shape

    for i in range(height):
        for j in range(width):
            red_value = c_image[i, j, 0]
            green_value = c_image[i, j, 1]
            blue_value = c_image[i, j, 2]

            if red_value > 0.6*255 or green_value < 0.8*255:
                c_image[i, j, 0] = 0
                c_image[i, j, 1] = 0
                c_image[i, j, 2] = 0

    plt.imshow(c_image)
    plt.title('Result')
    plt.axis('off')
    plt.show()

    red_channel = c_image[:, :, 0] / 255.0
    green_channel = c_image[:, :, 1] / 255.0
    blue_channel = c_image[:, :, 2] / 255.0

    value = 1 / 625
    matrix_25x25 = np.ones((25, 25)) * value
    kernel = matrix_25x25

    # Perform correlation for each RGB channel using scipy.signal.convolve2d
    correlation_red = sg.correlate2d(red_channel, kernel, mode='same', boundary='symm')
    correlation_green = sg.correlate2d(green_channel, kernel, mode='same', boundary='symm')
    correlation_blue = sg.correlate2d(blue_channel, kernel, mode='same', boundary='symm')

    # Combine the RGB channels back into one image
    after_low_pass = np.stack((correlation_red, correlation_green, correlation_blue), axis=-1)

    plt.imshow(after_low_pass)
    plt.title('After low pass')
    plt.axis('off')
    plt.show()

    for row in range(correlation_green.shape[0]):
        for col in range(correlation_green.shape[1]):
            if correlation_green[row, col] > 0:
                correlation_green[row, col] = 1

    labeled_red, num_labels = ndimage.label(correlation_green)
    pixel_coordinates = [None] * num_labels

    for row in range(labeled_red.shape[0]):
        for col in range(labeled_red.shape[1]):
            label = labeled_red[row, col]
            if label > 0 and pixel_coordinates[label - 1] is None:
                pixel_coordinates[label - 1] = (row, col)

    print("Pixel Coordinates:")
    for label, coordinate in enumerate(pixel_coordinates):
        print("Label", label + 1, ":", coordinate)

    x_values = np.array([coordinate[1]+13 for coordinate in pixel_coordinates])
    y_values = np.array([coordinate[0]+13 for coordinate in pixel_coordinates])

    # return [500, 700, 900], [500, 550, 600], [600, 800], [400, 300]
    return [600, 800], [400, 300], x_values, y_values


### GIVEN CODE TO TEST YOUR IMPLENTATION AND PLOT THE PICTURES
def show_image_and_gt(c_image: np.ndarray, objects: Optional[List[POLYGON_OBJECT]], fig_num: int = None):
    # ensure a fresh canvas for plotting the image and objects.
    plt.figure(fig_num).clf()
    # displays the input image.
    plt.imshow(c_image)
    labels = set()
    if objects:
        for image_object in objects:
            # Extract the 'polygon' array from the image object
            poly: np.array = np.array(image_object['polygon'])
            # Use advanced indexing to create a closed polygon array
            # The modulo operation ensures that the array is indexed circularly, closing the polygon
            polygon_array = poly[np.arange(len(poly)) % len(poly)]
            # gets the x coordinates (first column -> 0) anf y coordinates (second column -> 1)
            x_coordinates, y_coordinates = polygon_array[:, 0], polygon_array[:, 1]
            color = 'r'
            plt.plot(x_coordinates, y_coordinates, color, label=image_object['label'])
            labels.add(image_object['label'])
        if 1 < len(labels):
            # The legend provides a visual representation of the labels associated with the plotted objects.
            # It helps in distinguishing different objects in the plot based on their labels.
            plt.legend()


def test_find_tfl_lights(image_path: str, image_json_path: Optional[str]=None, fig_num=None):
    """
    Run the attention code.
    """
    # using pillow to load the image
    image: Image = Image.open(image_path)
    # converting the image to a numpy ndarray array
    c_image: np.ndarray = np.array(image)

    # my code---------------------------------------------------------

    # c_image: np.ndarray = np.array(image)
    # ----------------------------------------------------------------------------

    objects = None
    if image_json_path:
        image_json = json.load(Path(image_json_path).open())
        objects: List[POLYGON_OBJECT] = [image_object for image_object in image_json['objects']
                                         if image_object['label'] in TFL_LABEL]

    show_image_and_gt(c_image, objects, fig_num)

    red_x, red_y, green_x, green_y = find_tfl_lights1(c_image)

    c_image: np.ndarray = np.array(image)
    plt.imshow(c_image)

    # 'ro': This specifies the format string. 'r' represents the color red, and 'o' represents circles as markers.
    plt.plot(red_x, red_y, 'rx', markersize=6)
    plt.plot(green_x, green_y, 'g+', markersize=8)


def main(argv=None):
    """
    It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually examine the results.
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module.

    :param argv: In case you want to programmatically run this.
    """

    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to image json file -> GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)

    # If you entered a custom dir to run from or the default dir exist in your project then:
    directory_path: Path = Path(args.dir or DEFAULT_BASE_DIR)
    if directory_path.exists():
        # gets a list of all the files in the directory that ends with "_leftImg8bit.png".
        file_list: List[Path] = list(directory_path.glob('*_leftImg8bit.png'))

        for image in file_list:
            # Convert the Path object to a string using as_posix() method
            image_path: str = image.as_posix()
            path: Optional[str] = image_path.replace('_leftImg8bit.png', '_gtFine_polygons.json')
            image_json_path: Optional[str] = path if Path(path).exists() else None
            test_find_tfl_lights(image_path, image_json_path)

    if args.image and args.json:
        test_find_tfl_lights(args.image, args.json)
    elif args.image:
        test_find_tfl_lights(args.image)
    plt.show(block=True)


if __name__ == '__main__':
    main()
