from typing import List, Optional, Union, Dict, Tuple
import json
import argparse
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt, image as mpimg
from scipy import signal as sg
from scipy.ndimage import maximum_filter
from PIL import Image
import scipy.signal

# if you wanna iterate over multiple files and json, the default source folder name is this.
DEFAULT_BASE_DIR: str = r'C:\BootCamp2023\Mobily\Part_1 - Traffic Light Detection - With Images\myImages'

# The label we wanna look for in the polygons json file
TFL_LABEL = ['traffic light']

POLYGON_OBJECT = Dict[str, Union[str, List[int]]]
RED_X_COORDINATES = List[int]
RED_Y_COORDINATES = List[int]
GREEN_X_COORDINATES = List[int]
GREEN_Y_COORDINATES = List[int]


def find_tfl_lights(c_image: np.ndarray,
                    **kwargs) -> Tuple[RED_X_COORDINATES, RED_Y_COORDINATES, GREEN_X_COORDINATES, GREEN_Y_COORDINATES]:
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement.

    :param c_image: The image itself as np.uint8, shape of (H, W, 3).
    :param kwargs: Whatever config you want to pass in here.
    :return: 4-tuple of x_red, y_red, x_green, y_green.
    """

    ### WRITE YOUR CODE HERE ###
    ### USE HELPER FUNCTIONS ###
    return [500, 700, 900], [500, 550, 600], [600, 800], [400, 300]


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

    # //////////////////////////////////////////////////////////

    value = 1 / 289
    matrix_17x17_high = np.ones((17, 17)) * (-value)
    matrix_17x17_high[8, 8] = 288 / 289

    matrix_17x17_low = np.ones((17, 17)) * value

    val = 0.999
    mat_traffic = np.ones((15, 15)) * (val)
    mat_traffic[0:3, :] = 0
    mat_traffic[-3:, :] = 0
    mat_traffic[:, 0:3] = 0
    mat_traffic[:, -3:] = 0

    print(mat_traffic)

    val2 = 0.111
    mat_traffic2 = np.ones((15, 15)) * (val2)
    mat_traffic2[0:3, :] = 0
    mat_traffic2[-3:, :] = 0
    mat_traffic2[:, 0:3] = 0
    mat_traffic2[:, -3:] = 0

    img = mpimg.imread(
        'C:/BootCamp2023/Mobily/Part_1 - Traffic Light Detection - With Images/myImages/aachen_000059_000019_leftImg8bit.png')

    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    plt.show()


    # filter green pixels:
    height, width, channels = img.shape

    for i in range(height):
        for j in range(width):
            red_value = img[i, j, 0]
            green_value = img[i, j, 1]
            blue_value = img[i, j, 2]

            if red_value > 0.6 or green_value < 0.8:
                img[i, j, 0] = 0
                img[i, j, 1] = 0
                img[i, j, 2] = 0

    plt.imshow(img)
    plt.title('white to black')
    plt.axis('off')
    plt.show()

    # low pass:
    red_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    blue_channel = img[:, :, 2]

    # Perform correlation for each RGB channel using scipy.signal.convolve2d
    correlation_red1 = scipy.signal.correlate2d(red_channel, matrix_17x17_low, mode='same', boundary='symm')
    correlation_green1 = scipy.signal.correlate2d(green_channel, matrix_17x17_low, mode='same', boundary='symm')
    correlation_blue1 = scipy.signal.correlate2d(blue_channel, matrix_17x17_low, mode='same', boundary='symm')

    # Combine the RGB channels back into one image
    correlation_result_low = np.stack((correlation_red1, correlation_green1, correlation_blue1), axis=-1)

    plt.imshow(correlation_result_low)
    plt.title('low')
    plt.axis('off')
    plt.show()

    # //////////////////////////////////////////////////////////


    objects = None
    if image_json_path:
        image_json = json.load(Path(image_json_path).open())
        objects: List[POLYGON_OBJECT] = [image_object for image_object in image_json['objects']
                                         if image_object['label'] in TFL_LABEL]

    show_image_and_gt(c_image, objects, fig_num)

    red_x, red_y, green_x, green_y = find_tfl_lights(c_image)
    # 'ro': This specifies the format string. 'r' represents the color red, and 'o' represents circles as markers.
    plt.plot(red_x, red_y, 'ro', markersize=4)
    plt.plot(green_x, green_y, 'go', markersize=4)


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
