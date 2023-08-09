from typing import Dict, Any

from consts import CROP_DIR, CROP_RESULT, SEQ, IS_TRUE, IGNOR, CROP_PATH, X0, X1, Y0, Y1, COLOR, SEQ_IMAG, COL, X, Y, \
    GTIM_PATH

from pandas import DataFrame
import cv2
import numpy as np


def make_big_crop(image, x, y, color):

    patch_width = 70
    patch_height = 210

    if color == 'r':
        patch_top_left_x = x - 20
        patch_top_left_y = y - 20
    else:
        patch_top_left_x = x - 20
        patch_top_left_y = y - 140 if y - 140 > 0 else 0

    big_crop = image[patch_top_left_y:patch_top_left_y + patch_height,
                      patch_top_left_x:patch_top_left_x + patch_width, :]

    return big_crop


def make_crop(image, x, y, color):
    """
    The function that creates the crops from the image.
    Your return values from here should be the coordinates of the crops in this format (x0, x1, y0, y1, crop content):
    'x0'  The bigger x value (the right corner)
    'x1'  The smaller x value (the left corner)
    'y0'  The smaller y value (the lower corner)
    'y1'  The bigger y value (the higher corner)
    """


    big_crop = make_big_crop(image, x, y, color)
    image = cv2.cvtColor(big_crop, cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, threshold1=30, threshold2=70)

    # Apply Hough Circle Transform
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=100, param2=30, minRadius=10,
                               maxRadius=50)

    diameter = circles[0][2] * 2
    crop_width = int(diameter * 1.4)
    crop_height = crop_width * 3

    if color == 'r':
        crop_top_left_x = x - 20
        crop_top_left_y = y - 20
    else:
        crop_top_left_x = x - 20
        crop_top_left_y = y - crop_width * 2 if y - crop_width * 2 > 0 else 0  # if the tfl go up of the image: y=0

    return crop_top_left_x, crop_top_left_x + crop_width, crop_top_left_y, crop_top_left_y + crop_height, 'crop_data'


def check_crop(*args, **kwargs):
    """
    Here you check if your crop contains a traffic light or not.
    Try using the ground truth to do that (Hint: easier than you think for the simple cases, and if you found a hard
    one, just ignore it for now :). )
    """
    return True, True


def create_crops(df: DataFrame) -> DataFrame:
    # Your goal in this part is to take the coordinates you have in the df, run on it, create crops from them, save them
    # in the 'data' folder, then check if crop you have found is correct (meaning the TFL is fully contained in the
    # crop) by comparing it to the ground truth and in the end right all the result data you have in the following
    # DataFrame (for doc about each field and its input, look at 'CROP_RESULT')
    #
    # *** IMPORTANT ***
    # All crops should be the same size or smaller!!!

    # creates a folder for you to save the crops in, recommended not must



    if not CROP_DIR.exists():
        CROP_DIR.mkdir()

    # For documentation about each key end what it means, click on 'CROP_RESULT' and see for each value what it means.
    # You wanna stick with this DataFrame structure because its output is the same as the input for the next stages.
    result_df = DataFrame(columns=CROP_RESULT)

    # A dict containing the row you want to insert into the result DataFrame.
    result_template: Dict[Any] = {SEQ: '', IS_TRUE: '', IGNOR: '', CROP_PATH: '', X0: '', X1: '', Y0: '', Y1: '',
                                  COL: ''}
    for index, row in df.iterrows():
        result_template[SEQ] = row[SEQ_IMAG]
        result_template[COL] = row[COLOR]

        # example code:
        # ******* rewrite ONLY FROM HERE *******

        x0, x1, y0, y1, crop = make_crop(df[X], df[Y], 'and_everything_else_you_need_here (and you need)')
        result_template[X0], result_template[X1], result_template[Y0], result_template[Y1] = x0, x1, y0, y1
        crop_path: str = '/data/crops/my_crop_unique_name.probably_containing_the original_image_name+somthing_unique'
        # crop.save(CROP_DIR / crop_path)
        result_template[CROP_PATH] = crop_path
        result_template[IS_TRUE], result_template[IGNOR] = check_crop(df[GTIM_PATH],
                                                                      crop,
                                                                      'everything_else_you_need_here')
        # ******* TO HERE *******

        # added to current row to the result DataFrame that will serve you as the input to part 2 B).
        result_df = result_df._append(result_template, ignore_index=True)
    return result_df
