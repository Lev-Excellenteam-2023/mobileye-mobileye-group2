import numpy as np
from matplotlib import pyplot as plt, image as mpimg
from scipy import signal as sg, ndimage


def show_image(image: mpimg.imread, title: str = '') -> None:
    """
    Show image on screen (using matplotlib).

    :param image: The image itself.
    :param title: Title of the image.
    """
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()


def low_pass(img: np.ndarray, kernel) -> mpimg.imread:
    """
    Run low pass kernel on image;

    param img: The image itself.
    param kernel: Kernel (2d array).
    return: Result of running the kernel on the image.
    """
    red_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    blue_channel = img[:, :, 2]

    # Perform correlation for each RGB channel using scipy.signal.convolve2d
    correlation_red = sg.correlate2d(red_channel, kernel, mode='same', boundary='symm')
    correlation_green = sg.correlate2d(green_channel, kernel, mode='same', boundary='symm')
    correlation_blue = sg.correlate2d(blue_channel, kernel, mode='same', boundary='symm')

    # Combine the RGB channels back into one image
    after_low_pass = np.stack((correlation_red, correlation_green, correlation_blue), axis=-1)
    # plt.imshow(after_low_pass)
    # plt.title('After_low_pass')
    # plt.axis('off')
    # plt.show()
    return after_low_pass


def filter_red(c_image):

    height, width, channels = c_image.shape

    for i in range(height):
        for j in range(width):
            red_value = c_image[i, j, 0]
            # green_value = c_image[i, j, 1]
            blue_value = c_image[i, j, 2]

            if red_value < 0.9 or blue_value > 0.7:
                c_image[i, j, 0] = 0
                c_image[i, j, 1] = 0
                c_image[i, j, 2] = 0
    return c_image


def filter_green(img):
    """
    Filter the green pixels by blackening all other pixels.

    param img: The image.
    return: filtered image.
    """
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
    return img


def process_red(c_image):

    # process red lights:
    red_lights = filter_red(c_image)
    # plt.imshow(red_lights)
    # plt.title('Red_lights_image')
    # plt.axis('off')
    # plt.show()

    # low pass:
    value = 1 / 625
    kernel_25x25 = np.ones((25, 25)) * value  # kernel for low pass

    after_low_pass = low_pass(red_lights, kernel_25x25)

    return after_low_pass


def process_green(img):

    # filter green pixels:
    green_lights = filter_green(img)
    # plt.imshow(green_lights)
    # plt.title('Green lights image')
    # plt.axis('off')
    # plt.show()

    # low pass:
    value = 1 / 625
    kernel_25x25 = np.ones((25, 25)) * value  # kernel for low pass

    after_low = low_pass(green_lights, kernel_25x25)

    return after_low


def process_image(img: np.ndarray):
    """
    Find green traffic light in image.
    """

    # plt.imshow(img)
    # plt.title('Original')
    # plt.show()

    height = img.shape[0]
    height = height // 2

    half_red = img[:height, :, 0]
    half_green = img[:height, :, 1]
    half_blue = img[:height, :, 2]

    half_img = np.stack((half_red, half_green, half_blue), axis=-1)
    half_img2 = half_img.copy()

    red_lights_image = process_red(half_img)
    green_lights_image = process_green(half_img2)

    return red_lights_image, green_lights_image


def find_coordinates_for_tfl(channel):

    for row in range(channel.shape[0]):
        for col in range(channel.shape[1]):
            if channel[row, col] > 0:
                channel[row, col] = 1

    labeled_pixels, num_labels = ndimage.label(channel)
    pixel_coordinates = [None] * num_labels

    for row in range(labeled_pixels.shape[0]):
        for col in range(labeled_pixels.shape[1]):
            label = labeled_pixels[row, col]
            if label > 0:
                if pixel_coordinates[label - 1] is None:
                    pixel_coordinates[label - 1] = [(row, col)]
                else:
                    pixel_coordinates[label - 1].append((row, col))

    # Filter coordinates to keep only those with more than 12 pixels
    filtered_coordinates = [coord[0] for coord in pixel_coordinates if len(coord) > 800]

    x_values = np.array([coordinate[1] + 13 for coordinate in filtered_coordinates]).tolist()
    y_values = np.array([coordinate[0] + 13 for coordinate in filtered_coordinates]).tolist()

    return x_values, y_values