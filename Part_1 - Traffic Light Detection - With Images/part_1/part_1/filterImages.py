import os
import json
import shutil

# Source directory containing images and JSON files
SOURCE_DIRECTORY = 'C:/BootCamp2023/Mobily/Part_1 - Traffic Light Detection - With Images/images_set'

# Destination directory to copy images with "traffic light" label
DESTINATION_DIRECTORY = 'C:/BootCamp2023/Mobily/Part_1 - Traffic Light Detection - With Images/part_1/part_1/images_with_trl'


def filter_images():
    """
    Iterate through all the josn file in the DB, and copy to destination directory
    all the images that contain traffic-lights.
    Image contain traffic-light if in it jxon file there is label "traffic light".
    """
    # Loop through files in the source directory
    for json_filename in os.listdir(SOURCE_DIRECTORY):
        if json_filename.endswith('_gtFine_polygons.json'):
            # Construct the corresponding PNG filename
            base_filename = json_filename.replace('_gtFine_polygons.json', '_leftImg8bit.png')
            png_filename = os.path.join(SOURCE_DIRECTORY, base_filename)

            # Load JSON data from the file
            with open(os.path.join(SOURCE_DIRECTORY, json_filename), 'r') as json_file:
                data = json.load(json_file)

            # Check if any object has "traffic light" label
            has_traffic_light = any(item['label'] == 'traffic light' for item in data['objects'])

            if has_traffic_light:
                destination_image_path = os.path.join(DESTINATION_DIRECTORY, base_filename)

                # Copy the image to the destination directory
                shutil.copy(png_filename, destination_image_path)


def main():
    filter_images()


if __name__ == "__main__":
    main()

