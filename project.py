import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
# Code from:
# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html

# Prepared for M1IV students, by Prof. Slimane Larabi
#===================================================
#and on this part is also modified to be accept a large query folder
# The code is modified for the specific database
#===================================================
# Function to get the names of the images from a folder in a list:
def create_image_path_list(directory_path):
    image_path_list = []
    if not os.path.isdir(directory_path):
        print(f"The specified path '{directory_path}' is not a valid directory.")
        return image_path_list

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_path_list.append(os.path.join(directory_path, filename).replace('\\', '/'))

    return image_path_list

# Function to match a group of images to another group of images and save the results
def match_images_group_to_group(query_folder, target_folder, output_folder):
    # Get list of query and target images
    query_images = create_image_path_list(query_folder)
    target_images = create_image_path_list(target_folder)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Threshold for Lowe's ratio test
    threshold = 0.75
    all_best_matches = []

    # Make sure the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through each query image
    for query_image_path in query_images:
        print(f"Processing query image: {query_image_path}")

        # Load the query image and compute keypoints and descriptors
        img1 = cv2.imread(query_image_path, 0)  # query image
        kp1, des1 = sift.detectAndCompute(img1, None)

        previous = 0  # Keep track of the number of good matches
        img_ideal = None  # Ideal matching image (best match)
        good_matches = []  # Store the good matches for the current query image

        # Iterate through each target image
        for target_image_path in target_images:
            img2 = cv2.imread(target_image_path, 0)  # target image
            kp2, des2 = sift.detectAndCompute(img2, None)
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)  # Find 2 nearest matches

            # Apply Lowe's ratio test
            good = []
            for m, n in matches:
                if m.distance < threshold * n.distance:
                    good.append([m])

            # If the number of good matches is the highest so far, store it
            if len(good) > previous:
                previous = len(good)
                img_ideal = img2
                kp_ideal = kp2
                des_ideal = des2
                good_matches = good

        # After processing all target images for this query image
        if img_ideal is not None:
            # Draw matches and save the image with key points drawn
            img3 = cv2.drawMatchesKnn(cv2.imread(query_image_path, 0), kp1, img_ideal, kp_ideal, good_matches, None, flags=2)

            # Get the filename of the query image and save the result to the output folder
            query_filename = os.path.basename(query_image_path)
            output_image_path = os.path.join(output_folder, f"matched_{query_filename}")

            # Save the result as an image file
            cv2.imwrite(output_image_path, img3)
            print(f"Saved matched result for {query_filename} to {output_image_path}")

# and save the results to the "output/matches" folder
match_images_group_to_group('dataset/query', 'dataset/test', 'output/matches')
