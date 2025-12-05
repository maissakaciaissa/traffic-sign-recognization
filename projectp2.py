import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog, messagebox
# ===================================================
# THIS CODE LET ME CHOSE THE QUERY IMAGE THAT I WANT 
# TO FIND ITS MATCHING IMAGE IN THE WHOLE FOLDER
# ===================================================

# ===================================================
# STEP 1: Create the main GUI
# ===================================================

class FeatureMatchingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Feature Matching App")
        self.root.geometry("600x400")  # Increased window size
        self.root.configure(bg='#F0F0F0')  # Light gray background

        # Define paths (keep your existing paths)
        self.DATA_TRAIN_PATH = "C:/Users/Acer/Documents/UNI/MIV 1/TAI/TAI_PROJECT/TAI_project/dataset/DATA/train"
        self.REALIMAGE_TRAIN_PATH = "C:/Users/Acer/Documents/UNI/MIV 1/TAI/TAI_PROJECT/TAI_project/dataset/RealImage/train"

        # Variables to store file paths
        self.query_image_path = None
        self.train_folder_path = None

        # Create GUI elements with styling
        self.label = tk.Label(root, 
                         text="Feature Matching App", 
                         font=("Helvetica", 16, "bold"), 
                         bg='#F0F0F0', 
                         fg='#2C3E50')
        self.label.pack(pady=20)

        # Styled query button
        self.query_button = tk.Button(root, 
                                 text="📁 Select Query Image", 
                                 command=self.select_query_image,
                                 font=("Arial", 12, "bold"),
                                 bg='#4CAF50',  # Green color
                                 fg='white',
                                 activebackground='#45A049',
                                 activeforeground='white',
                                 padx=20,
                                 pady=10,
                                 borderwidth=0,
                                 relief='raised')
        self.query_button.pack(pady=10)

    # Query path label with updated styling
        self.query_path_label = tk.Label(root, 
                                    text="No image selected", 
                                    font=("Arial", 10), 
                                    bg='#F0F0F0', 
                                    fg='#3498DB')
        self.query_path_label.pack(pady=5)

    # Styled process button
        self.process_button = tk.Button(root, 
                                   text="🔍 Find Best Match", 
                                   command=self.find_best_match, 
                                   state=tk.DISABLED,
                                   font=("Arial", 12, "bold"),
                                   bg='#E74C3C',  # Red color
                                   fg='white',
                                   activebackground='#C0392B',
                                   activeforeground='white',
                                   padx=20,
                                   pady=10,
                                   borderwidth=0)
        self.process_button.pack(pady=20)

    # Add hover effects
        self.query_button.bind("<Enter>", lambda e: self.query_button.config(bg='#45A049'))
        self.query_button.bind("<Leave>", lambda e: self.query_button.config(bg='#4CAF50'))
        self.process_button.bind("<Enter>", lambda e: self.process_button.config(bg='#C0392B'))
        self.process_button.bind("<Leave>", lambda e: self.process_button.config(bg='#E74C3C'))

    # ===================================================
    # STEP 2: Function to select query image
    # ===================================================
    def select_query_image(self):
        self.query_image_path = filedialog.askopenfilename(
            title="Select Query Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
        )
        if self.query_image_path:
            self.query_path_label.config(text=f"Query Image: {os.path.basename(self.query_image_path)}", fg="green")
            self.determine_train_folder()
            self.process_button.config(state=tk.NORMAL)

    # ===================================================
    # STEP 3: Determine the corresponding train folder
    # ===================================================
    def determine_train_folder(self):
        # Extract the parent folder of the query image
        query_folder = os.path.dirname(self.query_image_path)

        # Check if the query image is in "DATA/QUERY" or "REALIMAGE/QUERY"
        if "DATA" in query_folder:
            self.train_folder_path = self.DATA_TRAIN_PATH
        elif "RealImage" in query_folder:
            self.train_folder_path = self.REALIMAGE_TRAIN_PATH
        else:
            messagebox.showerror("Error", "Invalid query folder structure. Expected 'DATA/QUERY' or 'REALIMAGE/QUERY'.")
            return

        print(f"Query Image: {self.query_image_path}")
        print(f"Train Folder: {self.train_folder_path}")

    # ===================================================
    # STEP 4: Function to find the best match in the train folder
    # ===================================================
    
    def find_best_match(self):
        try:
            # Read the query image
            img1 = cv2.imread(self.query_image_path, 0)  # Query image (grayscale)
            if img1 is None:
                raise ValueError("Failed to load the query image. Please check the file path.")

            # Get the list of images in the train folder
            train_images = self.create_image_path_list(self.train_folder_path)
            if not train_images:
                raise ValueError(f"No images found in the train folder: {self.train_folder_path}")

            # Initiate the SIFT detector
            sift = cv2.SIFT_create()

            # Find the keypoints and descriptors for the query image
            kp1, des1 = sift.detectAndCompute(img1, None)

            # Variables to store the best match
            best_match = None
            best_kp = None
            best_des = None
            best_good_matches = []
            threshold = 0.75  # Lowe's ratio test threshold
            iteration=0
            # Iterate through all images in the train folder
            for train_image_path in train_images:
                iteration=iteration+1
                print(iteration)
                img2 = cv2.imread(train_image_path, 0)  # Train image (grayscale)
                if img2 is None:
                    print(f"Failed to load train image: {train_image_path}")
                    continue

                # Find the keypoints and descriptors for the train image
                kp2, des2 = sift.detectAndCompute(img2, None)

                # Use Brute-Force Matcher to find matches
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)

                # Apply Lowe's ratio test to filter good matches
                good_matches = []
                for m, n in matches:
                    if m.distance < threshold * n.distance:
                        good_matches.append([m])

                # Update the best match if this image has more good matches
                if len(good_matches) > len(best_good_matches):
                    best_match = img2
                    best_kp = kp2
                    best_des = des2
                    best_good_matches = good_matches

            # Check if a best match was found
            if best_match is None:
                raise ValueError("No good matches found in the train folder.")

            # Draw the matches
            img3 = cv2.drawMatchesKnn(img1, kp1, best_match, best_kp, best_good_matches, None, flags=2)

            # Display the result in a new window
            plt.imshow(img3)
            plt.title("Best Match Result")
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ===================================================
    # STEP 5: Function to create a list of image paths in a folder
    # ===================================================
    def create_image_path_list(self, directory_path):
        image_path_list = []
        if not os.path.isdir(directory_path):
            print(f"The specified path '{directory_path}' is not a valid directory.")
            return image_path_list

        for filename in os.listdir(directory_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                image_path_list.append(os.path.join(directory_path, filename).replace('\\', '/'))

        return image_path_list

# ===================================================
# STEP 6: Run the application
# ===================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = FeatureMatchingApp(root)
    root.mainloop()