# Import necessary libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
# ===================================================
# IN THIS ONE I GIVE IT TWO IMAGED AND IT FINDS THE 
# REGION IN WHICH THEY ARE SIMILAR
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

      # Variables to store file paths
      self.query_image_path = None
      self.train_image_path = None

    # Create GUI elements with modern styling
      self.label = tk.Label(root, 
                         text="Feature Matching App", 
                         font=("Helvetica", 16, "bold"), 
                         bg='#F0F0F0', 
                         fg='#2C3E50')  # Dark blue text
      self.label.pack(pady=20)

    # Button to select query image
      self.query_button = tk.Button(root, 
                                 text="📁 Select Query Image", 
                                 command=self.select_query_image,
                                 font=("Arial", 12, "bold"),
                                 bg='#4CAF50',  # Green color
                                 fg='white',
                                 activebackground='#45A049',  # Darker green on click
                                 activeforeground='white',
                                 padx=20,
                                 pady=10,
                                 borderwidth=0,
                                 relief='raised')
      self.query_button.pack(pady=10)

    # Label to display query image path
      self.query_path_label = tk.Label(root, 
                                    text="Query Image: Not Selected", 
                                    font=("Arial", 10), 
                                    bg='#F0F0F0', 
                                    fg='#3498DB')  # Blue text
      self.query_path_label.pack(pady=5)

    # Button to select train image
      self.train_button = tk.Button(root, 
                                 text="📂 Select Train Image", 
                                 command=self.select_train_image,
                                 font=("Arial", 12, "bold"),
                                 bg='#E67E22',  # Orange color
                                 fg='white',
                                 activebackground='#D35400',  # Darker orange on click
                                 activeforeground='white',
                                 padx=20,
                                 pady=10,
                                 borderwidth=0,
                                 relief='raised')
      self.train_button.pack(pady=10)

    # Label to display train image path
      self.train_path_label = tk.Label(root, 
                                    text="Train Image: Not Selected", 
                                    font=("Arial", 10), 
                                    bg='#F0F0F0', 
                                    fg='#3498DB')  # Blue text
      self.train_path_label.pack(pady=5)

    # Button to process the images
      self.process_button = tk.Button(root, 
                                   text="🔍 Process Images", 
                                   command=self.process_images, 
                                   state=tk.DISABLED,
                                   font=("Arial", 12, "bold"),
                                   bg='#E74C3C',  # Red color
                                   fg='white',
                                   activebackground='#C0392B',  # Darker red on click
                                   activeforeground='white',
                                   padx=20,
                                   pady=10,
                                   borderwidth=0,
                                   relief='raised')
      self.process_button.pack(pady=20)

    # Add hover effects for buttons
      self.query_button.bind("<Enter>", lambda e: self.query_button.config(bg='#45A049'))  # Darker green on hover
      self.query_button.bind("<Leave>", lambda e: self.query_button.config(bg='#4CAF50'))  # Restore green
      self.train_button.bind("<Enter>", lambda e: self.train_button.config(bg='#D35400'))  # Darker orange on hover
      self.train_button.bind("<Leave>", lambda e: self.train_button.config(bg='#E67E22'))  # Restore orange
      self.process_button.bind("<Enter>", lambda e: self.process_button.config(bg='#C0392B'))  # Darker red on hover
      self.process_button.bind("<Leave>", lambda e: self.process_button.config(bg='#E74C3C'))  # Restore red

    # ===================================================
    # STEP 2: Function to select query image
    # ===================================================
    def select_query_image(self):
        self.query_image_path = filedialog.askopenfilename(title="Select Query Image", filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if self.query_image_path:
            self.query_path_label.config(text=f"Query Image: {os.path.basename(self.query_image_path)}", fg="green")
            self.check_images_selected()

    # ===================================================
    # STEP 3: Function to select train image
    # ===================================================
    def select_train_image(self):
        self.train_image_path = filedialog.askopenfilename(title="Select Train Image", filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if self.train_image_path:
            self.train_path_label.config(text=f"Train Image: {os.path.basename(self.train_image_path)}", fg="green")
            self.check_images_selected()

    # ===================================================
    # STEP 4: Enable the "Process" button if both images are selected
    # ===================================================
    def check_images_selected(self):
        if self.query_image_path and self.train_image_path:
            self.process_button.config(state=tk.NORMAL)

    # ===================================================
    # STEP 5: Function to process the images
    # ===================================================
    def process_images(self):
        try:
            # Read the selected images in grayscale
            img1 = cv2.imread(self.query_image_path, 0)  # Query image
            img2 = cv2.imread(self.train_image_path, 0)  # Train image

            # Check if images were loaded successfully
            if img1 is None or img2 is None:
                raise ValueError("Failed to load one or both images. Please check the file paths.")

            # Initiate the SIFT detector
            sift = cv2.SIFT_create()

            # Detect keypoints and compute descriptors
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)

            # Use Brute-Force Matcher to find matches
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            # Apply Lowe's ratio test to filter good matches
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])

            # Draw the matches
            img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

            # Display the result in a new window
            plt.imshow(img3)
            plt.title("Feature Matching Result")
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", str(e))

# ===================================================
# STEP 6: Run the application
# ===================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = FeatureMatchingApp(root)
    root.mainloop()