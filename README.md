# Canny-edge-detection
Canny edge detection is widely used in computer vision tasks such as object detection, image segmentation, and feature extraction. Its ability to reduce noise and produce precise edge maps makes it a fundamental tool for many image processing applications. 
Canny edge detection is a popular image processing technique used to identify and extract edges in digital images. Developed by John F. Canny in 1986, it's widely used in various computer vision and image analysis applications. The technique is known for its ability to produce clean and well-defined edges while reducing noise and eliminating weak or spurious edges. 

Image Preprocessing:

The first step in Canny edge detection is to prepare the image by reducing noise. This is typically done using techniques like Gaussian smoothing. A Gaussian filter is applied to the image to blur it slightly, which helps to reduce noise while preserving the important structural features.

Gradient Calculation:

The next step involves computing the gradient magnitude and direction at each pixel in the smoothed image. The gradient magnitude represents the rate of change of intensity at each pixel, and the gradient direction indicates the direction of the steepest increase in intensity.
Gradient Magnitude Thresholding:

After calculating the gradient magnitude, a thresholding step is applied to create a binary image. This step helps in identifying potential edge pixels. Pixels with gradient magnitudes above a certain threshold are considered edge candidates, while those below the threshold are discarded as non-edge pixels.

Edge Tracking by Hysteresis:

To obtain continuous and well-connected edges, a process called edge tracking by hysteresis is employed. This step involves two threshold values: a high threshold (T_high) and a low threshold (T_low).
Pixels with gradient magnitudes above T_high are considered strong edge pixels.
Pixels with gradient magnitudes between T_low and T_high are considered weak edge pixels.
Pixels with gradient magnitudes below T_low are considered non-edge pixels.

Edge Tracing:

Starting from strong edge pixels, the algorithm traces along the edges by considering neighboring pixels. It uses connectivity rules to link strong and weak edge pixels to form continuous edge contours. This ensures that the final edges are continuous and well-defined.

Edge Thinning:

In this optional step, the width of the detected edges can be reduced by applying a thinning algorithm. This step results in a one-pixel-wide representation of edges.

Edge Output:

The output of the Canny edge detection process is a binary image where edges are represented as white pixels on a black background. 

How to compile Here's a detailed explanation of the Canny edge detection process: STEP BY STEP 

*The code you provided uses the Python OpenCV library and Matplotlib to perform Canny edge detection on an input image and display both the original image and the detected edges. Let's break down the code step by step:

1. Import Required Libraries:

python code 

import cv2
import numpy as np
import matplotlib.pyplot as plt                

                                                      Explanation of this code
 
 Import the necessary libraries:

                cv2 (OpenCV): Used for image processing and computer vision tasks.
                numpy (NumPy): Used for numerical operations on arrays.
                matplotlib.pyplot (Matplotlib): Used for displaying images and plots.


2.Load an Image:

  python code 

 image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
       
                                                Explanation of this code

                    Load an image named 'image.jpg' and read it in grayscale mode
                  (cv2.IMREAD_GRAYSCALE). Grayscale mode simplifies the image to a single channel, which is suitable for edge detection.


3.Apply Gaussian Blur to Reduce Noise:

     python code 

 blurred = cv2.GaussianBlur(image, (5, 5), 0)

                                             Explanation of this code

Apply Gaussian blur to the grayscale image using a 5x5 kernel. This step helps reduce noise and smoothes the image, which can improve the accuracy of edge detection.


4.Perform Canny Edge Detection:

     python code
  
   edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

                                              Explanation of this code

  Use the cv2.Canny function to perform Canny edge detection on the blurred image.
threshold1 and threshold2 are the low and high threshold values used in edge tracking by hysteresis. 
Pixels with gradient magnitudes below threshold1 are considered non-edge pixels, and pixels with gradient magnitudes above threshold2 are considered strong edge pixels.
 Pixels with gradient magnitudes between these thresholds are considered weak edge pixels.


5.Display the Original Image and Detected Edges:

   python code

  plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection'), plt.xticks([]), plt.yticks([])

plt.show()

                                       Explanation of this code

Use Matplotlib to create a subplot with two images side by side.
In the first subplot (plt.subplot(121)), display the original image using plt.imshow. The cmap='gray' argument specifies that the image should be displayed in grayscale.
Set the title for the first subplot to 'Original Image' using plt.title.
Remove the x and y axis ticks using plt.xticks([]) and plt.yticks([]) to make the plot cleaner.
In the second subplot (plt.subplot(122)), display the Canny edge detection result.
Set the title for the second subplot to 'Canny Edge Detection.'
Finally, use plt.show() to display both subplots.

You can use the OpenCV library in Python to perform Canny edge detection. If you haven't already installed OpenCV, you can do so using pip:

step 1 

   Install OpenCV (cv2):

If you haven't already installed OpenCV, you can do so using pip, the Python package manager. Open your command prompt or terminal and run the following command:

      pip install opencv-python

Import OpenCV (cv2) in your Python script:
Once OpenCV is installed, you can import it into your Python script as follows:
   
     import cv2





