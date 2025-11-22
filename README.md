# Image-processing-ToolBox
is a graphical user interface (GUI) application developed for a university course on Pattern Recognition and Image Processing. It allows users to load an image , apply various fundamental and advanced image manipulation operations, and visualize the results.

# 🖼️ Image Processing & Pattern Recognition Toolbox

## 📝 Overview

This repository contains the source code for the **CSC558 Image Processing and Pattern Recognition Toolbox**, a comprehensive Graphical User Interface (GUI) application developed as part of university programming assignments.

The goal of this project is to provide a practical platform for applying fundamental and advanced digital image processing algorithms, visualizing their effects, and exploring image analysis techniques like edge detection and feature extraction.

***

## ✨ Key Features

The application categorizes operations into four main groups, accessible via the main interface (Figure 1):

### 1. Point Transformations (`Point Transform Op's`)
* [cite_start]**Brightness & Contrast Adjustment:** Direct manipulation of image intensity[cite: 28, 29, 160].
* [cite_start]**Histogram Analysis:** Displaying and performing Histogram Equalization[cite: 31, 33, 160].

### 2. Local Transformations & Filtering (`Local Transform Op's`)
* **Noise Reduction:** Application of **Low Pass**, **Median**, and **Averaging Filters**[cite: 38, 50, 54, 286].
* [cite_start]**Enhancement:** Application of the **High Pass Filter**[cite: 40].

### 3. Edge Detection & Filtering
A wide range of filters for detecting boundaries and edges:
* [cite_start]**Laplacian, Gaussian, LOG (Laplacian of Gaussian)**[cite: 45, 48, 153].
* **Sobel (Vertical & Horizontal)**[cite: 45].
* [cite_start]**Prewitt (Vertical & Horizontal)**[cite: 41, 48].
* [cite_start]**Canny Method** and **Zero Cross**[cite: 48, 52].

### 4. Global & Morphological Operations
* **Feature Extraction:** **Line detection** and **Circles detection** using the **Hough Transform** method.
* [cite_start]**Morphological Operations:** **Dilation**, **Erosion**, **Open**, and **Close** operations, with user selection of an arbitrary kernel type (e.g., rectangle, diamond, disk).

***

## ⚙️ Setup and Installation

This application was originally developed for a university environment and is likely implemented in **MATLAB** (based on the provided file interface).

1.  **Prerequisites:** Ensure you have **MATLAB** installed on your system.
2.  **Clone Repository:**
    ```bash
    git clone [https://github.com/Aminayosri/Skin-Cancer-Detection.git](https://github.com/Aminayosri/Skin-Cancer-Detection.git)
    cd Skin-Cancer-Detection
    ```
3.  **Run the Main Script:** Open the main application file (e.g., `CSC558_AP_AandR.m` if it's MATLAB) in the MATLAB environment and run it.

***

## 🖼️ Usage

1.  Click **"Open..."** to load an image (supports JPEG and other types)[cite: 74, 87].
2.  Optionally convert the image to **Grayscale** for certain operations[cite: 91].
3.  You can apply **Noise** (Salt & Pepper, Gaussian, or Poisson) before filtering[cite: 20, 23, 26, 133].
4.  Apply any desired transformation or filter from the categorized sections.
5.  Click **"Save Result image"** to save the processed output[cite: 70, 623].

**Submitted by:** Ashwaq al rashed & Raqinah Al rabiah [cite: 18]
**Instructor:** Dr. Mohamed Berbar [cite: 10]
**Institution:** Menoufia University, Faculty of Electronic Engineering [cite: 3, 5]
