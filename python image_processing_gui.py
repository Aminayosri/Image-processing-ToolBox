# import tkinter as tk
# from tkinter import filedialog, messagebox, ttk
# from PIL import Image, ImageTk
# import cv2
# import numpy as np

# class ImageProcessingApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("CSCS58 Programming Assignments")
#         self.root.geometry("1200x600")
        
#         self.current_image = None
#         self.original_image = None
        
#         # Create image display
#         self.image_label = tk.Label(self.root)
#         self.image_label.place(x=600, y=50, width=550, height=500)
        
#         # Load and Save Buttons
#         tk.Button(self.root, text="Load Image", command=self.load_image).place(x=50, y=550, width=100, height=30)
#         tk.Button(self.root, text="Save Image", command=self.save_image).place(x=160, y=550, width=100, height=30)
        
#         # Color Mode
#         color_frame = tk.LabelFrame(self.root, text="Color Mode")
#         color_frame.place(x=50, y=450, width=200, height=80)
#         self.color_var = tk.StringVar(value="RGB")
#         tk.Radiobutton(color_frame, text="RGB", variable=self.color_var, value="RGB", command=self.update_color).place(x=10, y=20)
#         tk.Radiobutton(color_frame, text="Gray", variable=self.color_var, value="Gray", command=self.update_color).place(x=60, y=20)
        
#         # Noise Panel
#         noise_frame = tk.LabelFrame(self.root, text="Add Noise")
#         noise_frame.place(x=50, y=350, width=200, height=90)
#         tk.Button(noise_frame, text="Salt & Pepper", command=self.add_salt_pepper_noise).place(x=10, y=10, width=180, height=22)
#         tk.Button(noise_frame, text="Gaussian", command=self.add_gaussian_noise).place(x=10, y=34, width=180, height=22)
#         tk.Button(noise_frame, text="Poisson", command=self.add_poisson_noise).place(x=10, y=58, width=180, height=22)
        
#         # Point Transformations
#         point_frame = tk.LabelFrame(self.root, text="Point Transformations")
#         point_frame.place(x=50, y=200, width=200, height=140)
#         tk.Button(point_frame, text="Brightness", command=self.adjust_brightness).place(x=10, y=10, width=180, height=22)
#         tk.Button(point_frame, text="Contrast", command=self.adjust_contrast).place(x=10, y=34, width=180, height=22)
#         tk.Button(point_frame, text="Histogram", command=self.show_histogram).place(x=10, y=58, width=180, height=22)
#         tk.Button(point_frame, text="Histogram Equalization", command=self.histogram_equalization).place(x=10, y=82, width=180, height=22)
        
#         # Local Transformations
#         local_frame = tk.LabelFrame(self.root, text="Local Transformations")
#         local_frame.place(x=50, y=50, width=200, height=140)
#         tk.Button(local_frame, text="Low Pass Filter", command=self.low_pass_filter).place(x=10, y=10, width=180, height=22)
#         tk.Button(local_frame, text="High Pass Filter", command=self.high_pass_filter).place(x=10, y=34, width=180, height=22)
#         tk.Button(local_frame, text="Median Filter", command=self.median_filter).place(x=10, y=58, width=180, height=22)
#         tk.Button(local_frame, text="Averaging Filter", command=self.averaging_filter).place(x=10, y=82, width=180, height=22)
        
#         # Edge Detection
#         edge_frame = tk.LabelFrame(self.root, text="Edge Detection")
#         edge_frame.place(x=300, y=350, width=200, height=140)
#         tk.Button(edge_frame, text="Laplacian", command=self.laplacian_edge).place(x=10, y=10, width=180, height=22)
#         tk.Button(edge_frame, text="Sobel Vertical", command=self.sobel_vertical).place(x=10, y=34, width=180, height=22)
#         tk.Button(edge_frame, text="Sobel Horizontal", command=self.sobel_horizontal).place(x=10, y=58, width=180, height=22)
#         tk.Button(edge_frame, text="Canny", command=self.canny_edge).place(x=10, y=82, width=180, height=22)
        
#         # Global Transformations
#         global_frame = tk.LabelFrame(self.root, text="Global Transformations")
#         global_frame.place(x=300, y=200, width=200, height=90)
#         tk.Button(global_frame, text="Hough Line", command=self.hough_line).place(x=10, y=10, width=180, height=22)
#         tk.Button(global_frame, text="Hough Circle", command=self.hough_circle).place(x=10, y=34, width=180, height=22)
        
#         # Morphological Operations
#         morph_frame = tk.LabelFrame(self.root, text="Morphological Operations")
#         morph_frame.place(x=300, y=50, width=200, height=140)
#         tk.Button(morph_frame, text="Dilation", command=self.dilation).place(x=10, y=10, width=180, height=22)
#         tk.Button(morph_frame, text="Erosion", command=self.erosion).place(x=10, y=34, width=180, height=22)
#         self.morph_filter = ttk.Combobox(morph_frame, values=["square", "disk", "diamond", "line", "rectangle"])
#         self.morph_filter.set("square")
#         self.morph_filter.place(x=10, y=58, width=180, height=22)
    
#     def load_image(self):
#         file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.bmp")])
#         if file_path:
#             self.current_image = cv2.imread(file_path)
#             self.original_image = self.current_image.copy()
#             self.color_var.set("RGB")
#             self.display_image(self.current_image)
    
#     def save_image(self):
#         if self.current_image is None:
#             messagebox.showerror("Error", "No image to save!")
#             return
#         file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png")])
#         if file_path:
#             cv2.imwrite(file_path, self.current_image)
    
#     def display_image(self, image):
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image_pil = Image.fromarray(image_rgb)
#         image_pil = image_pil.resize((550, 500), Image.Resampling.LANCZOS)
#         photo = ImageTk.PhotoImage(image_pil)
#         self.image_label.configure(image=photo)
#         self.image_label.image = photo
    
#     def update_color(self):
#         if self.current_image is None:
#             return
#         if self.color_var.get() == "Gray":
#             self.current_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
#             if len(self.current_image.shape) == 2:
#                 self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_GRAY2BGR)
#         else:
#             self.current_image = self.original_image.copy()
#         self.display_image(self.current_image)
    
#     def add_salt_pepper_noise(self):
#         if self.current_image is None:
#             messagebox.showerror("Error", "Load an image first!")
#             return
#         noise = np.zeros(self.current_image.shape, np.uint8)
#         cv2.randu(noise, 0, 255)
#         self.current_image = self.current_image.copy()
#         self.current_image[noise < 10] = 0
#         self.current_image[noise > 245] = 255
#         self.display_image(self.current_image)
    
#     def add_gaussian_noise(self):
#         if self.current_image is None:
#             messagebox.showerror("Error", "Load an image first!")
#             return
#         noise = np.random.normal(0, 25, self.current_image.shape).astype(np.uint8)
#         self.current_image = cv2.add(self.current_image, noise)
#         self.display_image(self.current_image)
    
#     def add_poisson_noise(self):
#         if self.current_image is None:
#             messagebox.showerror("Error", "Load an image first!")
#             return
#         vals = len(np.unique(self.current_image))
#         vals = 2 ** np.ceil(np.log2(vals))
#         noisy = np.random.poisson(self.current_image * vals) / float(vals)
#         self.current_image = noisy.astype(np.uint8)
#         self.display_image(self.current_image)
    
#     def adjust_brightness(self):
#         if self.current_image is None:
#             messagebox.showerror("Error", "Load an image first!")
#             return
#         value = tk.simpledialog.askinteger("Input", "Enter brightness value (-50 to 50):", minvalue=-50, maxvalue=50)
#         if value is not None:
#             self.current_image = cv2.convertScaleAbs(self.current_image, beta=value)
#             self.display_image(self.current_image)
    
#     def adjust_contrast(self):
#         if self.current_image is None:
#             messagebox.showerror("Error", "Load an image first!")
#             return
#         self.current_image = cv2.convertScaleAbs(self.current_image, alpha=1.2)
#         self.display_image(self.current_image)
    
#     def show_histogram(self):
#         if self.current_image is None:
#             messagebox.showerror("Error", "Load an image first!")
#             return
#         import matplotlib.pyplot as plt
#         if len(self.current_image.shape) == 3:
#             gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = self.current_image
#         plt.hist(gray.ravel(), 256, [0, 256])
#         plt.title("Histogram")
#         plt.show()
    
#     def histogram_equalization(self):
#         if self.current_image is None:
#             messagebox.showerror("Error", "Load an image first!")
#             return
#         if len(self.current_image.shape) == 3:
#             gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
#             equalized = cv2.equalizeHist(gray)
#             self.current_image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
#         else:
#             self.current_image = cv2.equalizeHist(self.current_image)
#         self.display_image(self.current_image)
    
#     def low_pass_filter(self):
#         if self.current_image is None:
#             messagebox.showerror("Error", "Load an image first!")
#             return
#         self.current_image = cv2.blur(self.current_image, (5, 5))
#         self.display_image(self.current_image)
    
#     def high_pass_filter(self):
#         if self.current_image is None:
#             messagebox.showerror("Error", "Load an image first!")
#             return
#         gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY) if len(self.current_image.shape) == 3 else self.current_image
#         laplacian = cv2.Laplacian(gray, cv2.CV_64F)
#         laplacian = np.uint8(np.absolute(laplacian))
#         self.current_image = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR) if len(self.current_image.shape) == 3 else laplacian
#         self.display_image(self.current_image)
    
#     def median_filter(self):
#         if self.current_image is None:
#             messagebox.showerror("Error", "Load an image first!")
#             return
#         self.current_image = cv2.medianBlur(self.current_image, 5)
#         self.display_image(self.current_image)
    
#     def averaging_filter(self):
#         if self.current_image is None:
#             messagebox.showerror("Error", "Load an image first!")
#             return
#         self.current_image = cv2.blur(self.current_image, (5, 5))
#         self.display_image(self.current_image)
    
#     def laplacian_edge(self):
#         if self.current_image is None:
#             messagebox.showerror("Error", "Load an image first!")
#             return
#         gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY) if len(self.current_image.shape) == 3 else self.current_image
#         laplacian = cv2.Laplacian(gray, cv2.CV_64F)
#         laplacian = np.uint8(np.absolute(laplacian))
#         self.current_image = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR) if len(self.current_image.shape) == 3 else laplacian
#         self.display_image(self.current_image)
    
#     def sobel_vertical(self):
#         if self.current_image is None:
#             messagebox.showerror("Error", "Load an image first!")
#             return
#         gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY) if len(self.current_image.shape) == 3 else self.current_image
#         sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
#         sobel = np.uint8(np.absolute(sobel))
#         self.current_image = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR) if len(self.current_image.shape) == 3 else sobel
#         self.display_image(self.current_image)
    
#     def sobel_horizontal(self):
#         if self.current_image is None:
#             messagebox.showerror("Error", "Load an image first!")
#             return
#         gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY) if len(self.current_image.shape) == 3 else self.current_image
#         sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
#         sobel = np.uint8(np.absolute(sobel))
#         self.current_image = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR) if len(self.current_image.shape) == 3 else sobel
#         self.display_image(self.current_image)
    
#     def canny_edge(self):
#         if self.current_image is None:
#             messagebox.showerror("Error", "Load an image first!")
#             return
#         gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY) if len(self.current_image.shape) == 3 else self.current_image
#         edges = cv2.Canny(gray, 100, 200)
#         self.current_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) if len(self.current_image.shape) == 3 else edges
#         self.display_image(self.current_image)
    
#     def hough_line(self):
#         if self.current_image is None:
#             messagebox.showerror("Error", "Load an image first!")
#             return
#         gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY) if len(self.current_image.shape) == 3 else self.current_image
#         edges = cv2.Canny(gray, 50, 150, apertureSize=3)
#         lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
#         img_copy = self.current_image.copy()
#         if lines is not None:
#             for line in lines:
#                 x1, y1, x2, y2 = line[0]
#                 cv2.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         self.current_image = img_copy
#         self.display_image(self.current_image)
    
#     def hough_circle(self):
#         if self.current_image is None:
#             messagebox.showerror("Error", "Load an image first!")
#             return
#         radius = tk.simpledialog.askinteger("Input", "Enter radius value:", minvalue=1)
#         if radius is None:
#             return
#         gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY) if len(self.current_image.shape) == 3 else self.current_image
#         circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=radius, maxRadius=radius+10)
#         img_copy = self.current_image.copy()
#         if circles is not None:
#             circles = np.uint16(np.around(circles))
#             for i in circles[0, :]:
#                 cv2.circle(img_copy, (i[0], i[1]), i[2], (0, 255, 0), 2)
#         self.current_image = img_copy
#         self.display_image(self.current_image)
    
#     def dilation(self):
#         if self.current_image is None:
#             messagebox.showerror("Error", "Load an image first!")
#             return
#         kernel_type = self.morph_filter.get()
#         if kernel_type == "square":
#             kernel = np.ones((5, 5), np.uint8)
#         elif kernel_type == "disk":
#             kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#         elif kernel_type == "diamond":
#             kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
#         else:
#             kernel = np.ones((5, 5), np.uint8)
#         self.current_image = cv2.dilate(self.current_image, kernel, iterations=1)
#         self.display_image(self.current_image)
    
#     def erosion(self):
#         if self.current_image is None:
#             messagebox.showerror("Error", "Load an image first!")
#             return
#         kernel_type = self.morph_filter.get()
#         if kernel_type == "square":
#             kernel = np.ones((5, 5), np.uint8)
#         elif kernel_type == "disk":
#             kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#         elif kernel_type == "diamond":
#             kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
#         else:
#             kernel = np.ones((5, 5), np.uint8)
#         self.current_image = cv2.erode(self.current_image, kernel, iterations=1)
#         self.display_image(self.current_image)

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = ImageProcessingApp(root)
#     root.mainloop()










import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CSCS58 Programming Assignments")
        self.root.geometry("1200x600")
        
        self.current_image = None
        self.original_image = None
        
        # Create image display
        self.image_label = tk.Label(self.root)
        self.image_label.place(x=600, y=50, width=550, height=500)
        
        # Load and Save Buttons
        tk.Button(self.root, text="Load Image", command=self.load_image).place(x=50, y=550, width=100, height=30)
        tk.Button(self.root, text="Save Image", command=self.save_image).place(x=160, y=550, width=100, height=30)
        
        # Color Mode
        color_frame = tk.LabelFrame(self.root, text="**Color Mode**", fg="blue", font=("Arial", 10, "bold"))
        color_frame.place(x=50, y=450, width=200, height=80)
        self.color_var = tk.StringVar(value="RGB")
        tk.Radiobutton(color_frame, text="RGB", variable=self.color_var, value="RGB", command=self.update_color).place(x=10, y=20)
        tk.Radiobutton(color_frame, text="Gray", variable=self.color_var, value="Gray", command=self.update_color).place(x=60, y=20)
        
        # Noise Panel
        noise_frame = tk.LabelFrame(self.root, text="**Add Noise**", fg="blue", font=("Arial", 10, "bold"))
        noise_frame.place(x=50, y=350, width=200, height=90)
        tk.Button(noise_frame, text="Salt & Pepper", command=self.add_salt_pepper_noise).place(x=10, y=10, width=180, height=22)
        tk.Button(noise_frame, text="Gaussian", command=self.add_gaussian_noise).place(x=10, y=34, width=180, height=22)
        tk.Button(noise_frame, text="Poisson", command=self.add_poisson_noise).place(x=10, y=58, width=180, height=22)
        
        # Point Transformations
        point_frame = tk.LabelFrame(self.root, text="**Point Transformations**", fg="blue", font=("Arial", 10, "bold"))
        point_frame.place(x=50, y=200, width=200, height=140)
        tk.Button(point_frame, text="Brightness", command=self.adjust_brightness).place(x=10, y=10, width=180, height=22)
        tk.Button(point_frame, text="Contrast", command=self.adjust_contrast).place(x=10, y=34, width=180, height=22)
        tk.Button(point_frame, text="Histogram", command=self.show_histogram).place(x=10, y=58, width=180, height=22)
        tk.Button(point_frame, text="Histogram Equalization", command=self.histogram_equalization).place(x=10, y=82, width=180, height=22)
        
        # Local Transformations
        local_frame = tk.LabelFrame(self.root, text="**Local Transformations**", fg="blue", font=("Arial", 10, "bold"))
        local_frame.place(x=50, y=50, width=200, height=140)
        tk.Button(local_frame, text="Low Pass Filter", command=self.low_pass_filter).place(x=10, y=10, width=180, height=22)
        tk.Button(local_frame, text="High Pass Filter", command=self.high_pass_filter).place(x=10, y=34, width=180, height=22)
        tk.Button(local_frame, text="Median Filter", command=self.median_filter).place(x=10, y=58, width=180, height=22)
        tk.Button(local_frame, text="Averaging Filter", command=self.averaging_filter).place(x=10, y=82, width=180, height=22)
        
        # Edge Detection
        edge_frame = tk.LabelFrame(self.root, text="**Edge Detection**", fg="blue", font=("Arial", 10, "bold"))
        edge_frame.place(x=300, y=350, width=200, height=140)
        tk.Button(edge_frame, text="Laplacian", command=self.laplacian_edge).place(x=10, y=10, width=180, height=22)
        tk.Button(edge_frame, text="Sobel Vertical", command=self.sobel_vertical).place(x=10, y=34, width=180, height=22)
        tk.Button(edge_frame, text="Sobel Horizontal", command=self.sobel_horizontal).place(x=10, y=58, width=180, height=22)
        tk.Button(edge_frame, text="Canny", command=self.canny_edge).place(x=10, y=82, width=180, height=22)
        
        # Global Transformations
        global_frame = tk.LabelFrame(self.root, text="**Global Transformations**", fg="blue", font=("Arial", 10, "bold"))
        global_frame.place(x=300, y=200, width=200, height=90)
        tk.Button(global_frame, text="Hough Line", command=self.hough_line).place(x=10, y=10, width=180, height=22)
        tk.Button(global_frame, text="Hough Circle", command=self.hough_circle).place(x=10, y=34, width=180, height=22)
        
        # Morphological Operations
        morph_frame = tk.LabelFrame(self.root, text="**Morphological Operations**", fg="blue", font=("Arial", 10, "bold"))
        morph_frame.place(x=300, y=50, width=200, height=140)
        tk.Button(morph_frame, text="Dilation", command=self.dilation).place(x=10, y=10, width=180, height=22)
        tk.Button(morph_frame, text="Erosion", command=self.erosion).place(x=10, y=34, width=180, height=22)
        self.morph_filter = ttk.Combobox(morph_frame, values=["square", "disk", "diamond", "line", "rectangle"])
        self.morph_filter.set("square")
        self.morph_filter.place(x=10, y=58, width=180, height=22)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.bmp")])
        if file_path:
            self.current_image = cv2.imread(file_path)
            self.original_image = self.current_image.copy()
            self.color_var.set("RGB")
            self.display_image(self.current_image)
    
    def save_image(self):
        if self.current_image is None:
            messagebox.showerror("Error", "No image to save!")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png")])
        if file_path:
            cv2.imwrite(file_path, self.current_image)
    
    def display_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_pil = image_pil.resize((550, 500), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image_pil)
        self.image_label.configure(image=photo)
        self.image_label.image = photo
    
    def update_color(self):
        if self.current_image is None:
            return
        if self.color_var.get() == "Gray":
            self.current_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            if len(self.current_image.shape) == 2:
                self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_GRAY2BGR)
        else:
            self.current_image = self.original_image.copy()
        self.display_image(self.current_image)
    
    def add_salt_pepper_noise(self):
        if self.current_image is None:
            messagebox.showerror("Error", "Load an image first!")
            return
        self.current_image = self.original_image.copy()
        noise = np.zeros(self.current_image.shape, np.uint8)
        cv2.randu(noise, 0, 255)
        self.current_image[noise < 10] = 0
        self.current_image[noise > 245] = 255
        self.display_image(self.current_image)
    
    def add_gaussian_noise(self):
        if self.current_image is None:
            messagebox.showerror("Error", "Load an image first!")
            return
        self.current_image = self.original_image.copy()
        noise = np.random.normal(0, 25, self.current_image.shape).astype(np.uint8)
        self.current_image = cv2.add(self.current_image, noise)
        self.display_image(self.current_image)
    
    def add_poisson_noise(self):
        if self.current_image is None:
            messagebox.showerror("Error", "Load an image first!")
            return
        self.current_image = self.original_image.copy()
        vals = len(np.unique(self.current_image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(self.current_image * vals) / float(vals)
        self.current_image = noisy.astype(np.uint8)
        self.display_image(self.current_image)
    
    def adjust_brightness(self):
        if self.current_image is None:
            messagebox.showerror("Error", "Load an image first!")
            return
        self.current_image = self.original_image.copy()
        value = tk.simpledialog.askinteger("Input", "Enter brightness value (-50 to 50):", minvalue=-50, maxvalue=50)
        if value is not None:
            self.current_image = cv2.convertScaleAbs(self.current_image, beta=value)
            self.display_image(self.current_image)
    
    def adjust_contrast(self):
        if self.current_image is None:
            messagebox.showerror("Error", "Load an image first!")
            return
        self.current_image = self.original_image.copy()
        self.current_image = cv2.convertScaleAbs(self.current_image, alpha=1.2)
        self.display_image(self.current_image)
    
    def show_histogram(self):
        if self.current_image is None:
            messagebox.showerror("Error", "Load an image first!")
            return
        import matplotlib.pyplot as plt
        if len(self.current_image.shape) == 3:
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.current_image
        plt.hist(gray.ravel(), 256, [0, 256])
        plt.title("Histogram")
        plt.show()
    
    def histogram_equalization(self):
        if self.current_image is None:
            messagebox.showerror("Error", "Load an image first!")
            return
        self.current_image = self.original_image.copy()
        if len(self.current_image.shape) == 3:
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            self.current_image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        else:
            self.current_image = cv2.equalizeHist(self.current_image)
        self.display_image(self.current_image)
    
    def low_pass_filter(self):
        if self.current_image is None:
            messagebox.showerror("Error", "Load an image first!")
            return
        self.current_image = self.original_image.copy()
        self.current_image = cv2.blur(self.current_image, (5, 5))
        self.display_image(self.current_image)
    
    def high_pass_filter(self):
        if self.current_image is None:
            messagebox.showerror("Error", "Load an image first!")
            return
        self.current_image = self.original_image.copy()
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY) if len(self.current_image.shape) == 3 else self.current_image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        self.current_image = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR) if len(self.current_image.shape) == 3 else laplacian
        self.display_image(self.current_image)
    
    def median_filter(self):
        if self.current_image is None:
            messagebox.showerror("Error", "Load an image first!")
            return
        self.current_image = self.original_image.copy()
        self.current_image = cv2.medianBlur(self.current_image, 5)
        self.display_image(self.current_image)
    
    def averaging_filter(self):
        if self.current_image is None:
            messagebox.showerror("Error", "Load an image first!")
            return
        self.current_image = self.original_image.copy()
        self.current_image = cv2.blur(self.current_image, (5, 5))
        self.display_image(self.current_image)
    
    def laplacian_edge(self):
        if self.current_image is None:
            messagebox.showerror("Error", "Load an image first!")
            return
        self.current_image = self.original_image.copy()
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY) if len(self.current_image.shape) == 3 else self.current_image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        self.current_image = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR) if len(self.current_image.shape) == 3 else laplacian
        self.display_image(self.current_image)
    
    def sobel_vertical(self):
        if self.current_image is None:
            messagebox.showerror("Error", "Load an image first!")
            return
        self.current_image = self.original_image.copy()
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY) if len(self.current_image.shape) == 3 else self.current_image
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel = np.uint8(np.absolute(sobel))
        self.current_image = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR) if len(self.current_image.shape) == 3 else sobel
        self.display_image(self.current_image)
    
    def sobel_horizontal(self):
        if self.current_image is None:
            messagebox.showerror("Error", "Load an image first!")
            return
        self.current_image = self.original_image.copy()
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY) if len(self.current_image.shape) == 3 else self.current_image
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.uint8(np.absolute(sobel))
        self.current_image = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR) if len(self.current_image.shape) == 3 else sobel
        self.display_image(self.current_image)
    
    def canny_edge(self):
        if self.current_image is None:
            messagebox.showerror("Error", "Load an image first!")
            return
        self.current_image = self.original_image.copy()
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY) if len(self.current_image.shape) == 3 else self.current_image
        edges = cv2.Canny(gray, 100, 200)
        self.current_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) if len(self.current_image.shape) == 3 else edges
        self.display_image(self.current_image)
    
    def hough_line(self):
        if self.current_image is None:
            messagebox.showerror("Error", "Load an image first!")
            return
        self.current_image = self.original_image.copy()
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY) if len(self.current_image.shape) == 3 else self.current_image
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        img_copy = self.current_image.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        self.current_image = img_copy
        self.display_image(self.current_image)
    
    def hough_circle(self):
        if self.current_image is None:
            messagebox.showerror("Error", "Load an image first!")
            return
        self.current_image = self.original_image.copy()
        radius = tk.simpledialog.askinteger("Input", "Enter radius value:", minvalue=1)
        if radius is None:
            return
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY) if len(self.current_image.shape) == 3 else self.current_image
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=radius, maxRadius=radius+10)
        img_copy = self.current_image.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(img_copy, (i[0], i[1]), i[2], (0, 255, 0), 2)
        self.current_image = img_copy
        self.display_image(self.current_image)
    
    def dilation(self):
        if self.current_image is None:
            messagebox.showerror("Error", "Load an image first!")
            return
        self.current_image = self.original_image.copy()
        kernel_type = self.morph_filter.get()
        if kernel_type == "square":
            kernel = np.ones((5, 5), np.uint8)
        elif kernel_type == "disk":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        elif kernel_type == "diamond":
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        else:
            kernel = np.ones((5, 5), np.uint8)
        self.current_image = cv2.dilate(self.current_image, kernel, iterations=1)
        self.display_image(self.current_image)
    
    def erosion(self):
        if self.current_image is None:
            messagebox.showerror("Error", "Load an image first!")
            return
        self.current_image = self.original_image.copy()
        kernel_type = self.morph_filter.get()
        if kernel_type == "square":
            kernel = np.ones((5, 5), np.uint8)
        elif kernel_type == "disk":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        elif kernel_type == "diamond":
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        else:
            kernel = np.ones((5, 5), np.uint8)
        self.current_image = cv2.erode(self.current_image, kernel, iterations=1)
        self.display_image(self.current_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()