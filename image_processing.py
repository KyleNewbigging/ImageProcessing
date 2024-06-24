import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageOps
import numpy as np
import cv2

# Main GUI window

class ImageProcessingApp:

    # Initialize GUI
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")

        self.image = None
        self.img_pil = None

        self.canvas = tk.Canvas(root, width=600, height=400)
        self.canvas.pack()

        # List of buttons for GUI
        crop_button = tk.Button(self.root, text="Crop", command=self.open_crop_window)
        crop_button.pack()

        self.flip_horizontal_button = tk.Button(self.root, text="Flip Horizontal", command=lambda: self.flip_image('horizontal'))
        self.flip_horizontal_button.pack()

        self.flip_vertical_button = tk.Button(self.root, text="Flip Vertical", command=lambda: self.flip_image('vertical'))
        self.flip_vertical_button.pack()

        self.rotate_button = tk.Button(root, text="Rotate", command=self.rotate_image)
        self.rotate_button.pack()

        self.scale_button = tk.Button(root, text="Scale", command=self.scale_image)
        self.scale_button.pack()

        self.linear_mapping_button = tk.Button(self.root, text="Linear Mapping", command=self.open_linear_mapping_window)
        self.linear_mapping_button.pack()

        self.power_law_mapping_button = tk.Button(self.root, text="Power-Law Mapping", command=self.open_power_law_mapping_window)
        self.power_law_mapping_button.pack()

        self.histogram_button = tk.Button(self.root, text="Calculate Histogram", command=self.display_histogram)
        self.histogram_button.pack()

        self.equalize_histogram_button = tk.Button(self.root, text="Equalize Histogram", command=self.equalize_histogram)
        self.equalize_histogram_button.pack()

        self.convolution_button = tk.Button(self.root, text="Convolution", command=self.open_convolution_window)
        self.convolution_button.pack()

        self.load_button = tk.Button(root, text="Load", command=self.load_image)
        self.load_button.pack()

        self.save_button = tk.Button(root, text="Save", command=self.save_image)
        self.save_button.pack()

        self.save_as_button = tk.Button(root, text="Save As", command=self.save_image_as)
        self.save_as_button.pack()

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_image)
        self.clear_button.pack()

    #-------------------- GUI Functions --------------------
    def update_image_on_canvas(self):
        self.image = ImageTk.PhotoImage(self.img_pil)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

    # Add input dialogs for linear mapping
    def open_linear_mapping_window(self):
        slope = simpledialog.askfloat("Linear Mapping", "Enter the slope:")
        intercept = simpledialog.askfloat("Linear Mapping", "Enter the intercept:")
        if slope is not None and intercept is not None:
            self.linear_mapping(slope, intercept)

    # Add input dialogs for power-law mapping
    def open_power_law_mapping_window(self):
        gamma = simpledialog.askfloat("Power-Law Mapping", "Enter the gamma value:")
        c = simpledialog.askfloat("Power-Law Mapping", "Enter the constant c (optional, defaults to 1):", initialvalue=1)
        if gamma is not None and c is not None:
            self.power_law_mapping(gamma, c)

    #-------------------- Convolution --------------------
    def convolve(self, kernel):
        if not self.img_pil:
            return

        img_gray = self.img_pil.convert("L")
        img_np = np.array(img_gray)
        img_conv = cv2.filter2D(img_np, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        self.img_pil = Image.fromarray(img_conv)
        self.update_image_on_canvas()

    def open_convolution_window(self):
        if not self.img_pil:
            return

        self.convolution_window = tk.Toplevel(self.root)
        self.convolution_window.title("Convolution")

        kernel_label = tk.Label(self.convolution_window, text="Enter kernel (comma-separated values, e.g., '1,2,3,4' for a 2x2 kernel):")
        kernel_label.pack()

        self.kernel_entry = tk.Entry(self.convolution_window)
        self.kernel_entry.pack()

        kernel_size_label = tk.Label(self.convolution_window, text="Enter kernel size (e.g., '2,2' for a 2x2 kernel):")
        kernel_size_label.pack()

        self.kernel_size_entry = tk.Entry(self.convolution_window)
        self.kernel_size_entry.pack()

        convolve_btn = tk.Button(self.convolution_window, text="Apply Convolution", command=self.perform_convolution)
        convolve_btn.pack()

    def perform_convolution(self):
        kernel_str = self.kernel_entry.get()
        kernel_size_str = self.kernel_size_entry.get()

        try:
            kernel_values = list(map(float, kernel_str.split(',')))
            kernel_size = tuple(map(int, kernel_size_str.split(',')))

            if len(kernel_values) == kernel_size[0] * kernel_size[1]:
                kernel = np.array(kernel_values).reshape(kernel_size)
                self.convolve(kernel)
                self.convolution_window.destroy()
            else:
                messagebox.showerror("Error", "Invalid kernel size or kernel values.")
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter valid kernel values and size.")


    #-------------------- Crop --------------------
    def crop_image(self, left, upper, right, lower):
        if self.img_pil:
            self.img_pil = self.img_pil.crop((left, upper, right, lower))
            self.image = ImageTk.PhotoImage(self.img_pil)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

    def open_crop_window(self):
        if not self.img_pil:
            return

        self.crop_window = tk.Toplevel(self.root)
        self.crop_window.title("Crop Image")

        width, height = self.img_pil.size
        description = f"Enter the coordinate for each new side. top left corner is (0, 0) and bottom right corner is ({width}, {height})"
        description_label = tk.Label(self.crop_window, text=description, wraplength=300)
        description_label.grid(row=0, column=0, columnspan=2, padx=5, pady=25)

        left_label = tk.Label(self.crop_window, text="Left:")
        left_label.grid(row=1, column=0, padx=5, pady=5)
        self.left_entry = tk.Entry(self.crop_window)
        self.left_entry.grid(row=1, column=1, padx=5, pady=5)

        upper_label = tk.Label(self.crop_window, text="Upper:")
        upper_label.grid(row=2, column=0, padx=5, pady=5)
        self.upper_entry = tk.Entry(self.crop_window)
        self.upper_entry.grid(row=2, column=1, padx=5, pady=5)

        right_label = tk.Label(self.crop_window, text="Right:")
        right_label.grid(row=3, column=0, padx=5, pady=5)
        self.right_entry = tk.Entry(self.crop_window)
        self.right_entry.grid(row=3, column=1, padx=5, pady=5)

        lower_label = tk.Label(self.crop_window, text="Lower:")
        lower_label.grid(row=4, column=0, padx=5, pady=5)
        self.lower_entry = tk.Entry(self.crop_window)
        self.lower_entry.grid(row=4, column=1, padx=5, pady=5)

        crop_btn = tk.Button(self.crop_window, text="Crop", command=self.perform_crop)
        crop_btn.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

    def perform_crop(self):
        try:
            left = int(self.left_entry.get())
            upper = int(self.upper_entry.get())
            right = int(self.right_entry.get())
            lower = int(self.lower_entry.get())

            if left < right and upper < lower:
                self.crop_image(left, upper, right, lower)
                self.crop_window.destroy()
            else:
                messagebox.showerror("Error", "Invalid crop coordinates.")
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter integer values.")


    #-------------------- Clear --------------------
    def clear_image(self):
        self.canvas.delete("all")
        self.image = None
        self.img_pil = None

    #-------------------- Flip --------------------
    def flip_image(self, orientation):
        if self.img_pil:
            if orientation == 'horizontal':
                self.img_pil = self.img_pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            elif orientation == 'vertical':
                self.img_pil = self.img_pil.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            self.image = ImageTk.PhotoImage(self.img_pil)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
    
    #-------------------- Histogram --------------------
    def calculate_histogram(self):
        if not self.img_pil:
            return

        img_gray = self.img_pil.convert("L")
        img_np = np.array(img_gray)
        histogram, _ = np.histogram(img_np.ravel(), bins=256, range=(0, 256))
        return histogram

    def equalize_histogram(self):
        if not self.img_pil:
            return

        img_gray = self.img_pil.convert("L")
        img_equalized = ImageOps.equalize(img_gray)
        self.img_pil = img_equalized
        self.update_image_on_canvas()

    def display_histogram(self):
        histogram = self.calculate_histogram()
        if histogram is not None:
            histogram_window = tk.Toplevel(self.root)
            histogram_window.title("Histogram")

            histogram_canvas = tk.Canvas(histogram_window, width=256, height=100)
            histogram_canvas.pack()

            max_hist_value = max(histogram)
            normalized_histogram = [x / max_hist_value * 100 for x in histogram]

            for i, val in enumerate(normalized_histogram):
                histogram_canvas.create_line(i, 100, i, 100 - val)



    #-------------------- Load --------------------
    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.img_pil = Image.open(file_path)
            self.update_image_on_canvas()

    #-------------------- Linear Mapping --------------------
    def linear_mapping(self, slope, intercept):
        if not self.img_pil:
            return

        img_gray = self.img_pil.convert("L")
        img_np = np.array(img_gray, dtype=np.float32)
        img_np = np.clip(slope * img_np + intercept, 0, 255).astype(np.uint8)
        self.img_pil = Image.fromarray(img_np)
        self.update_image_on_canvas()

    #-------------------- Power Law Mapping --------------------
    def power_law_mapping(self, gamma, c=1):
        if not self.img_pil:
            return

        img_gray = self.img_pil.convert("L")
        img_np = np.array(img_gray, dtype=np.float32)
        img_np = np.clip(c * (img_np / 255.0) ** gamma * 255, 0, 255).astype(np.uint8)
        self.img_pil = Image.fromarray(img_np)
        self.update_image_on_canvas()


    #-------------------- Rotate --------------------
    def rotate_image(self):
        if self.img_pil:
            self.img_pil = self.rotate_image_90_degrees(self.img_pil)
            self.image = ImageTk.PhotoImage(self.img_pil)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

    def rotate_image_90_degrees(self, image):
        width, height = image.size
        new_image = Image.new("RGB", (height, width))

        for x in range(width):
            for y in range(height):
                new_image.putpixel((y, width - x - 1), image.getpixel((x, y)))

        return new_image
    #-------------------- Save --------------------
    def save_image(self):
        if self.img_pil and self.file_path:
            self.img_pil.save(self.file_path)

    def save_image_as(self):
        if self.img_pil:
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg")
            if file_path:
                self.img_pil.save(file_path)
                self.file_path = file_path
    
    #-------------------- Scale --------------------
    def scale_image(self):
        if not self.img_pil:
            return

        scale_factor = simpledialog.askfloat("Scale Image", "Enter the scale factor (Ex. 0.5 to reduce size by 50%):")
        if scale_factor:
            width, height = self.img_pil.size
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            self.img_pil = self.img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.update_image_on_canvas()


# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
