import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

root = tk.Tk()
root.title("Black Room")

def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        global image, image_tk
        image = Image.open(file_path)
        image_tk = ImageTk.PhotoImage(image)
        canvas.delete("all")
        canvas.config(width=image.width, height=image.height)
        canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
        
        create_image_processing_menu()

def create_image_processing_menu():
    global image_menu
    image_menu = tk.Menu(menu_bar, tearoff=0)
    image_menu.add_command(label="Common", state="disabled") 
    image_menu.add_command(label="Bilinear Interpolation", command=double_image_size)
    image_menu.add_command(label="Rotate", command=rotate)
    image_menu.add_command(label="Flip", command=flip_image)
    image_menu.add_command(label="Grayscale", command=grayscale)
    image_menu.add_command(label="Gaussian Blur", command=gaussian_blur)
    image_menu.add_command(label="Sharpen", command=sharpen_image)
    image_menu.add_command(label="Thresholding", command=threshold)
    image_menu.add_command(label="Histogram Equalization", command=equalize_histogram)
    image_menu.add_separator()  # Add a separator before the subheading
    image_menu.add_command(label="Morphology", state="disabled") 
    image_menu.add_command(label="Erosion", command=erode_image)
    image_menu.add_command(label="Dilation", command=dilate_image)
    image_menu.add_command(label="Canny Edge", command=edge_detection)
    image_menu.add_command(label="Segmentation", command=segmentation)
    image_menu.add_separator()  # Add a separator before the subheading
    image_menu.add_command(label="File Ops", state="disabled")
    image_menu.add_command(label="Save", command=save_image)
    # image_menu.add_command(label="Invert", command=invert_image)
    menu_bar.add_cascade(label="Image", menu=image_menu)

def grayscale():
    global image, image_tk
    img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_cv2_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    image = Image.fromarray(cv2.cvtColor(img_cv2_gray, cv2.COLOR_GRAY2RGB))
    image_tk = ImageTk.PhotoImage(image)
    canvas.delete("all")
    canvas.config(width=image.width, height=image.height)
    canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)

def double_image_size():
    global image, image_tk
    width, height = image.size
    doubled_width, doubled_height = width * 2, height * 2
    image = image.resize((doubled_width, doubled_height), Image.BILINEAR)
    image_tk = ImageTk.PhotoImage(image)
    canvas.delete("all")
    canvas.config(width=doubled_width, height=doubled_height)
    canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)


def gaussian_blur():
    global image, image_tk
    img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_cv2_blur = cv2.GaussianBlur(img_cv2, (5, 5), 0)
    image = Image.fromarray(cv2.cvtColor(img_cv2_blur, cv2.COLOR_BGR2RGB))
    image_tk = ImageTk.PhotoImage(image)
    canvas.delete("all")
    canvas.config(width=image.width, height=image.height)
    canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)

def threshold():
    global image, image_tk
    img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_cv2_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    _, img_cv2_thresh = cv2.threshold(img_cv2_gray, 127, 255, cv2.THRESH_BINARY)
    image = Image.fromarray(cv2.cvtColor(img_cv2_thresh, cv2.COLOR_GRAY2RGB))
    image_tk = ImageTk.PhotoImage(image)
    canvas.delete("all")
    canvas.config(width=image.width, height=image.height)
    canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)

def rotate():
    global image, image_tk
    img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_cv2_rotated = cv2.rotate(img_cv2, cv2.ROTATE_90_CLOCKWISE)
    image = Image.fromarray(cv2.cvtColor(img_cv2_rotated, cv2.COLOR_BGR2RGB))
    image_tk = ImageTk.PhotoImage(image)
    canvas.delete("all")
    canvas.config(width=image.width, height=image.height)
    canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)

def flip_image():
    global image, image_tk
    flipped_image = np.fliplr(image)
    image = Image.fromarray(flipped_image)
    image_tk = ImageTk.PhotoImage(image)
    canvas.delete("all")
    canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)

def sharpen_image():
    global image, image_tk
    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    sharpened_image = cv2.filter2D(np.array(image), -1, kernel)
    image_tk = ImageTk.PhotoImage(Image.fromarray(sharpened_image))
    canvas.delete("all")
    canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)

def equalize_histogram():
    global image, image_tk
    img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_cv2_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    img_cv2_equalized = cv2.equalizeHist(img_cv2_gray)
    image = Image.fromarray(cv2.cvtColor(img_cv2_equalized, cv2.COLOR_GRAY2RGB))
    image_tk = ImageTk.PhotoImage(image)
    canvas.delete("all")
    canvas.config(width=image.width, height=image.height)
    canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)


# def morphological_operations():
#     global image, image_tk
#     img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#     img_cv2_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
#     _, img_cv2_thresh = cv2.threshold(img_cv2_gray, 127, 255, cv2.THRESH_BINARY)
#     kernel = np.ones((5,5), np.uint8)
#     img_cv2_eroded = cv2.erode(img_cv2_thresh, kernel, iterations=1)
#     img_cv2_dilated = cv2.dilate(img_cv2_thresh, kernel, iterations=1)
#     image = Image.fromarray(cv2.cvtColor(img_cv2_eroded, cv2.COLOR_BGR2RGB))
#     image_tk = ImageTk.PhotoImage(image)
#     canvas.delete("all")
#     canvas.config(width=image.width, height=image.height)
#     canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)

def erode_image():
    global image, image_tk
    img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_cv2_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    _, img_cv2_thresh = cv2.threshold(img_cv2_gray, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    img_cv2_eroded = cv2.erode(img_cv2_thresh, kernel, iterations=1)
    image = Image.fromarray(cv2.cvtColor(img_cv2_eroded, cv2.COLOR_BGR2RGB))
    image_tk = ImageTk.PhotoImage(image)
    canvas.delete("all")
    canvas.config(width=image.width, height=image.height)
    canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)

def dilate_image():
    global image, image_tk
    img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_cv2_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    _, img_cv2_thresh = cv2.threshold(img_cv2_gray, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    img_cv2_dilated = cv2.dilate(img_cv2_thresh, kernel, iterations=1)
    image = Image.fromarray(cv2.cvtColor(img_cv2_dilated, cv2.COLOR_BGR2RGB))
    image_tk = ImageTk.PhotoImage(image)
    canvas.delete("all")
    canvas.config(width=image.width, height=image.height)
    canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)

def edge_detection():
    global image, image_tk
    edges = cv2.Canny(np.array(image),100,200)
    image = Image.fromarray(edges)
    image_tk = ImageTk.PhotoImage(image)
    canvas.delete("all")
    canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)

def segmentation():
    global image, image_tk
    img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_cv2_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    _, img_cv2_thresh = cv2.threshold(img_cv2_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img_cv2_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_cv2, contours, -1, (0,255,0), 3)
    image = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    image_tk = ImageTk.PhotoImage(image)
    canvas.delete("all")
    canvas.config(width=image.width, height=image.height)
    canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
    # Using kmeans from sklearn
    # img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # img_cv2_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    # img_cv2_gray = np.float32(img_cv2_gray)
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # k = 2
    # ret, label, center = cv2.kmeans(img_cv2_gray, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # center = np.uint8(center)
    # res = center[label.flatten()]
    # res2 = res.reshape((img_cv2_gray.shape))
    # image = Image.fromarray(cv2.cvtColor(res2, cv2.COLOR_BGR2RGB))
    # image_tk = ImageTk.PhotoImage(image)
    # canvas.delete("all")
    # canvas.config(width=image.width, height=image.height)
    # canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)

def save_image():
    global image
    image.save("output.png")

menu_bar = tk.Menu(root)

file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Open", command=open_file)
file_menu.add_command(label="Exit", command=root.quit)

menu_bar.add_cascade(label="File", menu=file_menu)

root.config(menu=menu_bar)

canvas = tk.Canvas(root)
canvas.pack()

root.mainloop()
