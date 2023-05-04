import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import copy

root = tk.Tk()
root.title("Black Room")

imageHistory = []

def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        global image, image_tk
        image = Image.open(file_path)
        image_tk = ImageTk.PhotoImage(image)
        canvas.delete("all")
        canvas.config(width=image.width, height=image.height)
        canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)

        # Clear the history if a new image is loaded
        imageHistory.clear()
        imageHistory.append({ "image": copy.deepcopy(image), "ops": "Loaded Image" })
        
        create_image_processing_menu()

        create_history_menu()

def create_slider_modal(title="Slider Modal", prompt="Slider Value: ",on_slider_change=None, on_settle=None, min=0, max=100):
    def settleHandler():
        on_settle()
        slider_modal.destroy()
    slider_modal = tk.Toplevel()
    slider_modal.title(title)
    tk.Label(slider_modal, text=prompt).pack(padx=10, pady=10)
    slider = tk.Scale(slider_modal, from_=min, to=max, orient=tk.HORIZONTAL, length=200, command=on_slider_change)
    slider.pack(padx=10, pady=10)

    ok_button = tk.Button(slider_modal, text="OK", command=settleHandler)
    ok_button.pack(padx=10, pady=10)

def create_history_menu():
    global history_menu, imageHistory
    history_menu = tk.Menu(menu_bar, tearoff=0)
    for i in range(len(imageHistory)):
        history_menu.add_command(label=imageHistory[i]["ops"], command=lambda i=i: show_history(i))
    # make last item in history menu disabled
    history_menu.entryconfig(len(imageHistory)-1, state="disabled")
    menu_bar.add_cascade(label="History", menu=history_menu)

def update_history_menu():
    global history_menu, imageHistory
    history_menu.delete(0, tk.END)
    for i in range(len(imageHistory)):
        history_menu.add_command(label=imageHistory[i]["ops"], command=lambda i=i: show_history(i))
    # make last item in history menu disabled
    history_menu.entryconfig(len(imageHistory)-1, state="disabled")

def show_history(index):
    global image, image_tk, imageHistory
    image = imageHistory[index]["image"]
    # clear the history after the selected image
    imageHistory[index + 1:] = []
    image_tk = ImageTk.PhotoImage(image)
    canvas.delete("all")
    canvas.config(width=image.width, height=image.height)
    canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
    update_history_menu()

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
    imageHistory.append({ "image": copy.deepcopy(image), "ops": "Grayscale" })
    update_history_menu()

def double_image_size():
    global image, image_tk
    width, height = image.size
    doubled_width, doubled_height = width * 2, height * 2
    image = image.resize((doubled_width, doubled_height), Image.BILINEAR)
    image_tk = ImageTk.PhotoImage(image)
    canvas.delete("all")
    canvas.config(width=doubled_width, height=doubled_height)
    canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
    imageHistory.append({ "image": copy.deepcopy(image), "ops": "Bilinear Interpolation" })
    update_history_menu()

def gaussian_blur():
    def gaussian_blur_callback(value):
        # make value odd, only odd sized kernels allowed
        if int(value) % 2 == 0:
            value = int(value) + 1
        global image, image_tk, temp_image, temp_image_tk
        # do temp image processing
        img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img_cv2_blur = cv2.GaussianBlur(img_cv2, (int(value), int(value)), 0)
        temp_image = Image.fromarray(cv2.cvtColor(img_cv2_blur, cv2.COLOR_BGR2RGB))
        temp_image_tk = ImageTk.PhotoImage(temp_image)
        canvas.delete("all")
        canvas.config(width=temp_image.width, height=temp_image.height)
        canvas.create_image(0, 0, anchor=tk.NW, image=temp_image_tk)

    def settleHandler():
        global imageHistory, temp_image
        imageHistory.append({ "image": copy.deepcopy(temp_image), "ops": "Gaussian Blur" })
        update_history_menu()
    
    # get min and max kernel size depending on image size
    width, height = image.size
    min_kernel_size = 1
    max_kernel_size = min(width, height)
    create_slider_modal("Gaussian Blur", "Enter kernel size", gaussian_blur_callback, settleHandler, min_kernel_size, max_kernel_size)

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
    imageHistory.append({ "image": copy.deepcopy(image), "ops": "Thresholding" })
    update_history_menu()

def rotate():
    global image, image_tk
    img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_cv2_rotated = cv2.rotate(img_cv2, cv2.ROTATE_90_CLOCKWISE)
    image = Image.fromarray(cv2.cvtColor(img_cv2_rotated, cv2.COLOR_BGR2RGB))
    image_tk = ImageTk.PhotoImage(image)
    canvas.delete("all")
    canvas.config(width=image.width, height=image.height)
    canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
    imageHistory.append({ "image": copy.deepcopy(image), "ops": "Rotate" })
    update_history_menu()

def flip_image():
    global image, image_tk
    flipped_image = np.fliplr(image)
    image = Image.fromarray(flipped_image)
    image_tk = ImageTk.PhotoImage(image)
    canvas.delete("all")
    canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
    imageHistory.append({ "image": copy.deepcopy(image), "ops": "Flip" })
    update_history_menu()

def sharpen_image():
    global image, image_tk
    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    sharpened_image = cv2.filter2D(np.array(image), -1, kernel)
    image_tk = ImageTk.PhotoImage(Image.fromarray(sharpened_image))
    canvas.delete("all")
    canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
    imageHistory.append({ "image": copy.deepcopy(image), "ops": "Sharpen" })
    update_history_menu()

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
    imageHistory.append({ "image": copy.deepcopy(image), "ops": "Equalize Histogram" })
    update_history_menu()

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
    imageHistory.append({ "image": copy.deepcopy(image), "ops": "Erode" })
    update_history_menu()

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
    imageHistory.append({ "image": copy.deepcopy(image), "ops": "Dilate" })
    update_history_menu()

def edge_detection():
    global image, image_tk
    edges = cv2.Canny(np.array(image),100,200)
    image = Image.fromarray(edges)
    image_tk = ImageTk.PhotoImage(image)
    canvas.delete("all")
    canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
    imageHistory.append({ "image": copy.deepcopy(image), "ops": "Edge Detection" })
    update_history_menu()

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
    imageHistory.append({ "image": copy.deepcopy(image), "ops": "Segmentation" })
    update_history_menu()

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
