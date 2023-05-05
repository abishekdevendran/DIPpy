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

def create_triple_modal(title="Triple Slider Modal", prompts={"p1":"Hello1","p2":"Hello2","p3":"Hello3"}, cb=[None,None,None], on_settle=None, min=0, max=100):
    def onSettleHandler():
        on_settle()
        triple_modal.destroy()
    triple_modal = tk.Toplevel()
    triple_modal.title(title)
    tk.Label(triple_modal, text=prompts["p1"]).pack(padx=10, pady=10)
    slider1 = tk.Scale(triple_modal, from_=min, to=max, orient=tk.HORIZONTAL, length=200, command=cb[0])
    slider1.pack(padx=10, pady=10)
    tk.Label(triple_modal, text=prompts["p2"]).pack(padx=10, pady=10)
    slider2 = tk.Scale(triple_modal, from_=min, to=max, orient=tk.HORIZONTAL, length=200, command=cb[1])
    slider2.pack(padx=10, pady=10)
    tk.Label(triple_modal, text=prompts["p3"]).pack(padx=10, pady=10)
    slider3 = tk.Scale(triple_modal, from_=min, to=max, orient=tk.HORIZONTAL, length=200, command=cb[2])
    slider3.pack(padx=10, pady=10)

    ok_button = tk.Button(triple_modal, text="OK", command=onSettleHandler)
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
    image_menu.add_command(label="Contrast", command=contrast)
    image_menu.add_command(label="Brightness", command=brightness)
    image_menu.add_command(label="Exposure", command=exposure)
    image_menu.add_separator()  # Add a separator before the subheading
    image_menu.add_command(label="Morphology", state="disabled") 
    image_menu.add_command(label="Erosion", command=erode_image)
    image_menu.add_command(label="Dilation", command=dilate_image)
    image_menu.add_command(label="Canny Edge", command=edge_detection)
    image_menu.add_command(label="Segmentation", command=segmentation)
    image_menu.add_command(label="Clarity", command=clarity)
    image_menu.add_command(label="Texture", command=texture)
    image_menu.add_command(label="Grain", command=grain)
    image_menu.add_command(label="Red Gain", command=red_gain)
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

def contrast():
    def contrast_callback(value):
        global image, image_tk, temp_image, temp_image_tk
        # do temp image processing
        value=float(value)
        if (value<0):
            value=(1/(-value+10))*(10)
        else:
            value=1+(value/10)
        img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img_cv2_gray = img_cv2
        img_cv2_cont=cv2.convertScaleAbs(img_cv2_gray, alpha=value, beta=0)
        temp_image = Image.fromarray(cv2.cvtColor(img_cv2_cont, cv2.COLOR_BGR2RGB))
        temp_image_tk = ImageTk.PhotoImage(temp_image)
        canvas.delete("all")
        canvas.config(width=temp_image.width, height=temp_image.height)
        canvas.create_image(0, 0, anchor=tk.NW, image=temp_image_tk)
    
    def settleHandler():
        global image, image_tk, temp_image, temp_image_tk
        image = copy.deepcopy(temp_image)
        image_tk = ImageTk.PhotoImage(image)
        canvas.delete("all")
        canvas.config(width=image.width, height=image.height)
        canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
        imageHistory.append({ "image": copy.deepcopy(image), "ops": "Contrast" })
        update_history_menu()

    # get min and max kernel size depending on image size
    width, height = image.size
    min_kernel_size = -100
    max_kernel_size = 100
    create_slider_modal("Contrast", "Contrast value:", contrast_callback, settleHandler, min_kernel_size, max_kernel_size)

def brightness():
    def brightness_callback(value):
        global image, image_tk, temp_image, temp_image_tk
        # do temp image processing
        img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img_cv2_gray = img_cv2
        img_cv2_cont=cv2.convertScaleAbs(img_cv2_gray, alpha=1, beta=int(value))
        temp_image = Image.fromarray(cv2.cvtColor(img_cv2_cont, cv2.COLOR_BGR2RGB))
        temp_image_tk = ImageTk.PhotoImage(temp_image)
        canvas.delete("all")
        canvas.config(width=temp_image.width, height=temp_image.height)
        canvas.create_image(0, 0, anchor=tk.NW, image=temp_image_tk)
    
    def settleHandler():
        global image, image_tk, temp_image, temp_image_tk
        image = copy.deepcopy(temp_image)
        image_tk = ImageTk.PhotoImage(image)
        canvas.delete("all")
        canvas.config(width=image.width, height=image.height)
        canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
        imageHistory.append({ "image": copy.deepcopy(image), "ops": "Brightness" })
        update_history_menu()

    # get min and max kernel size depending on image size
    width, height = image.size
    min_kernel_size = -127
    max_kernel_size = 127
    create_slider_modal("Brightness", "Brightness value:", brightness_callback, settleHandler, min_kernel_size, max_kernel_size)

def clarity():
    def clarity_callback(value):
        global image, image_tk, temp_image, temp_image_tk
        # do temp image processing
        value=float(value)
        img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        sharpen_amount = value/10
        blur_radius = 3
        # Apply unsharp masking
        img_blur = cv2.GaussianBlur(img_gray, (blur_radius, blur_radius), 0)
        img_sharpened = cv2.addWeighted(img_gray, 1 + sharpen_amount, img_blur, -sharpen_amount, 0)
        temp_image = Image.fromarray(cv2.cvtColor(img_sharpened.astype(np.uint8), cv2.COLOR_GRAY2RGB))
        temp_image_tk = ImageTk.PhotoImage(temp_image)
        canvas.delete("all")
        canvas.config(width=temp_image.width, height=temp_image.height)
        canvas.create_image(0, 0, anchor=tk.NW, image=temp_image_tk)
    
    def settleHandler():
        global image, image_tk, temp_image, temp_image_tk
        image = copy.deepcopy(temp_image)
        image_tk = ImageTk.PhotoImage(image)
        canvas.delete("all")
        canvas.config(width=image.width, height=image.height)
        canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
        imageHistory.append({ "image": copy.deepcopy(image), "ops": "Clarity" })
        update_history_menu()

    # get min and max kernel size depending on image size
    width, height = image.size
    min_kernel_size = -100
    max_kernel_size = 100
    create_slider_modal("Clarity", "Clarityt value:", clarity_callback, settleHandler, min_kernel_size, max_kernel_size)

def texture():
    def texture_callback(value):
        global image, image_tk, temp_image, temp_image_tk
        # do temp image processing
        value=float(value)
        img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        texture_strength = value/100
        bilateral_size = 20
        img_filtered = cv2.bilateralFilter(img_gray, bilateral_size, 75, 75)
        img_diff = cv2.subtract(img_gray, img_filtered)
        img_norm = cv2.normalize(img_diff, None, 0, 255, cv2.NORM_MINMAX)
        img_texture = cv2.addWeighted(img_gray, 1 + texture_strength, img_norm, -texture_strength, 0)
        temp_image = Image.fromarray(cv2.cvtColor(img_texture.astype(np.uint8), cv2.COLOR_GRAY2RGB))
        temp_image_tk = ImageTk.PhotoImage(temp_image)
        canvas.delete("all")
        canvas.config(width=temp_image.width, height=temp_image.height)
        canvas.create_image(0, 0, anchor=tk.NW, image=temp_image_tk)
    
    def settleHandler():
        global image, image_tk, temp_image, temp_image_tk
        image = copy.deepcopy(temp_image)
        image_tk = ImageTk.PhotoImage(image)
        canvas.delete("all")
        canvas.config(width=image.width, height=image.height)
        canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
        imageHistory.append({ "image": copy.deepcopy(image), "ops": "Texture" })
        update_history_menu()

    # get min and max kernel size depending on image size
    width, height = image.size
    min_kernel_size = -100
    max_kernel_size = 100
    create_slider_modal("Texture", "Texture value:", texture_callback, settleHandler, min_kernel_size, max_kernel_size)

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

def grain():
    global grain_amount, grain_roughness, grain_size
    grain_size =1
    grain_amount=1
    grain_roughness=1
    #use triple modal for amount, size and roughness
    def grain_amount_cb(value):
        # print("a:",value)
        global grain_amount
        grain_amount=float(value)
        imgConstructor()
    
    def grain_size_cb(value):
        # print("b:",value)
        global grain_size
        grain_size=float(value)
        imgConstructor()
    
    def grain_roughness_cb(value):
        # print("c:",value)
        global grain_roughness
        grain_roughness=float(value)
        imgConstructor()
    
    def imgConstructor():
        global grain_amount, grain_size, grain_roughness, image, image_tk, temp_image, temp_image_tk
        img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        grain_amount = float(grain_amount/100)
        mean_intensity = np.mean(img_cv2)
        stddev = np.sqrt(grain_roughness * mean_intensity / grain_size)
        noise = np.zeros(img_cv2.shape, np.float32)
        cv2.randn(noise, 0, stddev)
        noise *= grain_amount
        noise = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        img_grain = cv2.add(img_cv2, noise)
        img_grain = cv2.normalize(img_grain, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        temp_image = Image.fromarray(cv2.cvtColor(img_grain, cv2.COLOR_BGR2RGB))
        temp_image_tk = ImageTk.PhotoImage(temp_image)
        canvas.delete("all")
        canvas.config(width=temp_image.width, height=temp_image.height)
        canvas.create_image(0, 0, anchor=tk.NW, image=temp_image_tk)

    def settleHandler():
        global imageHistory, temp_image
        imageHistory.append({ "image": copy.deepcopy(temp_image), "ops": "Gaussian Blur" })
        update_history_menu()
    
    # get min and max grain amnt, size and roughness
    min_grain_amnt = 1
    max_grain_amnt = 100
    create_triple_modal("Grain", {"p1":"Grain Amount", "p2":"Grain Size", "p3":"Grain Roughness"}, [grain_amount_cb, grain_size_cb, grain_roughness_cb], settleHandler, min_grain_amnt, max_grain_amnt)

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
    max_kernel_size = 99
    create_slider_modal("Gaussian Blur", "Kernel size:", gaussian_blur_callback, settleHandler, min_kernel_size, max_kernel_size)

def red_gain():
    def red_callback(value):
        global image, image_tk, temp_image, temp_image_tk
        # do temp image processing
        img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        red_gain = 1.5  # Red channel gain
        img_cv2=img_cv2.astype(np.uint8)
        # Apply the color balance correction
        print(img_cv2[:,:,2].flags)
        img_cv2[:,:,2] = np.ascontiguousarray(img_cv2[:,:,2])
        print(img_cv2[:,:,2].flags)
        print(img_cv2[:,:,2].dtype)
        img_tinted = cv2.merge([
            img_cv2[:,:,0],
            img_cv2[:,:,1],
            cv2.LUT(img_cv2[:,:,2].astype(np.uint8), red_gain)
        ])
        temp_image = Image.fromarray(cv2.cvtColor(img_tinted, cv2.COLOR_BGR2RGB))
        temp_image_tk = ImageTk.PhotoImage(temp_image)
        canvas.delete("all")
        canvas.config(width=temp_image.width, height=temp_image.height)
        canvas.create_image(0, 0, anchor=tk.NW, image=temp_image_tk)

    def settleHandler():
        global imageHistory, temp_image
        imageHistory.append({ "image": copy.deepcopy(temp_image), "ops": "Red Gain" })
        update_history_menu()
    
    # get min and max kernel size depending on image size
    width, height = image.size
    min_kernel_size = 1
    max_kernel_size = 99
    create_slider_modal("Red Gain", "Red amount:", red_callback, settleHandler, min_kernel_size, max_kernel_size)


def threshold():
    def threshold_callback(value):
        global image, image_tk, temp_image, temp_image_tk
        # do temp image processing
        value=float(value)
        img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img_cv2_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        _, img_cv2_thresh = cv2.threshold(img_cv2_gray, value, 255, cv2.THRESH_BINARY)
        temp_image = Image.fromarray(cv2.cvtColor(img_cv2_thresh, cv2.COLOR_GRAY2RGB))
        temp_image_tk = ImageTk.PhotoImage(temp_image)
        canvas.delete("all")
        canvas.config(width=temp_image.width, height=temp_image.height)
        canvas.create_image(0, 0, anchor=tk.NW, image=temp_image_tk)
    
    def settleHandler():
        global image, image_tk, temp_image, temp_image_tk
        image = copy.deepcopy(temp_image)
        image_tk = ImageTk.PhotoImage(image)
        canvas.delete("all")
        canvas.config(width=image.width, height=image.height)
        canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
        imageHistory.append({ "image": copy.deepcopy(image), "ops": "Thresholding" })
        update_history_menu()

    min_kernel_size = 0
    max_kernel_size = 255
    create_slider_modal("Thresholding", "Threshold value:", threshold_callback, settleHandler, min_kernel_size, max_kernel_size)

def exposure():
    def exp_callback(value):
        value=float(value)
        img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        exposure_factor = value/100
        img_float = img_cv2.astype(np.float32) / 255.0
        img_exposure = np.clip(img_float * exposure_factor, 0, 255)
        img_exposure = (img_exposure * 255).astype(np.uint8)
        temp_image=Image.fromarray(cv2.cvtColor(img_exposure, cv2.COLOR_BGR2RGB))
        # temp_image = Image.fromarray(cv2.cvtColor(img_exposure, cv2.COLOR_BGR2RGB))
        temp_image_tk = ImageTk.PhotoImage(temp_image)
        canvas.delete("all")
        canvas.config(width=temp_image.width, height=temp_image.height)
        canvas.create_image(0, 0, anchor=tk.NW, image=temp_image_tk)

    def settleHandler():
        global image, image_tk, temp_image, temp_image_tk
        image = copy.deepcopy(temp_image)
        image_tk = ImageTk.PhotoImage(image)
        canvas.delete("all")
        canvas.config(width=image.width, height=image.height)
        canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
        imageHistory.append({ "image": copy.deepcopy(image), "ops": "Exposure" })
        update_history_menu()
    
    # get min and max kernel size depending on image size
    width, height = image.size
    min_kernel_size = -99
    max_kernel_size = 99
    create_slider_modal("Exposure", "Exposure value:", exp_callback, settleHandler, min_kernel_size, max_kernel_size)
    

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
    def erosion_callback(value):
        global image, image_tk, temp_image, temp_image_tk
        # do temp image processing
        img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img_cv2_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((int(value),int(value)),np.uint8)
        img_cv2_eroded = cv2.erode(img_cv2_gray, kernel, iterations=1)
        temp_image = Image.fromarray(cv2.cvtColor(img_cv2_eroded, cv2.COLOR_GRAY2RGB))
        temp_image_tk = ImageTk.PhotoImage(temp_image)
        canvas.delete("all")
        canvas.config(width=temp_image.width, height=temp_image.height)
        canvas.create_image(0, 0, anchor=tk.NW, image=temp_image_tk)
    
    def settleHandler():
        global image, image_tk, temp_image, temp_image_tk
        image = copy.deepcopy(temp_image)
        image_tk = ImageTk.PhotoImage(image)
        canvas.delete("all")
        canvas.config(width=image.width, height=image.height)
        canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
        imageHistory.append({ "image": copy.deepcopy(image), "ops": "Erosion" })
        update_history_menu()
    
    create_slider_modal("Erosion", "Erosion kernel:", erosion_callback, settleHandler)

def dilate_image():
    def dilation_callback(value):
        global image, image_tk, temp_image, temp_image_tk
        # do temp image processing
        img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img_cv2_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((int(value),int(value)),np.uint8)
        img_cv2_dilated = cv2.dilate(img_cv2_gray, kernel, iterations=1)
        temp_image = Image.fromarray(cv2.cvtColor(img_cv2_dilated, cv2.COLOR_GRAY2RGB))
        temp_image_tk = ImageTk.PhotoImage(temp_image)
        canvas.delete("all")
        canvas.config(width=temp_image.width, height=temp_image.height)
        canvas.create_image(0, 0, anchor=tk.NW, image=temp_image_tk)
    def settleHandler():
        global image, image_tk, temp_image, temp_image_tk
        image = copy.deepcopy(temp_image)
        image_tk = ImageTk.PhotoImage(image)
        canvas.delete("all")
        canvas.config(width=image.width, height=image.height)
        canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
        imageHistory.append({ "image": copy.deepcopy(image), "ops": "Dilation" })
        update_history_menu()
    
    create_slider_modal("Dilation", "Dilation kernel:", dilation_callback, settleHandler)

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