import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageFilter, ImageOps

root = tk.Tk()
root.title("My Image Viewer")

def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        global image, image_tk
        image = Image.open(file_path)
        image_tk = ImageTk.PhotoImage(image)
        canvas.delete("all")
        canvas.config(width=root.winfo_screenwidth(), height=root.winfo_screenheight())
        canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
        canvas.image = image_tk
        create_image_processing_menu()

def create_image_processing_menu():
    global image_menu
    image_menu = tk.Menu(menu_bar, tearoff=0)
    image_menu.add_command(label="Blur", command=blur_image)
    image_menu.add_command(label="Invert", command=invert_image)
    menu_bar.add_cascade(label="Image", menu=image_menu)

def blur_image():
    global image, image_tk
    blurred_image = image.filter(ImageFilter.BLUR)
    image_tk = ImageTk.PhotoImage(blurred_image)
    canvas.delete("all")
    canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
    canvas.image = image_tk

def invert_image():
    global image, image_tk
    inverted_image = ImageOps.invert(image)
    image_tk = ImageTk.PhotoImage(inverted_image)
    canvas.delete("all")
    canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
    canvas.image = image_tk

menu_bar = tk.Menu(root)

file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Open", command=open_file)
file_menu.add_command(label="Exit", command=root.quit)

menu_bar.add_cascade(label="File", menu=file_menu)

root.config(menu=menu_bar)

canvas = tk.Canvas(root)
canvas.pack()

root.mainloop()
