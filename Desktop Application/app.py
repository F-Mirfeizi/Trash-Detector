# import libraries
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import tensorflow.keras.utils as utils

# Load the trained model
model = tf.keras.models.load_model('model/trash.h5')

# Create a function to preprocess the image and make predictions
def predict_trash(file_path):
    img = utils.load_img(
        file_path,
        grayscale=False,
        color_mode='rgb',
        target_size=(180, 180),
        interpolation='nearest',
        keep_aspect_ratio=False
    )
    
    img = utils.img_to_array(img)
    img = img / 255.0  # Normalize the image
    result = model.predict(tf.expand_dims(img, axis=0))

    class_id = np.argmax(result)

    classes = {
        0: 'Cardboard',
        1: 'Glass',
        2: 'Metal',
        3: 'Paper',
        4: 'Plastic',
        5: 'Trash'
    }

    return classes[class_id]

# Create a function to open a file dialog for image selection
def open_file_dialog():
    file_path = filedialog.askopenfilename()
    if file_path:
        trash_type = predict_trash(file_path)
        result_label.config(text=f'This is an image of a {trash_type}')
        load_image(file_path)

# Create a function to display the selected image
def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize((300, 300))
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo

# Create a function to display about 
def about():
    about_window = Toplevel(app)
    about_window.title('About')
    
    # Define the about message
    about_message = '''This application is designed for a student project using convolutional neural network-based modeling,
                    with a model accuracy of approximately 66%. The model has been trained on approximately 2527 different images.'''
    
    # Create a label to display the message with wrapping
    message_label = tk.Label(about_window, text=about_message, wraplength=300, padx=10, pady=10)
    message_label.pack()
    
    # Adjust the window size based on the text length
    message_width = message_label.winfo_reqwidth() + 20
    message_height = message_label.winfo_reqheight() + 20
    about_window.geometry(f'{message_width}x{message_height}')

# Create a function to display contact information 
def contact():
    contact_window = Toplevel(app)
    contact_window.title('Contact Us')
    
    # Define the contact message
    contact_message = "For further information, feel free to contact us at f.mirfaizi@gmail.com"

    # Create a label to display the message
    message_label = tk.Label(contact_window, text=contact_message, wraplength=300, padx=10, pady=10)
    message_label.pack()

    # Adjust the window size based on the text length
    message_width = message_label.winfo_reqwidth() + 20
    message_height = message_label.winfo_reqheight() + 20
    contact_window.geometry(f'{message_width}x{message_height}')

# Create the main application window
app = tk.Tk()
app.title('Trash Type Detector')

# Create a button for opening a file dialog
open_button = tk.Button(app, text='Open Image',font=("Verdana",12,'bold'), width=25 ,bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',command=open_file_dialog) 
open_button.pack(pady=50)

# Create a menu
menu = Menu(app)
app.config(menu=menu)
filemenu = Menu(menu)
menu.add_cascade(label='File', menu=filemenu)
filemenu.add_command(label='Open...', command=open_file_dialog)
filemenu.add_separator()
filemenu.add_command(label='Exit', command=app.quit)
helpmenu = Menu(menu)
menu.add_cascade(label='Help', menu=helpmenu)
helpmenu.add_command(label='About', command=about)
helpmenu.add_command(label='Contact', command=contact)

# Create a label to display the predicted trash type
result_label = ttk.Label(app, text='Select an image to detect its type.')
result_label.pack()

# Create a label to display the selected image
image_label = ttk.Label(app)
image_label.pack()

# Run the application
app.mainloop()