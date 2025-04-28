import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy
from keras.models import load_model

# Load model
model = load_model('my_model.h5')

# Dictionary
classes = {1: 'Speed limit (20km/h)', 2: 'Speed limit (30km/h)', 3: 'Speed limit (50km/h)', 4: 'Speed limit (60km/h)',
           5: 'Speed limit (70km/h)', 6: 'Speed limit (80km/h)', 7: 'End of speed limit (80km/h)', 8: 'Speed limit (100km/h)',
           9: 'Speed limit (120km/h)', 10: 'No passing', 11: 'No passing veh over 3.5 tons', 12: 'Right-of-way at intersection',
           13: 'Priority road', 14: 'Yield', 15: 'Stop', 16: 'No vehicles', 17: 'Veh > 3.5 tons prohibited', 18: 'No entry',
           19: 'General caution', 20: 'Dangerous curve left', 21: 'Dangerous curve right', 22: 'Double curve', 23: 'Bumpy road',
           24: 'Slippery road', 25: 'Road narrows on the right', 26: 'Road work', 27: 'Traffic signals', 28: 'Pedestrians',
           29: 'Children crossing', 30: 'Bicycles crossing', 31: 'Beware of ice/snow', 32: 'Wild animals crossing',
           33: 'End speed + passing limits', 34: 'Turn right ahead', 35: 'Turn left ahead', 36: 'Ahead only', 37: 'Go straight or right',
           38: 'Go straight or left', 39: 'Keep right', 40: 'Keep left', 41: 'Roundabout mandatory', 42: 'End of no passing',
           43: 'End no passing veh > 3.5 tons'}

# Initialize window
top = tk.Tk()
top.geometry('900x650')
top.title('🚦 Nhận dạng biển báo giao thông 🚦')

# Add background image
background_img = Image.open("background.jpg")  # <-- bạn cần có 1 file background.jpg đẹp
background_img = background_img.resize((900, 650))
bg_img = ImageTk.PhotoImage(background_img)

bg_label = tk.Label(top, image=bg_img)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Add title
heading = tk.Label(top, text="NHẬN DẠNG BIỂN BÁO GIAO THÔNG", font=('Helvetica', 24, 'bold'), bg='#000000', fg='white', pady=20)
heading.pack()

# Image preview and label
sign_image = tk.Label(top, bg="#000000")
sign_image.pack(pady=20)

label = tk.Label(top, font=('Helvetica', 16, 'bold'), bg='#000000', fg='yellow')
label.pack(pady=10)

def classify(file_path):
    image = Image.open(file_path)
    image = image.resize((30, 30))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    pred_probabilities = model.predict(image)[0]
    pred = pred_probabilities.argmax(axis=-1)
    sign = classes[pred + 1]
    label.config(text=sign)

def show_classify_button(file_path):
    classify_b = tk.Button(top, text="Nhận dạng", command=lambda: classify(file_path),
                           bg="#FF5733", fg="white", font=('Helvetica', 14, 'bold'), padx=20, pady=10, borderwidth=0)
    classify_b.pack(pady=20)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail((350, 350))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.config(text="")
        show_classify_button(file_path)
    except:
        pass

upload = tk.Button(top, text="Upload ảnh", command=upload_image,
                   bg="#007ACC", fg="white", font=('Helvetica', 14, 'bold'), padx=20, pady=10, borderwidth=0)
upload.pack(pady=20)

top.mainloop()
