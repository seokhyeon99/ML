from tkinter import *
import urllib.request
import os
  
from PIL import ImageTk, Image 

def img_processing():
    from keras.models import model_from_json
    json_file = open("model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")
    loaded_model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
    
    import cv2
    im = cv2.imread('test.jpg')
    
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (32,32)) 
    
    in_size = 32*32*3

    import numpy as np
    im = im.reshape(-1,32,32,3)/ 255
    
    r = loaded_model.predict(im)
    
    res = r[0]
    labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    for i, acc in enumerate(res) :
        print(labels[i], "=", int(acc*100))
    return labels[res.argmax()]

def download_img():
    url = txt.get()
    #print(url)
    os.system("curl " + url + " > test.jpg")
    open_img()

def open_img(): 
    x = 'C:/Users/이석현/딥러닝/test.jpg' 
    img = Image.open(x) 
    img = img.resize((250, 250), Image.ANTIALIAS) 
    img = ImageTk.PhotoImage(img) 
    panel = Label(root, image = img) 
    panel.image = img 
    panel.grid(row = 2)
    
    result = img_processing()
    lbl3 = Label(root, text=result)
    lbl3.grid(row=2,column=2)
    
root = Tk()  
root.title("Image Classification") 
root.geometry("550x300+300+150") 
root.resizable(width = True, height = True) 
txt = Entry(root, width=30)
txt.grid(row=1)

btn = Button(root, text ='Upload image', command = download_img).grid(row = 1, column=1, columnspan = 4) 
lbl2 = Label(root, text="This image is: ")
lbl2.grid(row=2,column=1)

root.mainloop() 