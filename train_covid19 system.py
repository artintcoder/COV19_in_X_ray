#Artint Coder
#COVID-19 in X-ray
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
from tensorflow.keras.models import load_model
from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
from tkinter import filedialog
import math
import argparse

def main():
    root = Tk()
    app = Window1(root)
    root.mainloop()
    return
 

class Window1:

    def __init__(self, master):
        
        self.master = master
        self.master.title("COVID-19 in X-ray")
        self.master.geometry('630x600+400+100')
        self.master.config(bg='black')
        self.frame = Frame(self.master)
        self.frame.pack()
        
        
        self.CORONA = StringVar()
        self.Path = StringVar()
        self.Cor = StringVar()

        #=================================== Title =================================================
        self.TitleFrame = Frame(self.master, relief = 'ridge', bd = 4,bg = 'white')
        self.TitleFrame.place(x=0, y=0,width = 630, height = 40)
        
        self.lblTitle = Label(self.TitleFrame,  text = 'COVID-19 in X-ray',bg='white', fg= 'black', font = ('times new romana',15,'bold'))
        self.lblTitle.place(x=240, y=0)
    
        #================================= Frames ===========================================
        
        self.MainFrame = LabelFrame(self.master,  relief = 'ridge', bd = 6,bg = 'white')
        self.MainFrame.place(x=0, y=45,width = 630, height = 555)
        
        self.loadimg2 = Image.open("img.jpg")
        self.renderimg2 = ImageTk.PhotoImage(self.loadimg2)
        self.img2_Frame = Label(self.MainFrame, image=self.renderimg2, relief = 'ridge', bd = 10,bg = 'black')
        self.img2_Frame.image = self.renderimg2
        self.img2_Frame.place(x=10, y=10 ,width = 600, height=430)
        
        self.BtnMainFrame = Frame(self.MainFrame, relief = 'ridge', bd = 8,bg = 'black')
        self.BtnMainFrame.place(x=10, y=500,width = 600, height = 40)

        # The warning sentence in the second frame
        COR= 'The result of a COVID-19 diagnosis is positive or negative.'
        self.COR_Label = Entry(self.MainFrame, fg='black',  font = ('times new romana',15,'bold'), relief = 'ridge', bd = 4,bg = 'white', textvariable= self.Cor)
        self.COR_Label.place(x=10, y=445,width = 600, height = 50 )
        self.Cor.set(COR)
        
        self.Test_path = Entry(self.MainFrame ,textvariable= self.Path).place(x=1000, y=445)

        #================================ Button Frame two =====================
        self.BTN_Covid =  ttk.Button(self.BtnMainFrame,  style = 'TButton', text = 'Covid'  , width = 31,command=self.FunOpenImg).grid(row=0, column=0, padx=0, pady=0)
        self.BTN_Train =  ttk.Button(self.BtnMainFrame, text = 'Train'  , style = 'TButton', width = 31,command=self.FunTrain).grid(row=0, column=1, padx=2, pady=0)
        self.BTN_Exit =  ttk.Button(self.BtnMainFrame, text = 'Exit'  , style = 'TButton', width = 30,command=self.Exite).grid(row=0, column=2, padx=0, pady=0)
        


        
    def FunOpenImg (self):
        global my_image
        
        self.OpenImgFrame = LabelFrame(self.img2_Frame)
        self.OpenImgFrame.place(x=0, y=0,width = 580, height = 410)
        
        self.OpenImgFrame.filename = filedialog.askopenfilename (initialdir="/gui/images", title="Select A File", filetypes = (("png files", "*.png"), ("all files", "*.*")))
        my_image = ImageTk.PhotoImage(Image.open(self.OpenImgFrame.filename))
        
        self.Path.set(self.OpenImgFrame.filename)

        self.TwoFramee = Label(self.OpenImgFrame,  image = my_image )
        self.TwoFramee.place(x=0, y=0,width = 577, height = 428)
        self.FunCovImg()


        #=================================================================================
        #function to open image     
    def FunTrain(self):
        
        DIRECTORY = r"dataset"
        CATEGORIES = ["covid", "normal"]
                    
        
        # initialize the initial learning rate, number of epochs to train for,
        # and batch size
        INIT_LR = 1e-3
        EPOCHS = 25
        BS = 8
        
        # grab the list of images in our dataset directory, then initialize
        # the list of data (i.e., images) and class images
        print("[INFO] loading images...")
        # imagePaths = list(paths.list_images(args["dataset"]))
        data = []
        labels = []
        
        # loop over the image paths
        for category in CATEGORIES:
            path = os.path.join(DIRECTORY, category)
            for img in os.listdir(path):
            	img_path = os.path.join(path, img)
            	image = load_img(img_path, target_size=(224, 224))
            	image = img_to_array(image)
            	image = preprocess_input(image)
        
            	data.append(image)
            	labels.append(category)
        
        # convert the data and labels to NumPy arrays while scaling the pixel
        # intensities to the range [0, 255]
        # data = np.array(data) / 255.0
        data = np.array(data, dtype="float32")
        
        labels = np.array(labels)
        
        # perform one-hot encoding on the labels
        lb = LabelBinarizer()
        labels = lb.fit_transform(labels)
        labels = to_categorical(labels)
        
        # partition the data into training and testing splits using 80% of
        # the data for training and the remaining 20% for testing
        (trainX, testX, trainY, testY) = train_test_split(data, labels,
        	test_size=0.20, stratify=labels, random_state=42)
        
        # initialize the training data augmentation object
        trainAug = ImageDataGenerator(
        	rotation_range=15,
        	fill_mode="nearest")
        
        # load the VGG16 network, ensuring the head FC layer sets are left
        # off
        baseModel = VGG16(weights="imagenet", include_top=False,
        	input_tensor=Input(shape=(224, 224, 3)))
        
        # construct the head of the model that will be placed on top of the
        # the base model
        headModel = baseModel.output
        headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(64, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(2, activation="softmax")(headModel)
        
        # place the head FC model on top of the base model (this will become
        # the actual model we will train)
        model = Model(inputs=baseModel.input, outputs=headModel)
        
        # loop over all layers in the base model and freeze them so they will
        # *not* be updated during the first training process
        for layer in baseModel.layers:
        	layer.trainable = False
        
        # compile our model
        print("[INFO] compiling model...")
        opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
        model.compile(loss="binary_crossentropy", optimizer=opt,
        	metrics=["accuracy"])
        
        # train the head of the network
        print("[INFO] training head...")
        H = model.fit_generator(
        	trainAug.flow(trainX, trainY, batch_size=BS),
        	steps_per_epoch=len(trainX) // BS,
        	validation_data=(testX, testY),
        	validation_steps=len(testX) // BS,
        	epochs=EPOCHS)
        
        # make predictions on the testing set
        print("[INFO] evaluating network...")
        predIdxs = model.predict(testX, batch_size=BS)
        
        # for each image in the testing set we need to find the index of the
        # label with corresponding largest predicted probability
        predIdxs = np.argmax(predIdxs, axis=1)
        
        # show a nicely formatted classification report
        print(classification_report(testY.argmax(axis=1), predIdxs,
        	target_names=lb.classes_))
        
        # compute the confusion matrix and and use it to derive the raw
        # accuracy, sensitivity, and specificity
        cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
        total = sum(sum(cm))
        acc = (cm[0, 0] + cm[1, 1]) / total
        sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
        
        # show the confusion matrix, accuracy, sensitivity, and specificity
        print(cm)
        print("acc: {:.4f}".format(acc))
        print("sensitivity: {:.4f}".format(sensitivity))
        print("specificity: {:.4f}".format(specificity))
        
        # plot the training loss and accuracy
        N = EPOCHS
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy on COVID-19 Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig("plot.png")
        
        # serialize the model to disk
        print("[INFO] saving COVID-19 detector model...")
        model.save("covid19.model", save_format="h5")
                   
      #function to open camera  
    def FunCovImg (self):
        
        model=load_model(r'covid19.model')
        
        face=cv2.imread (self.Path.get())
        
        
        face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
        face=cv2.resize(face,(224,224))
        face=img_to_array(face)
        face=preprocess_input(face)
        face=np.expand_dims(face,axis=0)
        
        (covid,withoutCovid)=model.predict(face)[0]
        covid
        #determine the class label and color we will use to draw the bounding box and text
        COR1 = ('Positive result .. infected with Covid-19')
        COR2 = ('Negative result.. not infected with Covid 19')

        self.Cor.set(COR1) if covid>withoutCovid else self.Cor.set(COR2)
        self.COR_Label.config (fg ='red') if covid>withoutCovid else self.COR_Label.config (fg ='green') 
        
    def Exite(self):
       self.master.destroy()

        
if __name__ == '__main__':
    main()