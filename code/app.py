from flask import Flask,request
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from flask import jsonify,json
import io
import base64
from PIL import Image
import cv2
import numpy as np
#import hp5
import tensorflow as tf
from keras.models import _clone_layers_and_model_config, load_model
from keras.models import model_from_json
import pandas as pd
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib
# This function will preprocess images
from tensorflow.keras.applications.efficientnet import preprocess_input
matplotlib.use('Agg')

# from flask import jsonify

import os
app = Flask(__name__)
CORS(app,supports_credentials=True,origins="*")

@app.route("/")  # flask routing
# @crossdomain(origin='*',headers=['access-control-allow-origin','Content-Type'])
def hello():
    return "Hello World…"

@app.route("/step1" ,methods=["POST"])
def step1():
    print("step1 function called ...")
    print(request.form['imageName'])
    
    imageName = request.form['imageName']
    if imageName.lower() == "img1.jpg":
        res = "/Users/rkumar/work/GL/capstone/milestone3/images/stepImage/figure_1.jpg"
    elif imageName.lower() == "img2.jpg":
        res = "/Users/rkumar/work/GL/capstone/milestone3/images/stepImage/figure_2.jpg"
    elif imageName.lower() == "img3.jpg":
        res = "/Users/rkumar/work/GL/capstone/milestone3/images/stepImage/figure_3.jpg"
    elif imageName.lower() == "img4.jpg":
        res = "/Users/rkumar/work/GL/capstone/milestone3/images/stepImage/figure_4.jpg"
    else :
        res = "/Users/rkumar/work/GL/capstone/milestone3/images/stepImage/figure_5.jpg"

    data = readImageDetails(imageName)
    print("data from readImageDetails", data)

    with open(res, mode='rb') as file:
            img = file.read()
            data['img'] = base64.encodebytes(img).decode('utf-8')
    
    return(json.dumps(data,cls=NpEncoder))

def readImageDetails(imageId) :
    imgData = pd.read_csv("/Users/rkumar/work/GL/capstone/milestone3/images/stepImage/milestone1.csv")
    print(imgData.head)
    data = {}
    indx = imgData.file_name[imgData.image == imageId].index.tolist()[0]
    file_name = imgData.file_name[indx]
    car_name = imgData.car_name[indx]
    class_id = imgData.class_id[indx]
    img_ht = imgData.img_ht[indx]
    img_wt = imgData.img_wt[indx]
    bbox_coordinates = imgData.bbox_coordinates[indx]
    data['file_name']=file_name
    data['car_name'] =car_name
    data['class_id']=class_id
    data['img_ht']=img_ht
    data['img_wt']=img_wt
    data['bbox_coordinates']=bbox_coordinates
    print(file_name)
    print(car_name)
    print(class_id)
    print(img_ht)
    print(img_wt)
    print(bbox_coordinates)
    return data

@app.route("/upload_image",methods=["POST"])
# @crossdomain(origin='*',headers=['access-control-allow-origin','Content-Type'])
def processImage():
    # Get a specific file using the name attribute
    # print("modelName :",request.files["modelName"])
    modelSelected = "notSelected";
    
    if request.form["modelName"]:
        print("modelName :",request.form["modelName"])
        modelSelected = request.form["modelName"]

    if request.files.get("File"):
        image = request.files["File"]

        filename =secure_filename(image.filename)
        filepath = os.path.join("/Users/rkumar/work/GL/capstone/milestone3/images",filename)
        image.save(os.path.join("/Users/rkumar/work/GL/capstone/milestone3/images",filename))
        # Check the model name and call loadData or loadCombi data
        if modelSelected=="model1" :
            print("inside modelName :",modelSelected)
            return load_data_combi(filepath)
        else :
            res,car_name = loadData(filepath)

            print("res",res)
            print('car_name',car_name)       
            data = {}
            with open(res, mode='rb') as file:
                img = file.read()
                data['img'] = base64.encodebytes(img).decode('utf-8')
            data['car_name'] = car_name
            return(json.dumps(data))

#if modelName == otherone
    

def load_data_combi(img_path):
    image = cv2.resize(cv2.imread(img_path), (224, 224), interpolation = cv2.INTER_AREA)
    #image = np.expand_dims(image, axis=0)
    print('image loaded',image.shape)

    json_file = open('/Users/rkumar/work/GL/capstone/milestone3/pickle_data/model_combi_json.json', 'r') 
    loaded_model_json = json_file.read()
    json_file.close()
    model_combi= model_from_json(loaded_model_json)

    # load weights into new model
    model_combi.load_weights("/Users/rkumar/work/GL/capstone/milestone3/pickle_data/model_combi_wt.h5")
    print("Loaded model from disk")
    res,label = predict(image,model_combi)
    print("res",res)
    print("label ",label)
    data = {}
    with open(res, mode='rb') as file:
        img = file.read()
        data['img'] = base64.encodebytes(img).decode('utf-8')
    data['car_name'] = label
    return (json.dumps(data))


################
# We will use this function to make prediction on images.
def predict(image, model_combi, returnimage = False,  scale = 0.3):
    processed_image = preprocess(image)
    results = model_combi.predict(processed_image)
    label, (x1, y1, x2, y2), confidence = postprocess(image, results)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 100), 2)
    cv2.putText(
        image,
        '{}'.format(label, confidence),
        (x1, y2 + int(35 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX, scale,
        (255, 0, 0),
        1
        )
      
    # Show the Image with matplotlib
    plt.figure(figsize=(10,10))
    plt.imshow(image[:,:,::-1])
    plt.savefig('/Users/rkumar/work/GL/capstone/milestone3/images/croppedImage/figure_1.jpg',bbox_inches='tight',pad_inches = 0)
    plt.cla()
    plt.clf()  
    return "/Users/rkumar/work/GL/capstone/milestone3/images/croppedImage/figure_1.jpg",label

def preprocess(img):
    image = preprocess_input(img)
    # Expand dimensions as predict expect image in batches
    image = np.expand_dims(image, axis=0)
    # image.shape
    return image


#  postprocessing to extract the class label and the real bounding box coordinates

def postprocess(image, results):
    # Split the results into class probabilities and box coordinates
    bounding_box, class_probs = results

    # First let's get the class label
    # The index of class with the highest confidence is our target class
    class_index = np.argmax(class_probs)

    # Use this index to get the class name.
    cols=['car_name'] 
    car_names = pd.read_csv('/Users/rkumar/work/GL/capstone/milestone3/pickle_data/Car names and make.csv', header=None, names=cols)
    #car_names.head()
    car_names['car_name']= car_names['car_name'].str.replace('/', '-')
    car_dict = car_names.T.to_dict('list')
    
    key_list = list(car_dict.keys())
    val_list = list(car_dict.values())
    val_list = [item for sublist in val_list for item in sublist] # converting val_lit which is a list of list into a flat list
    # val_car_name = val_list[image_classification_id[0]] # will return car name.
    # print(val_car_name)


    class_label = val_list[class_index]

    # Now you can extract the bounding box too.
    # Get the height and width of the actual image
    h, w = image.shape[:2]
    # Extract the Coordinates
    x1, y1, x2, y2 = bounding_box[0]
    # Convert the coordinates from relative (i.e. 0-1) to actual values
    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)
    # return the lable and coordinates
    return class_label, (x1,y1,x2,y2),class_probs
################

def loadData(img_path) :
   # Step 1
# load the image and check size it should be 224 X224 -> resize to 224 X 224
# read the image
    if os.path.exists("/Users/rkumar/work/GL/capstone/milestone3/images/croppedImage/figure_1.jpg"):
        os.remove("/Users/rkumar/work/GL/capstone/milestone3/images/croppedImage/figure_1.jpg")

    image = cv2.resize(cv2.imread(img_path), (224, 224), interpolation = cv2.INTER_AREA)
    image = np.expand_dims(image, axis=0)
    print('image loaded',image.shape)
    # Step 2
    # load the model and detect bounding box

    json_file = open('/Users/rkumar/work/GL/capstone/milestone3/pickle_data/model_bbox_json.json', 'r') 
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model_bounding_box = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model_bounding_box.load_weights("/Users/rkumar/work/GL/capstone/milestone3/pickle_data/model_bbox_wt.h5")
    print("Loaded model from disk")
    
    boundingbox = loaded_model_bounding_box.predict(image)
    print("before loaded bounding box :",boundingbox)

    #saving bounding box image
    # boundingbox_im = Image.fromarray(boundingbox)
    # boundingbox_im.save(os.path.join("/Users/rkumar/work/GL/capstone/milestone3/images/croppedImage",'bound.jpeg'))
    
    plt.imshow(cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB))
    ax = plt.gca()

        # print bounding box for each detected face
    rect = patches.Rectangle((boundingbox[0][0], boundingbox[0][1]), boundingbox[0][2], boundingbox[0][3], 
                            linewidth=2, edgecolor='lime', facecolor='none')
    ax.add_patch(rect)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    res = plt.savefig('/Users/rkumar/work/GL/capstone/milestone3/images/croppedImage/figure_1.jpg',bbox_inches='tight',pad_inches = 0)
    plt.cla()
    plt.clf()
    print(res)


    # Step3 
    # crop the image which return with bounding box 
    X = int(boundingbox[0][0])
    Y = int(boundingbox[0][1])
    W = int(boundingbox[0][2])
    H = int(boundingbox[0][3])
    print([X,Y,W,H])
    #TDOD check if multiple bouding box
    image = np.squeeze(image, axis = 0)
    cropped_image = image[Y:Y+H, X:X+W, :]
    cropped_image = cv2.resize(cropped_image, (224, 224), interpolation = cv2.INTER_AREA)
    print("cropped image shape", cropped_image.shape)
    cropped_image = np.expand_dims(cropped_image, axis=0)
    
    print("cropped image shape", cropped_image.shape)

    # Step 4 
    #Classification 
    # load json and create model

    json_file = open('/Users/rkumar/work/GL/capstone/milestone3/pickle_data/model_classification_json.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model_classification = model_from_json(loaded_model_json)

    # load weights into new model

    loaded_model_classification.load_weights("/Users/rkumar/work/GL/capstone/milestone3/pickle_data/model_classification_wt.h5")
    print("Loaded model from disk")
    print()

    image_classification_id = loaded_model_classification.predict(cropped_image).argmax(axis=1)
    # y_pred2 = y_pred.argmax(axis=1)
    print("detected class id :",image_classification_id)

    # Importing car names in a dataframe

    cols=['car_name'] 
    car_names = pd.read_csv('/Users/rkumar/work/GL/capstone/milestone3/pickle_data/Car names and make.csv', header=None, names=cols)
    #car_names.head()
    car_names['car_name']= car_names['car_name'].str.replace('/', '-')
    car_dict = car_names.T.to_dict('list')
    
    key_list = list(car_dict.keys())
    val_list = list(car_dict.values())
    val_list = [item for sublist in val_list for item in sublist] # converting val_lit which is a list of list into a flat list
    val_car_name = val_list[image_classification_id[0]] # will return car name.
    print(val_car_name)
    if res is None:
        res = "/Users/rkumar/work/GL/capstone/milestone3/images/croppedImage/figure_1.jpg"
    return res,val_car_name

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)
    app.run()