from flask import Flask, request, render_template
import numpy as np
import cv2
import tensorflow as tf
import torch 
from my_model import Generator 
import matplotlib.pyplot as plt
import os
import torch 
from deocc import deocclude
from occ import occlude
from deocc_HUY import deocclude_HUY,GeneratorUNet,UNetDown,UNetUp,Attention
from crop_face import mediapipe_face_detection

OUTPUT_FOLDER = os.path.join('static', 'output_image')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = OUTPUT_FOLDER
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# load machine learning model

def load_img(input_image):
    # input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
    input_image = tf.image.decode_png(input_image, channels= 3)
    #normalize
    input_image = tf.cast(input_image, tf.float32)
    input_image = (input_image / 127.5) - 1
    #resize
    input_image = tf.image.resize(input_image, (256, 256), method= 'bilinear')
    return input_image

def resize_image_bytes(image_bytes, size, format='jpeg'):
    img_array = np.frombuffer(image_bytes, np.uint8)  # convert bytes to numpy array
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # decode image from numpy array
    resized_img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)  # resize the image
    _, output_img_data = cv2.imencode(f'.{format}', resized_img)  # convert to bytes in requested format
    return output_img_data.tobytes()

def resize_image_bytes_for_occ(image_bytes, size, format='jpeg'):
    # img_array = np.frombuffer(image_bytes, np.uint8)  # convert bytes to numpy array
    # img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # decode image from numpy array
    resized_img = cv2.resize(image_bytes, size, interpolation=cv2.INTER_LINEAR)  # resize the image
    _, output_img_data = cv2.imencode(f'.{format}', resized_img)  # convert to bytes in requested format
    return output_img_data.tobytes()   

def resize_occ_image(image_bytes, size, format='jpeg'):
    img_array = np.frombuffer(image_bytes, np.uint8)  # convert bytes to numpy array
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # decode image from numpy array
    resized_img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)  # resize the image
    return resized_img 

def resize_for_oc(image_bytes, size, format='jpeg'):
    # img_array = np.frombuffer(image_bytes, np.uint8)  # convert bytes to numpy array
    # img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # decode image from numpy array
    resized_img = cv2.resize(image_bytes, size, interpolation=cv2.INTER_LINEAR)  # resize the image
    return resized_img 


@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        
        operation = request.form['operation']
        model = request.form['model']
        file = request.files['image']  # get uploaded image file
        img_data = file.read()  # read file contents
        if img_data == b'':
            return render_template("index.html")
        if operation == 'deocclusion': 
            img_data_2 = resize_image_bytes(img_data,(256,256))
            img = load_img(img_data_2)
            new_file_name =  os.path.join(app.config['UPLOAD_FOLDER'], f'input.jpg')
            show_image = img.numpy()*127+127
            show_image =  cv2.cvtColor(show_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(new_file_name,show_image)
            img = load_img(img_data_2)
            if model == 'Pytorch':
                #Use for Pytorch
                img = resize_occ_image(img_data,(256,256)) 
                net_G = GeneratorUNet()
                net_G = torch.load("Generator_494.pt",map_location=torch.device('cpu'))
                gen = deocclude_HUY()
                output_image = gen.rec_image(net_G=net_G,image=img)
            else:
                #Use for Tensorflow
                output_image = deocclude(img)
            file_name =  os.path.join(app.config['UPLOAD_FOLDER'], f'output.jpg')
            output_image = cv2.resize(output_image,(256,256),interpolation = cv2.INTER_LINEAR )
            cv2.imwrite(file_name,(output_image+1)*127)
        else:
            img_data = mediapipe_face_detection(img_data)
            img_data_2 = resize_image_bytes_for_occ(img_data,(256,256))
            img = load_img(img_data_2)
            new_file_name =  os.path.join(app.config['UPLOAD_FOLDER'], f'input.jpg')
            show_image = img.numpy()*127+127
            show_image =  cv2.cvtColor(show_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(new_file_name,show_image)

            img = resize_for_oc(img_data,(256,256)) 
            occ = occlude() 
            output_image = occ.blend(img)
            file_name =  os.path.join(app.config['UPLOAD_FOLDER'], f'output.jpg')
            output_image = cv2.resize(output_image,(256,256),interpolation = cv2.INTER_LINEAR )
            cv2.imwrite(file_name,output_image)
        if file_name != None:
            return render_template("index.html", output  = file_name, input=new_file_name)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
