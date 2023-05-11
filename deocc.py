import numpy as np
import cv2
import tensorflow as tf
import torch 
from my_model import Generator
from my_model_2 import Generator_model
import matplotlib.pyplot as plt
import os


def deocclude(input_image):
    #model
    # gen_model = Generator(train_attention=True)
    # gen_model.build((None, 256,256,3))
    gen_model = Generator_model(train_attention=True)
    status = gen_model.load_weights(r"model_3.h5", by_name=True)
    input_image_decoded = tf.expand_dims(input_image, axis=0)
    # output
    output_image = gen_model(input_image_decoded, training=True)
    output_image = output_image[0].numpy()
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    return output_image