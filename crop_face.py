import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import numpy as np
import tensorflow as tf

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def location_crop_img(relative_box, height, width):
    rect_start_point = _normalized_to_pixel_coordinates(
             relative_box.xmin, 
             relative_box.ymin, 
             width,height)
    
    rect_end_point = _normalized_to_pixel_coordinates(
             relative_box.xmin + relative_box.width,
             relative_box.ymin + relative_box.height, 
             width,height)
    
    boundingbox_hw = _normalized_to_pixel_coordinates(
             relative_box.width, 
             relative_box.height, 
             width,height)
    
    if rect_start_point[0]<=int(boundingbox_hw[0]/3):
        x_min=1
    else:
        x_min = rect_start_point[0]- int(boundingbox_hw[0]/4)

    if rect_start_point[1]<=int(boundingbox_hw[1]/2):
        y_min=1
    else:
        y_min = rect_start_point[1]-int(boundingbox_hw[1]/1.25)

    # if rect_end_point[0]>=int(boundingbox_hw[0]/3):
    #     x_min=0
    # else:
    #     x_min = rect_start_point[0]- int(hw[0]/3)

    # if rect_start_point[1]<=int(hw[1]/2):
    #     y_min=0
    # else:
    #     y_min = rect_start_point[1]-int(hw[1]/2)
    
    (x_max,y_max) = (rect_end_point[0] + int(boundingbox_hw[0]/5), 
                      rect_end_point[1] + int(boundingbox_hw[1]/3))
    return x_min, y_min, x_max, y_max
    # return x_min, y_min, rect_end_point[0], rect_end_point[1]


def mediapipe_face_detection(image_file):
    with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5) as face_detection:
        # input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
        img_array = np.frombuffer(image_file, np.uint8)  # convert bytes to numpy array
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # image = cv2.imread(image_path) 
        img_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(img_input)
        if not results.detections:
            print("Have no face in image")
            # continue
        height, width, _ = image.shape

        print("Action")
        detection = results.detections[0]
        relative_box = detection.location_data.relative_bounding_box
        cropy_img = image.copy()
        x_min, y_min, x_max, y_max=location_crop_img(relative_box, height, width)

        new_img = cropy_img[y_min:y_max, x_min:x_max]
        return new_img