import numpy as np
import cv2
import tensorflow as tf
import torch 
from my_model import Generator
import matplotlib.pyplot as plt
import os
import mediapipe as mp 
import random
from scipy import ndimage
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
class occlude():
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.cloth = cv2.imread("new_masks/cloth.png", cv2.IMREAD_UNCHANGED)
        self.cloth_left = cv2.imread("new_masks/cloth_left.png", cv2.IMREAD_UNCHANGED)
        self.cloth_right = cv2.imread("new_masks/cloth_right.png", cv2.IMREAD_UNCHANGED)
        self.surgical = cv2.imread("new_masks/surgical.png", cv2.IMREAD_UNCHANGED)
        self.surgical_left = cv2.imread("new_masks/surgical_left.png", cv2.IMREAD_UNCHANGED)
        self.surgical_right = cv2.imread("new_masks/surgical_right.png", cv2.IMREAD_UNCHANGED)
        self.surgical_blue = cv2.imread("new_masks/surgical_blue.png", cv2.IMREAD_UNCHANGED)
        self.surgical_blue_left = cv2.imread("new_masks/surgical_blue_left.png", cv2.IMREAD_UNCHANGED)
        self.surgical_blue_right = cv2.imread("new_masks/surgical_blue_right.png", cv2.IMREAD_UNCHANGED)
        self.surgical_green = cv2.imread("new_masks/surgical_green.png", cv2.IMREAD_UNCHANGED)
        self.surgical_green_left = cv2.imread("new_masks/surgical_green_left.png", cv2.IMREAD_UNCHANGED)
        self.surgical_green_right = cv2.imread("new_masks/surgical_green_right.png", cv2.IMREAD_UNCHANGED)


    def get_facemesh_coords(self, landmark_list, img):

        h, w = img.shape[:2]  # grab width and height from image
        xyz = [(lm.x, lm.y, lm.z) for lm in landmark_list.landmark]

        return np.multiply(xyz, [w, h, w]).astype(int)


    def get_landmarks(self,image):
        land_marks = 0
        with self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
            # Convert the BGR image to RGB before processing.
            results =   face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            land_marks = self.get_facemesh_coords(results.multi_face_landmarks[0], image)
        return land_marks


    def get_pos_center(self,mask, alpha, w_t, h_t, w_l, type_mask):
        #- np.cos(np.pi/2 - alpha) * w_l
        p = 0
        if type_mask == 1:
            w_t_new = np.cos(alpha) * w_t + np.cos(np.pi/2 - alpha) * mask.shape[0]
            h_t_new = np.cos(np.pi/2 - alpha) * w_t - np.cos(np.pi/2 - alpha) * w_l
            h_sub = h_t_new + h_t
        elif type_mask == -1:
            w_t_new = np.cos(alpha) * w_t
            h_t_new = np.cos(np.pi/2 - alpha) * (mask.shape[1] - w_l) - np.cos(np.pi/2 - alpha) * w_l
            h_sub = np.abs(h_t_new) + h_t     
        else:
            w_t_new = np.cos(alpha) * w_t + np.cos(np.pi/2 - alpha) * mask.shape[0]
            h_t_new = np.cos(np.pi/2 - alpha) * w_t - np.cos(np.pi/2 - alpha) * w_l
            h_sub = h_t_new + h_t            
            p = np.cos(np.pi/2 - alpha) * mask.shape[0]
        return np.abs(w_t_new), np.abs(h_t_new), np.abs(h_sub), p


    def resize_mask(self,mask, type_mask, w_t, h_t, w_l, pad, pos_nose_5, pos_chin_152, \
                left_ear_127, right_ear_356, pos_195):
    
        vec_nose_chin = pos_chin_152 - pos_nose_5
        #vec_nose_chin = np.array([-vec_nose_chin[1], vec_nose_chin[0]])
        o_nose = np.array([pos_nose_5[0], 2000])
        vec_o_nose = o_nose - pos_nose_5 
        #vec_o_nose = np.array([-vec_o_nose[1], vec_o_nose[0]])
        cos_alpha = (vec_o_nose@vec_nose_chin) / (np.linalg.norm(vec_o_nose, 2) * np.linalg.norm(vec_nose_chin, 2))
        alpha = np.arccos(cos_alpha) * 180 / np.pi
        alpha_rad = np.arccos(cos_alpha)

        w_resize = int((right_ear_356[0] - left_ear_127[0]) * 1.15)
        h_resize = int(((pos_chin_152[1] - pos_nose_5[1]) / cos_alpha) * 1.18)
        w_t_resize = int(w_t * w_resize / mask.shape[1])

        mask_resize = cv2.resize(mask, (w_resize, h_resize))

        #d = np.tan(np.arccos(cos_alpha)) * mask_resize.shape[1]/2
        #w_t_resize = (w_resize * w_t) / mask.shape[1]
        h_t_resize = (h_resize * h_t) / mask.shape[0]
        w_l_resize = int(w_l * w_resize / mask.shape[1])

        if type_mask == 1 and right_ear_356[0] - pos_nose_5[0] > w_resize - 0.87 * w_t_resize:
            t = int(w_t_resize * 0.85)
            new_mask = np.zeros((h_resize, int(t + (w_resize - t) * 2.1), 4))
            new_mask[:, :t, :] = mask_resize[:, :t, :]
            buffer = mask_resize[:, t:, :]
            replace = cv2.resize(buffer, (new_mask.shape[1] - t, h_resize))
            new_mask[:, t:, :] = replace
        elif type_mask == -1 and pos_nose_5[0] - left_ear_127[0] > w_resize - 0.87 * (w_resize - w_t_resize):
            t = int(w_t_resize * 1.15)
            new_mask = np.zeros((h_resize, int(t *1.7 + (w_resize - t)), 4))
            new_mask[:, int(t * 1.7):, :] = mask_resize[:, t:, :]
            buffer = mask_resize[:, :t, :]
            replace = cv2.resize(buffer, (int(t * 1.7), h_resize))
            new_mask[:, :int(t * 1.7), :] = replace
            w_t_resize = int(w_t_resize * 2.1)
        else:
            new_mask = mask_resize

        w_t_new, h_t_new, h_sub, p = self.get_pos_center(mask_resize, alpha_rad, w_t_resize, h_t_resize, w_l_resize, type_mask)

        if vec_nose_chin[0] > vec_o_nose[0]:
            rotated = ndimage.rotate(new_mask, alpha)
            if type_mask == -1:
                w_pos = int(pos_nose_5[0] - w_t_new * 1.15)
            elif type_mask == 1:
                w_pos = int(pos_nose_5[0] - w_t_new * 0.85)
            else:
                w_pos = int(pos_nose_5[0] - w_t_new + p)
            h_pos = int(pos_nose_5[1] * 0.97 - h_sub) 
        else:
            rotated = ndimage.rotate(new_mask, -alpha)
            if type_mask == -1:
                w_pos = int(pos_nose_5[0] - w_t_new * 1.15)
            elif type_mask == 1:
                w_pos = int(pos_nose_5[0] - w_t_new * 0.85)
            else:
                w_pos = int(pos_nose_5[0] - w_t_new)
            h_pos = int(pos_nose_5[1] * 0.97 - h_sub) 
        return rotated, np.abs(w_pos), np.abs(h_pos) 


    def get_pos(self,land_marks, mask, w_t, h_t, w_l, pad, type_mask):
        pos_nose_5 = land_marks[5][:2]
        pos_chin_152 = land_marks[152][:2]
        left_ear_127 = land_marks[35][:2]
        right_ear_356 = land_marks[265][:2]
        pos_195 = land_marks[195][:2]
        img_resize, w_pos, h_pos = self.resize_mask(mask, type_mask, w_t, h_t, w_l, pad, \
                                            pos_nose_5, pos_chin_152, left_ear_127, \
                                            right_ear_356, pos_195)
        return img_resize, w_pos, h_pos


    def get_type_mask(self, land_marks):
        pos_5 = land_marks[5][:2]
        pos_4 = land_marks[4][:2]
        pos_203 = land_marks[203][:2]
        pos_423 = land_marks[423][:2]
        pos_36 = land_marks[36][:2]
        pos_266 = land_marks[266][:2]
        vec_4_2 = pos_4 - pos_5
        vec_4_203 = pos_203 - pos_4
        vec_4_423 = pos_423 - pos_4 
        cos_alpha_203 = (vec_4_2@vec_4_203) / (np.linalg.norm(vec_4_203, 2) * np.linalg.norm(vec_4_2, 2))
        cos_alpha_423 = (vec_4_2@vec_4_423) / (np.linalg.norm(vec_4_423, 2) * np.linalg.norm(vec_4_2, 2))
        alpha_rad_203 = np.arccos(cos_alpha_203)
        alpha_rad_423 = np.arccos(cos_alpha_423)
        div = alpha_rad_203 / alpha_rad_423
        print("alpha_rad_203",alpha_rad_203)
        print("alpha_rad_423",alpha_rad_423)
        if div > 1.5:
            return 1
        elif 1 / div > 1.5:
            return -1
        else:
            return 0


    def overlay_transparent(self,background, overlay, x, y):

        background_width = background.shape[1]
        background_height = background.shape[0]

        if x >= background_width or y >= background_height:
            return background

        h, w = overlay.shape[0], overlay.shape[1]

        if x + w > background_width:
            w = background_width - x
            overlay = overlay[:, :w]

        if y + h > background_height:
            h = background_height - y
            overlay = overlay[:h]

        if overlay.shape[2] < 4:
            overlay = np.concatenate(
                [
                    overlay,
                    np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
                ],
                axis = 2,
            )

        overlay_image = overlay[..., :3]
        mask = overlay[..., 3:] /255.0

        background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

        return background
    def blend(self,image):
    
        list_mask_0 = [(self.cloth, 0, 310, 10, 40, 20), (self.surgical, 0, 300, 10, 20, 20), \
                    (self.surgical_blue, 0, 320, 10, 20, 20)]
        list_mask_nev = [(self.cloth_left, -1, 140, 10, 440, 20), (self.surgical_left, -1, 100, 10, 350, 20), \
                        (self.surgical_blue_left, -1, 100, 10, 350, 20)]
        list_mask_pos = [(self.cloth_right, 1, 375, 10
                        , 60, 20), (self.surgical_right, 1, 375, 10, 50, 20), \
                        (self.surgical_blue_right, 1, 375, 10, 50, 20)]
        land_marks = self.get_landmarks(image)
        type_mask = self.get_type_mask(land_marks)
        random_mask = random.randint(0, 2)
        if type_mask == -1 :
            mask, _, w_t, h_t, w_l, pad = list_mask_nev[random_mask]
        elif type_mask == 0:
            mask, _, w_t, h_t, w_l, pad = list_mask_0[random_mask]
        else:
            mask, _, w_t, h_t, w_l, pad = list_mask_pos[random_mask]

        mask_resize, w_pos, h_pos = self.get_pos(land_marks, mask, w_t, h_t, w_l, pad, type_mask)
        blend = self.overlay_transparent(image, mask_resize, w_pos, h_pos)

        return blend
