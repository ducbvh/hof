import torch
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import cv2
from pytorch_msssim import SSIM, ssim
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
import random

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x
    
class Attention(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.conv1 = nn.LazyConv2d(8 * 128, kernel_size=3, padding=1,
                                            stride=1)
        self.conv2 = nn.LazyConv2d(4 * 128, kernel_size=3, padding=1,
                                            stride=2)
        self.conv3 = nn.LazyConv2d(2 * 128, kernel_size=3, padding=1,
                                            stride=1)
        self.conv4 = nn.LazyConv2d(64, kernel_size=3, padding=1,
                                            stride=1)
        self.conv_t = nn.LazyConvTranspose2d(64, kernel_size=3, padding=1, 
                                            stride=2, output_padding=1)   
        self.max_pool_1 = nn.MaxPool2d(2)    
        self.max_pool_2 = nn.MaxPool2d(2)  

    def forward(self, X):
        #X = torch.cat((f_enc, f_dec), 1)
        out = self.conv1(X)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        #f_used = out[:, :128, :, :] * f_enc + out[:, 128:, :, :] * f_dec 
        #out = self.conv_t(f_used)
        return out
    
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )
        
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        
        upsampling = []
        for out_features in range(2):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)
        
        self.conv2 = nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.norm = nn.InstanceNorm2d(3)
    def forward(self, X, I_mask_ones, I_mask_zeros):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(X)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        out = self.final(u7)
        out_U = I_mask_ones * out + I_mask_zeros * X
        #out_U = self.dropout(out_U)
#         out = self.conv1(out)
#         out = self.upsampling(out)
#         out = self.dropout(out)
#         out = self.conv2(out)
#         #out = self.norm(out)
#         out = self.dropout(out)
#         out = self.conv3(out)
#         return F.tanh(out)
        return out_U

class deocclude_HUY():
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        
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
    
    def create_mask(self,pos_152, pos_127, pos_356, pos_195):
        mask_ones = np.zeros((256, 256), dtype=np.uint8)
        mask_zeros = np.ones((256, 256), dtype=np.uint8)
        w_i = max(0, pos_127[0] - 55)
        w_j = min(pos_356[0] + 55, 256)
        h_i = max(0,pos_195[1] - 35)
        h_j = 256
        w_t = w_i + (w_j - w_i) / 2
        h_t = h_i + (h_j - h_i) / 2
        delta_h = h_t - h_i
        delta_w = w_t - w_i
        
        for i in range(h_i, h_j):
            lamda_h = (h_t - i)**8 * 1 / delta_h**8 + random.random() / 20
            for j in range(w_i, w_j):               
                lamda_w = (w_t - j)**8 * 1 / delta_w**8 + random.random() / 20
                mask_ones[i, j] = np.cos(lamda_h * np.pi/4 + lamda_w * np.pi/4) + 0.01     
                mask_zeros[i, j] = 1. - mask_ones[i, j]
        return mask_zeros, mask_ones

    def get_2_mask(self,image, crop=False):
        land_marks = self.get_landmarks(image)
        pos_152 = land_marks[152][:2]
        pos_127 = land_marks[35][:2]
        pos_356 = land_marks[265][:2]
        pos_195 = land_marks[195][:2]
        mask_zeros, mask_ones = self.create_mask(pos_152, pos_127, pos_356, pos_195)
        return mask_zeros, mask_ones
    
    def f(self,arr, mask):
        indices = np.argwhere(mask != 0)
        min_row = np.min(indices[:, 0]) #get first row, col meeting cond
        max_row = np.max(indices[:, 0]) #get last row, col meeting cond
        min_col = np.min(indices[:, 1]) #get first row, col meeting cond
        max_col = np.max(indices[:, 1]) #get last row, col meeting cond
        return arr[:, min_row:max_row + 1, min_col:max_col + 1]
    
    def rec_image(self,net_G,image):
        img_o = cv2.resize(image, (256, 256))
        #img_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)
        mask_zeros, mask_ones = self.get_2_mask(image)
        
        img_o = np.transpose(img_o, (2, 0, 1))
        #img_o = f(img_o, mask_ones)
        img_o = np.transpose(img_o, (1, 2, 0))
        
        img = torch.Tensor(img_o)
        img = torch.permute(img, (2, 0, 1))
        X = (img / 255.0 - 0.5) * 2
        net_G.eval()
        # net_G.cpu()
        X =torch.Tensor(X)
        mask_ones = torch.Tensor(mask_ones)
        mask_zeros = torch.Tensor(mask_zeros)
        X_rec = net_G(torch.unsqueeze(X, 0), torch.unsqueeze(mask_ones, 0), torch.unsqueeze(mask_zeros, 0))
        X_rec = torch.squeeze(X_rec)
        X_rec = torch.permute(X_rec, (1, 2, 0)).cpu().data.numpy()  
     
        return X_rec