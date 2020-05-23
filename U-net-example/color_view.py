## 표지판의 우회전을 검출하고 IOU 값 즉 검출 정확도를 확인할 수 있도록 코딩한 파일입니다.
from __future__ import print_function

import cv2
from train import img_cols, img_rows
import matplotlib.pyplot as plt
import numpy as np


def prep(img):
    img = img.astype('float32')
    img = cv2.threshold(img, 0.7, 1., cv2.THRESH_BINARY)[1].astype(np.uint8)
    img = cv2.resize(img, (img_cols, img_rows))
    return img

def dice_coef2(y_true_f,y_pred_f):
    smooth = 1.
    intersection=np.sum(y_true_f * y_pred_f)
    #print(np.sum(y_true_f * y_pred_f))
    return print("IOU : ",(2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f)+ smooth) )

def visualize():
    from data import load_train_data
    imgs_test = np.load('imgs_test.npy')
    imgs_test_pred = np.load('imgs_mask_test.npy')
    y_tru = np.load('imgs_test_real_mask.npy')
    total=imgs_test.shape[0]

    #print (imgs_test_pred.shape)

    plt.figure(figsize=(10,6))

    for i in range(total):


        ####test image
        gray_img = imgs_test[i]
        gray_img = cv2.resize(gray_img, (img_cols, img_rows))

        ###test_mask_true
        y_true = y_tru[i]#0~1,resize
        y_true = cv2.resize(y_true, (img_cols, img_rows))
        #cv2.imshow("true",y_true)
        #cv2.waitKey(0)
        y_true[y_true<128]=0
        y_true[y_true>=128]=1

        ##test_mask_pred
        t_img = imgs_test_pred[i,:,:,0]
        bin_img = prep(t_img)#이진화

        print(t_img.shape)
        print(y_true.shape)

        dice_coef2(y_true, bin_img)

        rgb_img = np.ndarray((img_rows, img_cols, 3), dtype=np.uint8)
        rgb_img[:,:,0] = gray_img
        rgb_img[:,:,1] = gray_img
        rgb_img[:,:,2] = gray_img

        for i in range (0, img_rows, 1):
            for j in range (0, img_cols, 1):
                if bin_img[i,j] != 0:
                    rgb_img[i,j,0] = 0
                    rgb_img[i,j,2] = 0

        for i in range (1, img_rows-1, 1):
            for j in range (1, img_cols-1, 1):
                if bin_img[i,j] != 0 and gray_img[i,j] != 0:
                    if (bin_img[i-1,j-1] == 0
                    or bin_img[i-1,j] == 0
                    or bin_img[i-1,j+1] == 0
                    or bin_img[i,j-1] == 0
                    or bin_img[i,j+1] == 0
                    or bin_img[i+1,j-1] == 0
                    or bin_img[i+1,j] == 0
                    or bin_img[i+1,j+1] == 0) :
                        rgb_img[i,j,0] = 0
                        rgb_img[i,j,1] = 255
                        rgb_img[i,j,2] = 0

        plt.figure(figsize=(10,6))
        plt.subplot(121)
        plt.imshow(t_img, cmap='gray')
        plt.subplot(122)
        plt.imshow(rgb_img, cmap='gray')
        plt.pause(2)


if __name__ == '__main__':
    visualize()

cv2.waitKey()
