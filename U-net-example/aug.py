from __future__ import print_function

import numpy as np
import cv2
import os
import os.path

data_path = 'C:/Users/konyang/PycharmProjects/untitled8/raw2'
save_path = './'
image_rows = 256
image_cols = 256

def augmentation(image, imageB, max_angle=10, org_width=160,org_height=224, width=190, height=262):

    image=cv2.resize(image,(width,height))
    imageB=cv2.resize(imageB,(width,height))

    angle=np.random.randint(max_angle)
    if np.random.randint(2):
        angle=-angle

    rotation_matrix = cv2.getRotationMatrix2D((image_cols/2, image_rows/2), angle, 1)
    image = cv2.warpAffine(image, rotation_matrix, (image_cols, image_rows))
    imageB = cv2.warpAffine(imageB, rotation_matrix, (image_cols, image_rows))


    xstart=np.random.randint(width-org_width)
    ystart=np.random.randint(height-org_height)

    image=image[xstart:xstart+org_width,ystart:ystart+org_height]
    imageB=imageB[xstart:xstart+org_width,ystart:ystart+org_height]

    if np.random.randint(2):
        image=cv2.flip(image,1)
        imageB=cv2.flip(imageB,1)

    if np.random.randint(2):
        image=cv2.flip(image,0)
        imageB=cv2.flip(imageB,0)

    return image, imageB

def aug_train_data():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    images.sort()
    print (images)
    total = len(images) / 2

    print(total)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.bmp'
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)

        for m in range(29):
            a_img, a_mask_img = augmentation(img, img_mask, max_angle=10, org_width=300,org_height=300, width=332, height=332)

            a_img_name = "a"+str(m)+image_name;
            a_mask_img_name = "a"+str(m)+image_mask_name;

            cv2.imwrite(os.path.join(train_data_path, a_img_name), a_img)
            cv2.imwrite(os.path.join(train_data_path, a_mask_img_name), a_mask_img)

    print('Aug done.')

if __name__ == '__main__':
    aug_train_data()
