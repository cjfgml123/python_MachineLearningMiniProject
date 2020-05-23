## 데이터를 .npy 형식으로 만들기 위한 코드이다.
#data파일은 훈련및 테스트용 데이터로 데이터 셋을 만들고 .NPY형식으로 저장하고 train을 진행할 때 쉽게 데이터를 불러올수 있도록 하기 위한 파일입니다.
from __future__ import print_function

import os
import numpy as np

import cv2

data_path = 'raw2/'

image_rows = 256
image_cols = 256


def create_train_data(): # 이 메소드는 훈련데이터를 생성하는 함수
    train_data_path = os.path.join(data_path, 'train') #train 이미지 폴더있는 경로
    images = os.listdir(train_data_path)
    total = int(len(images) / 2) #마스크랑 이미지랑 두 개씩있으니까 나누기 2한거임

    #train 이미지랑 그 마스크 넣을 배열 만들기
    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)


    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.bmp' #마스크 파일이름이 _mask로 끝나야 순서대로 불러올 수 있음

	#파일 읽어와서 img랑 img_mask에 저장
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
        print(image_name)
        print(image_mask_name)
        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_train.npy', imgs) #파일 읽어온담에 imgs_train.npy 라는 이름으로 저장됨
    np.save('imgs_mask_train.npy', imgs_mask) #마찬가지임
    print('Saving to .npy files done.')

#def load_train_data(): 이건 나중에 학습할 때 훈련데이터 만들어놓은거 바로 불러올수있도록한 함수이고 이하 테스트 부분은 훈련함수와 같습니다.
def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train


def create_test_data():
    train_data_path = os.path.join(data_path, 'test1')
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        imgs[i] = img

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    return imgs_test

if __name__ == '__main__':
    create_train_data()
    create_test_data()
