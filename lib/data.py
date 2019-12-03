from __future__ import print_function
import os
import numpy as np
import skimage
from skimage import data, draw
from skimage import transform, util
from PIL import Image
import cv2
data_path = '../../images/combined_images/DRIVE_SKIN'

image_rows = 192
image_cols = 240

def preprocess_img(data_path, image_name):
    if 'ISIC' in image_name:
        img = cv2.imread(os.path.join(data_path, image_name))
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        resized_image = cv2.resize(gray_image, (image_cols, image_rows), interpolation=cv2.INTER_CUBIC)
        img = np.array([resized_image])
    if 'training' in image_name or 'test' in image_name:
        ori_img = Image.open(os.path.join(data_path, image_name)).convert('L', (0.2989, 0.5870, 0.1140, 0))
        resized_image = ori_img.resize((image_cols, image_rows), Image.ANTIALIAS)
        img = np.asarray(resized_image)
    return img

def preprocess_mask(data_path, image_name):
    if 'ISIC' in image_name:
        img = cv2.imread(os.path.join(data_path, image_name))
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        resized_image = cv2.resize(binary_mask, (image_cols,image_rows), interpolation=cv2.INTER_CUBIC)
        img = np.array([resized_image])
    if 'manual1' in image_name or 'test' in image_name:
        g_truth = Image.open(os.path.join(data_path, image_name))
        resized_image = g_truth.resize((image_cols, image_rows), Image.ANTIALIAS)
        img = np.asarray(resized_image)
#     img = np.expand_dims(resized_image, axis = 2)
    return img
    
def create_train_data():
    train_data_path = os.path.join(data_path, 'training/images')
    images = os.listdir(train_data_path)
    total = len(images)
    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    
    train_masks_data_path = os.path.join(data_path, 'training/1st_manual')
    masks = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    
    i = 0
    print('-'*30)
    print('Creating train images...')
    print('-'*30)
    for image_name in images:
        print(image_name)
        if 'ISIC' in image_name:
            imgs[i] = preprocess_img(train_data_path, image_name)
            masks[i] = preprocess_mask(train_masks_data_path, image_name.split(".")[0]+'_Segmentation.png')
            print('Done: {0}/{1} images'.format(i, total))
        if 'training' in image_name:
            imgs[i] = preprocess_img(train_data_path, image_name)
            masks[i] = preprocess_mask(train_masks_data_path, image_name.split("_")[0]+'_manual1.gif')
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
#     assert(np.max(masks)==255 and np.min(masks)==255)
    print('Loading done.')
    np.save('imgs_train.npy', imgs)
    print('Saving to train.npy files done.')
    
    np.save('imgs_mask_train.npy', masks)
    print('Saving to masks.npy files done.')

def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train

def create_test_data():
    test_data_path = os.path.join(data_path, 'test/images')
    images = os.listdir(test_data_path)
    total = len(images)

    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype='S')
    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        print(image_name)
        imgs[i] = preprocess_img(test_data_path, image_name)
        imgs_id[i] = image_name

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')
    np.save('imgs_test.npy', imgs)
    np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    #imgs_id = np.load('imgs_id_test.npy')
    #return imgs_test, imgs_id
    return imgs_test

def main():
    create_train_data()
    create_test_data()

if __name__ == '__main__':
    main()

