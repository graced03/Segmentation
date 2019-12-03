import glob
import re
import os
from os.path import join, isfile, exists
import cv2
import numpy as np
import csv
from data import preprocess_mask
import argparse

img_rows = 192
img_cols = 240
# def preprocess_mask(path, image_name):
#     img = cv2.imread(os.path.join(path, image_name))
#     resized_img = cv2.resize(img[:,:,0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
#     img_vec = cv2.resize(resized_img, (img_rows*img_cols, 1))
#     return img_vec

def import_and_convert_pred(path, image_name):
    img = cv2.imread(os.path.join(path, image_name))
    greyscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(greyscale_image, 127, 255, cv2.THRESH_BINARY)
    img_vec = cv2.resize(binary_image, (img_rows*img_cols, 1))
    return img_vec

def import_and_convert_pred_adapt(path, image_name):

    img = cv2.imread(os.path.join(path, image_name))
    greyscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.adaptiveThreshold(greyscale_image, 255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY,11,2)
    img_vec = cv2.resize(binary_image, (img_rows*img_cols, 1))
    return img_vec

def eval_metrics(y_true, y_pred):
    smooth = 1
    # Compute TP, FP, TN, FN
    TP = np.sum(np.logical_and(y_pred == 255, y_true == 255))
    TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    FP = np.sum(np.logical_and(y_pred == 255, y_true == 0))
    FN = np.sum(np.logical_and(y_pred == 0, y_true == 255))

    # Accuracy
    acc = (y_true == y_pred).mean()

    # Dice score.
    im1 = np.asarray(y_true).astype(np.bool)
    im2 = np.asarray(y_pred).astype(np.bool)
    
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im1, im2)
    dice = (2. * intersection.sum() + smooth) / (im1.sum() + im2.sum() + smooth)
    
    # Jaccard similarity coefficient score
    jacc = (intersection.sum() + smooth) / (im1.sum() + im2.sum() - intersection.sum() + smooth)

#     jacc = metrics.jaccard_similarity_score(y_true, y_pred)

    # Sensitivity (recall)
    sensitivity = TP/float(TP + FN)

    # Specificity.
    specificity = TN/float(TN+FP)
    
    return (TP,TN,FP,FN,acc,dice,jacc,sensitivity,specificity)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_mask_path",
                    default=None,
                    type=str,
                    required=True,
                    help="ground truth test mask dir, should be the directory with the predicted images.")
    parser.add_argument("--pred_mask_path",
                    default=None,
                    type=str,
                    required=True,
                    help="The predicted mask dir. Should be the directory with the predicted images.")
    parser.add_argument("--test_id_file",
                    default=None,
                    type=str,
                    required=True,
                    help="test image ids .txt file.")
    parser.add_argument("--output_file",
                    default=None,
                    type=str,
                    required=True,
                    help="name of the output .csv file.")
    args = parser.parse_args()
    
    test_mask_path = args.test_mask_path
    test_masks = os.listdir(test_mask_path)
    
    pred_mask_path = args.pred_mask_path

    with open('./' + args.test_id_file) as f:
        content = f.readlines()
    img_idx = [x.strip() for x in content]
    
    total = len(test_masks)
    result = []
    for i in range(total):
        if 'ISIC' in img_idx[i]:
            img_name = img_idx[i].split('.')[0]+'_Segmentation.png'
        if 'test' in img_idx[i]:
            img_name = img_idx[i].split('_')[0]+'_manual1.gif'
        true_mask = preprocess_mask(test_mask_path, img_name)
        pred_mask = import_and_convert_pred(pred_mask_path, 'pred_'+img_idx[i].split('.')[0]+'.png')
        true_mask = true_mask.reshape(1,img_rows*img_cols)
        metric = eval_metrics(true_mask, pred_mask)
        result.append(list(metric)[4:9])
    
    with open(args.output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(result)
    metrics_ave = np.mean(np.array(result), axis=0)
    print(dict(zip(['acc','dice','jacc','sensitivity','specificity'],metrics_ave)))

if __name__ == '__main__':
    main()