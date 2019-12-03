import argparse
import logging
import os
import sys
import numpy as np
from PIL import Image 
import imageio
logger = logging.getLogger(__name__)

#  output directory
def ensure_directory_exist(directory_name):
    if not os.path.isdir('./' + directory_name):
        os.mkdir(directory_name)

def convert_numpy_array_to_int_array(img_array):
    print(len(img_array))   # will return number of pictures
    image_list = []
    i = 0
    while i < len(img_array):
        for photo_indiv in img_array[i]:
            image = photo_indiv.astype('float32')
            image_list.append(image)
        i += 1
    return image_list

def convert_int_array_to_png(output_dir, image_list, img_idx):
    for i in range(len(img_idx)):
        name = 'pred_' + img_idx[i].split(".")[0]
        print(name)
        file_name = name + '.png'
        full_path = os.path.join(output_dir, file_name)
        imageio.imwrite(full_path, image_list[i])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_data_file",
                    default=None,
                    type=str,
                    required=True,
                    help="Should be a .npy file of the train output(predicted labels).")
    parser.add_argument("--test_id",
                    default=None,
                    type=str,
                    required=True,
                    help="The .npy file of the test ids.")
    parser.add_argument("--output_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The output data dir. Should be a directory where to store the output images.")
    args = parser.parse_args()
    
    img_array = np.load('./' + args.pred_data_file)
    
    with open('./' + args.test_id) as f:
        content = f.readlines()
    img_idx = [x.strip() for x in content]
    output_dir = '../results/' + args.output_dir
    ensure_directory_exist(output_dir)
    
    image_list = convert_numpy_array_to_int_array(img_array)
    convert_int_array_to_png(output_dir, image_list, img_idx)
    
if __name__ == '__main__':
    main()
    
