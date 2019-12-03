### Convert a float 32 image into a binary image by thresholding
import numpy as np
import cv2
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The input dir, should be the directory with the predicted images.")
    parser.add_argument("--output_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The output data dir. Should be the directory with the binary images.")
    args = parser.parse_args()
    
    for filename in os.listdir(args.input_dir):
        mask = os.path.join(args.input_dir, filename)
        print (mask)
        img = cv2.imread(mask)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        file_name = 'binary_' + str(filename)
        outfile = os.path.join(args.output_dir, file_name)
        cv2.imwrite(outfile, binary_image)

if __name__ == '__main__':
    main()