# Segmentation
Go to the lib dir and run the following in sequence:

### To get the .npy of the data for training
python data.py
### Train the Unet
python train.py

### To get the predicted images(sample)
You have to create a file called combined
python pred_to_img.py \
       --pred_data_file=imgs_mask_test_combined.npy \
       --test_id=test_id.txt \
       --output_dir=./results/combined
       
You have to create a file called binary_combined    
### To get the binary predicted images(sample)
python float32_image_to_binary.py \
       --input_dir=../results/combined \
       --output_dir=../results/combined_binary
       
### To eval the result(sample)
python eval.py \
       --test_mask_path=../../images/combined_images/DRIVE_SKIN/test/1st_manual/ #dir of the groundtruth segmentation
       --pre_mask_path=../results/combined \ #dir of the predicted segmentation from pred_to_img.py
       --test_id_file=test_id.txt \
       --output_file=eval_results.csv
       
