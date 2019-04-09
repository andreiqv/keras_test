
import os
if os.path.exists('.local'):
	train_data_path = '/mnt/lin2/datasets/natural_images/'
	validation_data_path = '/mnt/lin2/datasets/natural_images/'
else:
	train_data_path = '/home/andrei/work/t7_cv/natural_images_splited/train'
	validation_data_path = '/home/andrei/work/t7_cv/natural_images_splited/valid'