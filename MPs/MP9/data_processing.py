import numpy as np
import os
import sys
import time

from helperFunctions import getUCF101
from helperFunctions import loadFrame

import h5py
import cv2
import numpy as np

data_directory = '/projects/training/bayw/hdf5/UCF-101-hdf5/'
class_list, train, test = getUCF101(base_directory = data_directory)

con_matrix = np.load('single_frame_confusion_matrix.npy')

to_sort_list = []

for i in range(len(con_matrix)):
	for j in range(len(con_matrix[0])):
		if i == j:
			continue
		to_sort_list.append((con_matrix[i,j], i, j))

to_sort_list.sort()
result_list = to_sort_list[10:]

for prob, i, j in result_list:
	print ('Misidentify {} to be {} of percentage {}'.format(i, j, prob))

