#!/usr/bin/env python
import cv2
import sys
import numpy as np 
import math
import torch
from torch.autograd import Variable


if __name__ == '__main__':
	# var to change
	# stride when slide window on image
	# l is the fixed window size. In this problem I make this assumption
	# because the given training images seem to have fixed size of iphone in view
	l = 23
	
	# gpu options
	if torch.cuda.is_available():
		# stride choose between 1-46
		stride = 1
		device = torch.device("cuda")
	else: 
		stride = 25
		device = torch.device("cpu")
		
	# load model
	from cnn_simple import Net 

	# only load param 
	# model = Net()
	# model.load_state_dict(torch.load('trained.pth'))
	# or load the entire model
	model = torch.load('trained.pth')
	model.eval()

	# load the image
	path = sys.argv[1]
	mat = cv2.imread(path)
	img_wid = mat.shape[1]
	img_hgt = mat.shape[0]

	# run a window through the image to find highest score region
	# can improve using YOLO with muliti scale/coarse level window, 
	# and evaluate IOT 
	windows = []
	for j in range(0,img_hgt-2*l,stride):
	    for i in range(0,img_wid-2*l,stride):
	        window = mat[j:j+2*l,i:i+2*l]
	        window = np.rollaxis(window, 2, 0)
	        windows.append(window)

	# infer
	windows = Variable(torch.from_numpy(np.array(windows)).type(torch.FloatTensor)).to(device)
	scores = model(windows)
	score, max_idx = torch.max(scores, 0)

	# calculate the output
	row_grid = math.ceil((img_hgt-2*l)/stride)
	col_grid = math.ceil((img_wid-2*l)/stride)
	y = (math.ceil(max_idx/col_grid))*stride + l
	x = (max_idx.item()-math.ceil(max_idx/col_grid)*col_grid)*stride + l
	out_x = x/img_wid
	out_y = y/img_hgt

	print(str(round(out_x,4)),str(round(out_y,4)))

	# draw it!
	# cv2.circle(mat,(x,y), 2, (0,0,255), -1)
	# cv2.imshow('w',mat)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()





