import cv2
import sys
import numpy as np 
import math
import torch
from torch.autograd import Variable

import util


if __name__ == '__main__':
	# var to change
	# stride when slide window on image
	# l is the fixed window size. In this problem I make this assumption
	# because the given training images seem to have fixed size of iphone in view
	l = 23
	# gpu options
	if torch.cuda.is_available():
		# stride choose between 1-46
		stride = 5
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
	path = sys.argv[1]
	model = torch.load('trained.pth')
	model.eval()
	acc = 0

	# load the image
	images,labels,names = util.read_dataset(path)
	for k,mat in enumerate(images):
		# mat = cv2.imread(img)
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

		# print(labels[k])
		act_x = labels[k][0]
		act_y = labels[k][1]

		err = math.sqrt((act_x-out_x)**2 + (act_y-out_y)**2)
		if (err<0.05):
			acc += 1
		else:
			# cv2.circle(mat,(x,y), 2, (0,0,255), -1)
			# cv2.imshow(names[k],mat)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
			continue
	print(acc/len(labels))

	# draw it!
	# cv2.circle(mat,(x,y), 2, (0,0,255), -1)
	# cv2.imshow('w',mat)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()





