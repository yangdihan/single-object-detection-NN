#!/usr/bin/env python
import cv2
import math
import numpy as np
# from PIL import Image

def read_dataset(path):
	images = []
	labels = []
	names = []
	with open(path+'/labels.txt','r') as raw_data:
		for line in raw_data:
			image_name,x_norm,y_norm = line.strip('\n').split(' ')
			names.append(image_name)
			image_value = cv2.imread(path+'/'+image_name)
			images.append(image_value)
			labels.append((float(x_norm),float(y_norm)))

	return images,labels,names


def mark_center(images,labels,names,note):
	""" 
		a tool function I used for myself
    """
	l_box = 23
	if not(len(images)==len(labels)==len(names)):
		print('need same amount of images and labels')
		return

	for i in range(len(labels)):
		# find center pixel idx
		x = int(labels[i][0]*images[i].shape[1])
		y = int(labels[i][1]*images[i].shape[0])
		# draw a red dot at center
		cv2.circle(images[i],(x,y), 2, (0,0,255), -1)
		# draw a box around center
		cv2.line(images[i],(x-l_box,y-l_box),(x-l_box,y+l_box),(0,255,0),2) 
		cv2.line(images[i],(x-l_box,y-l_box),(x+l_box,y-l_box),(0,255,0),2) 
		cv2.line(images[i],(x-l_box,y+l_box),(x+l_box,y+l_box),(0,255,0),2) 
		cv2.line(images[i],(x+l_box,y-l_box),(x+l_box,y+l_box),(0,255,0),2) 
		cv2.imwrite('./marker/'+names[i].strip('.jpg')+'_'+note+'.jpg',images[i])

	return



def crop_out(images,labels):
	""" 
		To get the data from images
    """
	l_box = 23

	if (len(images)!=len(labels)):
		print('need same amount of images and labels')
		return
	img_ph_list = []
	img_bg_list = []
	for i in range(len(labels)):
		# crop out the phone box
		img_wid = images[i].shape[1]
		img_hgt = images[i].shape[0]
		x = int(labels[i][0]*img_wid)
		y = int(labels[i][1]*img_hgt)
		img_phone = images[i][y-l_box:y+l_box, x-l_box:x+l_box]
		# in case a box is outside the image bound
		if (img_phone.shape==(2*l_box,2*l_box,3)):
			img_ph_list.append(img_phone)
			# cv2.imwrite('./test_ph/'+names[i].strip('.jpg')+'_phone.jpg',img_phone)

		# crop out the background boxes
		row_num_box = math.floor(img_hgt/(2*l_box))
		col_num_box = math.floor(img_wid/(2*l_box))
		for p in range(row_num_box):
			for q in range(col_num_box):
				# top-left pixel idx of background box
				m = int(p/row_num_box*img_hgt)
				n = int(q/col_num_box*img_wid)
				if not overlap(m,n,x,y,l_box):
					img_bg = images[i][m:m+2*l_box,n:n+2*l_box]
					# cv2.imwrite('./test_bg/'+names[i].strip('.jpg')+'-'+str(m)+','+str(n)+'.jpg',img_bg)
					img_bg_list.append(img_bg)

	return img_ph_list,img_bg_list


def overlap(m,n,x,y,l):
	# check whether the background box has any overlap with phone box
	x_min = x-l
	y_min = y-l
	x_max = x+l
	y_max = y+l
	if (m>=y_max or m+2*l<=y_min or n>=x_max or n+2*l<=x_min):
		return False
	return True


