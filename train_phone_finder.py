#!/usr/bin/env python
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable

import util
import data_loader

def main():
	# training params
	lr = 0.0001
	root = sys.argv[1]

	# gpu options
	if torch.cuda.is_available():
		# stride choose between 1-46
		num_epochs = 10
		batch_size = 10
		device = torch.device("cuda")
	else: 
		num_epochs = 1
		batch_size = 4
		device = torch.device("cpu")

	# test if gpu should be used
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# data augmentation transformations
	train_transformations = transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.RandomCrop(32,padding=4),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	# train dataset loader
	train_data_loader = data_loader.TrainingSetLoader(root, train_transformations)
	train_loader = torch.utils.data.DataLoader(dataset=train_data_loader,
													batch_size=batch_size,
													shuffle=True)
	# can add test loader here


	from cnn_simple import Net 
	model = Net().to(device)
	# can also use SGD and add momentum
	optimizer = optim.Adam(model.parameters(), lr=lr)
	criterion = nn.BCELoss()

	model.train()
	for epoch in range(num_epochs):
		# Randomly shuffle data every epoch
		train_accu = []
		for batch_idx, (data, target) in enumerate(train_loader, 0):
			data = data.type(torch.FloatTensor)
			target = target.type(torch.FloatTensor)
			data, target = Variable(data), Variable(target)
			data, target = data.to(device), target.to(device)

			optimizer.zero_grad()
			output = model(data)
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()

			prediction = (output >= 0.5).type(torch.FloatTensor).to(device) # first column has actual prob.
			accuracy =  (prediction.eq(target.view(prediction.shape)).sum().item()/batch_size) * 100
			train_accu.append(accuracy)

		accuracy_epoch = np.mean(train_accu)
		print(str(epoch+1), accuracy_epoch)
	
	# only save the params
	# torch.save(model.state_dict(), 'trained.pth')
	# save the whole model
	torch.save(model, 'trained.pth')
	print('======= weights saved as trained.pth =======')

if __name__ == '__main__':
	main()