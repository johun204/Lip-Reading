import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from tqdm import tqdm
import numpy as np

from model import *
from lip_motion_dataloader import LipMotionDataset

def train(model_N = 0, _LEARNING_RATE = 0.01, _BATCH_SIZE = 16):

	_START_EPOCH = 0
	_TOTAL_EPOCH = 150
	
	_MOMENTUM = 0.6
	input_dim, hidden_dim, num_layers = 256, 100, 2

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	
	
	trainset = [1, 2, 3, 4, 5, 7, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 45, 46, 47, 48, 50, 53]
	train_dataset = LipMotionDataset(trainset)
	data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)

	model_C3DLSTM = C3DLSTM(input_dim, hidden_dim, _BATCH_SIZE, num_layers).to(device)
	if model_N > 0:
		checkpoint = torch.load('./model/model{}.pt.tar'.format(model_N))
		model_C3DLSTM.load_state_dict(checkpoint['model_state_dict'])	
	print(model_C3DLSTM)
	model_C3DLSTM.train()
	

	optimizer = optim.SGD(model_C3DLSTM.parameters(), lr=_LEARNING_RATE, momentum=_MOMENTUM, weight_decay=0.00155)
	criterion = nn.CrossEntropyLoss().to(device)

	if model_N > 0:
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		_START_EPOCH = checkpoint['epoch']
		print('/model/model{}.pt.tar loaded successfully.'.format(model_N))
		print("epoch : {0}".format(checkpoint['epoch']))
		print("loss : {0}".format(checkpoint['loss']))


	h0 = Variable(torch.zeros(num_layers, _BATCH_SIZE, hidden_dim).float().cuda())
	c0 = Variable(torch.zeros(num_layers, _BATCH_SIZE, hidden_dim).float().cuda())
	init_hidden = (h0, c0)


	for ep in tqdm(range(_TOTAL_EPOCH)):
		if ep < _START_EPOCH: continue
		lr = _LEARNING_RATE
		for i in range(int(ep / 30)): lr = lr / 2
		for param_group in optimizer.param_groups: param_group['lr'] = lr
		print("\nLearning Rate : {0}".format(lr))

		if ep+1 >= 50: 
			for param_group in optimizer.param_groups: param_group['momentum'] = 0.9

		running_loss, loss_cnt = 0.0, 0
		save_loss = 0
		for i, (idx, image, label) in tqdm(enumerate(data_loader)):
			image = image.view(16, -1, 3, 16, 32)
			label = label.view(16, -1)
			hidden = init_hidden
			optimizer.zero_grad()
			batch_loss = 0
			
			for s, image_s in enumerate(torch.split(image, 1, dim=1)):
				image_s = image_s.view(16, 3, 16, 32)
				lb = label[:,s]
				output, hidden = model_C3DLSTM(image_s.to(device), hidden) #(ep+1 >= 15)
				loss = criterion(output, lb.to(device))
				batch_loss += loss

			
			batch_loss.backward()
			optimizer.step()

			running_loss += batch_loss.item()
			loss_cnt += 1
			if loss_cnt % 100 == 0:
				print('[%d, %5d] loss: %.3f' % (ep + 1, i + 1, running_loss / loss_cnt))
				save_loss = running_loss / loss_cnt
				running_loss, loss_cnt = 0.0, 0
		if loss_cnt > 0: print('[%d] loss: %.3f' % (ep + 1, running_loss / loss_cnt))
		torch.save({'epoch': (ep + 1), 'model': model_C3DLSTM, 'model_state_dict': model_C3DLSTM.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'optimizer': optimizer, 'loss': save_loss}, './model/model{}.pt.tar'.format(ep + 1))


def eval(model_N = 0, _VIDEO=False, _MODEL_PATH):

	
	_BATCH_SIZE = 16

	input_dim, hidden_dim, num_layers = 256, 100, 2

	validset = [6, 8, 9, 15, 26, 30, 34, 43, 44, 49, 51, 52]
	classes = {0: 'Excuse me', 1: 'Goodbye', 2: 'Hello', 3: 'How are you', 4: 'Nice to meet you', 5: 'See you', 6: 'I am sorry', 7: 'Thank you', 8: 'Have a good time', 9: 'You are welcome'}

	testset = LipMotionDataset(validset)
	data_loader = DataLoader(testset, batch_size=1, shuffle=False)

	

	with torch.no_grad():
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		model_C3DLSTM = C3DLSTM(input_dim, hidden_dim, _BATCH_SIZE, num_layers).to(device)
		checkpoint = torch.load(_MODEL_PATH)
		model_C3DLSTM.load_state_dict(checkpoint['model_state_dict'])
		model_C3DLSTM.eval()
		print("epoch : {0}".format(checkpoint['epoch']))
		print("loss : {0}".format(checkpoint['loss']))
		

		h0 = Variable(torch.zeros(num_layers, _BATCH_SIZE, hidden_dim).float().cuda())
		c0 = Variable(torch.zeros(num_layers, _BATCH_SIZE, hidden_dim).float().cuda())
		init_hidden = (h0, c0)


		class_correct = [0 for i in range(10)]
		class_total = [0 for i in range(10)]


		for tidx, (idx, image, label) in enumerate(tqdm(data_loader)):
			tmp1 = list()
			hidden = init_hidden
			idx = idx.item()
			s, u = testset.slist[idx // 30], idx % 30 + 31
			
			image = image.view(16, -1, 3, 16, 32)
			label = label.view(16, -1)
			for sidx, image_s in enumerate(torch.split(image, 1, dim=1)):
				image_s = image_s.view(16, 3, 16, 32)
				lb = label[:,sidx]
				output, hidden = model_C3DLSTM(image_s.to(device), hidden)

				_, predicted = torch.max(output, 1)
				c = (predicted == lb.to(device)).squeeze()
				
				for k in range(16):
					class_correct[lb[k].item()] += c[k].item()
					class_total[lb[k].item()] += 1




	file = open('./{0}/model{1}.txt'.format(_MODEL_NAME, model_N),'w', encoding='utf-8')
	file.writelines("{0}\n".format(checkpoint['model']))
	file.writelines("{0}\n".format(checkpoint['optimizer']))
	file.writelines("epoch : {0}\n".format(checkpoint['epoch']))
	file.writelines("loss : {0}\n".format(checkpoint['loss']))

	total_correct, total = 0.0, 0.0
	for i in range(10):
		file.writelines('Accuracy of "{0}" ({1}) : {2} [{3} / {4}]\n'.format(classes[i], i, 100 * class_correct[i] / class_total[i], class_correct[i], class_total[i]))
		total_correct += class_correct[i]
		total += class_total[i]
	print('Total Accuracy : {0} [{1} / {2}]\n'.format(100 * total_correct / total, total_correct, total))
	file.writelines('Total Accuracy : {0} [{1} / {2}]\n'.format(100 * total_correct / total, total_correct, total))
