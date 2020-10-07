import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F



class C3DLSTM(nn.Module):
	def __init__(self, input_dim=256, hidden_dim=100, bs=16, nl=1):
		super(C3DLSTM, self).__init__()

		self.hidden_dim = hidden_dim
		self.input_dim = input_dim
		self.bs = bs
		self.nl = nl

		self.conv1 = nn.Conv2d(3, 6, kernel_size=(3, 7)) #6@14x26
		self.pool1 = nn.MaxPool2d(2) #6@7x13
		self.conv2 = nn.Conv2d(6, 16, kernel_size=(3, 6)) #16@5x8
		self.conv3 = nn.Conv2d(16, 64, kernel_size=(4, 7)) #64@2x2
		self.bn1 = nn.BatchNorm2d(64)

		self.lstm = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, num_layers = nl, batch_first=True, dropout=0.5)
		self.linear = nn.Linear(hidden_dim, 10, bias=True)

		self.dropout = nn.Dropout(p=0.5)
		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()
		self.softmax = nn.LogSoftmax(dim=1)


	def forward(self, image, hidden):

		out = self.relu(self.conv1(image))
		out = self.pool1(out)
		out = self.relu(self.conv2(out))
		out = self.relu(self.conv3(out))
		out = self.bn1(out)
		
		out = self.dropout(out)

		out = out.view(self.bs, -1, self.input_dim)

		out, hidden = self.lstm(out, hidden)
		out = out.view(-1, self.hidden_dim)

		out = self.linear(self.tanh(out))
		
		return self.softmax(out), hidden