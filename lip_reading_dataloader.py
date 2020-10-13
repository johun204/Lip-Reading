import torch
from torch.utils.data import Dataset, DataLoader

import pickle
import numpy as np

def get_imgs(s, u, k):
	with open("./datasets/s{0}/s{0}_v{1}_u{2}/{3}.pickle".format(s, 1, u, k), 'rb') as f:
		arr = pickle.load(f)
	return arr

class LipReadingDataset(Dataset):
	def __init__(self, slist, transform=None):
		self.slist = slist
		self.length = len(self.slist) * 30
		self.transform = transform

	def __getitem__(self, index):
		s, u = self.slist[index // 30], index % 30 + 31
		result = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

		x = torch.as_tensor([get_imgs(s, u, k) for k in range(16)]).float()
		y = torch.as_tensor([[result[(u - 31) // 3] for i in range(x.shape[1])] for k in range(16)]).long()

		if self.transform:
			x = self.transform(x)

		return index, x, y

	def __len__(self):
		return self.length