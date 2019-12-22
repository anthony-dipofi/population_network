import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.distributions import Categorical
import math
import numpy as np

class FC_Agent(nn.Module):
	def __init__(self, input_features = 3, linear_features = 32, outs = 1):
		super(FC_Agent,self).__init__()

		self.linear_features = linear_features

		self.input_features = input_features
		self.outs_features = outs

		self.linear1 = nn.Linear(input_features,  linear_features)
		self.policy_head = nn.Linear(linear_features,  outs)

		self.initialize_weights()

		self.cuda()


		

	def initialize_weights(self, std=0.1):
		for i in range(1,len(list(self.modules()))):
			l = list(self.modules())[i]
			for p in l.parameters():
				p = std* torch.normal(torch.zeros(p.shape), torch.ones(p.shape))

	def reset(self):
		pass


	def forward(self, ins):

		net_ins = ins['obs'].cuda()
		mask = ins['mask'].cuda()
		do_ensemble = ins['ensemble']

		#print("ni",net_ins)

		v = F.relu(self.linear1(net_ins))

		v = v * mask

		action = 2 * F.tanh(self.policy_head(v))

		action = action.cpu().numpy()
		#action = torch.argmax(F.softmax(self.policy_head(v))) - 1

		#action = int(action)

		return action

class FC_Agent_2Deep(nn.Module):
	def __init__(self, input_features = 3, linear_features = 32, outs = 2):
		super(FC_Agent_2Deep,self).__init__()

		self.linear_features = linear_features

		self.input_features = input_features
		self.outs_features = outs

		self.linear1 = nn.Linear(input_features,  linear_features)
		self.linear2 = nn.Linear(linear_features,  linear_features)
		self.policy_head = nn.Linear(linear_features,  outs)

		self.initialize_weights()

		self.cuda()


		

	def initialize_weights(self, std=0.1):
		for i in range(1,len(list(self.modules()))):
			l = list(self.modules())[i]
			for p in l.parameters():
				p = std* torch.normal(torch.zeros(p.shape), torch.ones(p.shape))

	def reset(self):
		pass


	def forward(self, ins):

		net_ins = ins['obs'].cuda()
		mask = ins['mask'].cuda()
		do_ensemble = ins['ensemble']

		#print("ni",net_ins)

		v = F.relu(self.linear1(net_ins))

		#print("mask", mask, mask.shape)
		#print("v",v.shape)

		v = v * mask[:,:self.linear_features]

		v = F.relu(self.linear2(v))
		v = v * mask[:,self.linear_features:]

		action = -2 + 4 * F.softmax(self.policy_head(v))[:,0].unsqueeze(1)

		#print("action", action)

		#action = 2 * F.tanh(self.policy_head(v))

		action = action.cpu().numpy()
		#action = torch.argmax(F.softmax(self.policy_head(v))) - 1

		#action = int(action)

		return action
