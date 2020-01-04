import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.distributions import Categorical
import math
import numpy as np

class Conv_Agent_1(nn.Module):
	def __init__(self, input_features = 3, linear_features = 32, outs = 7):
		super(Conv_Agent_1,self).__init__()

		self.linear_features = linear_features

		self.input_features = input_features
		self.outs_features = outs

		self.conv1 = nn.Conv2d( 12,  16, (8, 8), stride = 4)
		self.conv2 = nn.Conv2d( 16,  32, (4, 4), stride = 2)
		self.linear1 = nn.Linear(6*6*32, linear_features)

		self.linear2 = nn.Linear(linear_features, linear_features)

		self.policy_head = nn.Linear(linear_features,  outs)

		self.initialize_weights()

		self.total_mask_size = 16 + 32 + (2 * linear_features) + outs

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
		net_ins = net_ins.permute((0,3,1,2))

		batch_size = len(net_ins)

		mask = ins['mask'].cuda()
		do_ensemble = ins['ensemble']

		#build conv1_mask
		conv1_mask = mask[:,:16].unsqueeze(2).unsqueeze(2)
		conv1_mask = conv1_mask.repeat((1,1,15,15))

		#build conv2_mask
		conv2_mask = mask[:,16:(16+32)].unsqueeze(2).unsqueeze(2)
		conv2_mask = conv2_mask.repeat((1,1,6,6))

		#build linear1_mask
		linear1_mask = mask[:,(16+32):(16+32+self.linear_features)]
		#build linear2_mask
		linear2_mask = mask[:,(16+32+self.linear_features):(16+32+self.linear_features*2)]

		#build policy_mask
		policy_mask = mask[:,(16+32+self.linear_features*2):]


		#First conv and mask
		v = F.relu(self.conv1(net_ins))
		#print("v",v.shape)
		v = v * conv1_mask

		#second conv and mask
		v = F.relu(self.conv2(v))
		v = v * conv2_mask

		v = v.view([batch_size, -1])


		v = F.relu(self.linear1(v))
		v = v * linear1_mask


		v = F.relu(self.linear2(v))
		v = v * linear2_mask

		v = self.policy_head(v) * policy_mask

		v = F.softmax(v)
		action = torch.argmax(v, dim = 1)

		action = action.cpu().numpy()

		return action

class Conv_Agent_1t(nn.Module):
	def __init__(self, input_features = 3, linear_features = 32, outs = 7, t_features = 4):
		super(Conv_Agent_1t,self).__init__()

		self.linear_features = linear_features

		self.input_features = input_features
		self.outs_features = outs
		self.t_features = t_features

		self.conv1 = nn.Conv2d( 12,  16, (8, 8), stride = 4)
		self.conv2 = nn.Conv2d( 16,  32, (4, 4), stride = 2)
		self.linear1 = nn.Linear(6*6*32 + t_features, linear_features)

		self.linear2 = nn.Linear(linear_features, linear_features)

		self.policy_head = nn.Linear(linear_features ,  outs)

		self.initialize_weights()

		self.total_mask_size = 16 + 32 + (2 * linear_features) + outs + t_features*3

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
		net_ins = net_ins.permute((0,3,1,2))

		batch_size = len(net_ins)

		mask = ins['mask'].cuda()
		do_ensemble = ins['ensemble']

		t = ins['t'].repeat((self.t_features,1)).permute((1,0)).cuda()

		#build conv1_mask
		conv1_mask = mask[:,:16].unsqueeze(2).unsqueeze(2)
		conv1_mask = conv1_mask.repeat((1,1,15,15))

		#build conv2_mask
		conv2_mask = mask[:,16:(16+32)].unsqueeze(2).unsqueeze(2)
		conv2_mask = conv2_mask.repeat((1,1,6,6))

		#t mask
		t_mask = mask[:,(16+32+self.linear_features*2+self.outs_features):]
		t_mask_offset = t_mask[:,:self.t_features]
		t_mask_rate = t_mask[:,self.t_features:2*self.t_features]
		t_mask_scale = t_mask[:,2*self.t_features:]

		#build linear1_mask
		linear1_mask = mask[:,(16+32):(16+32+self.linear_features)]
		#build linear2_mask
		linear2_mask = mask[:,(16+32+self.linear_features):(16+32+self.linear_features*2)]

		#build policy_mask
		policy_mask = mask[:,(16+32+self.linear_features*2):(16+32+self.linear_features*2)+self.outs_features]


		#First conv and mask
		v = F.relu(self.conv1(net_ins))
		#print("v",v.shape)
		v = v * conv1_mask

		#second conv and mask
		v = F.relu(self.conv2(v))
		v = v * conv2_mask

		v = v.view([batch_size, -1])

		t_v = t_mask_scale * torch.sin(t * t_mask_rate**3 + t_mask_offset)

		#t_v = 0 * t_v

		v = torch.cat([v, t_v], dim = 1)

		v = F.relu(self.linear1(v))
		v = v * linear1_mask

		v = F.relu(self.linear2(v))
		v = v * linear2_mask

		v = self.policy_head(v) * policy_mask

		v = F.softmax(v)
		action = torch.argmax(v, dim = 1)

		action = action.cpu().numpy()

		return action

