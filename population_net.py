import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.distributions import Categorical
import math
import numpy as np



class Pop_net(nn.Module):
	def __init__(self, input_features = 3, linear_features = 32, outs = 2):
		super(FC_Agent_2Deep,self).__init__()

		self.linear_features = linear_features

		self.input_features = input_features
		self.outs_features = outs

		self.linear1 = nn.Linear(input_features,  linear_features)
		self.linear2 = nn.Linear(linear_features,  linear_features)
		self.policy_head = nn.Linear(linear_features,  outs)

		self.initialize_weights()

        self.masks = {}
        self.mask_schema['linear_mask_1'] = {'size': linear_features, 'type': "Binary", 'mutation_rate': 0.05}
        self.mask_schema['linear_mask_2'] = {'size': linear_features, 'type': "Binary", 'mutation_rate': 0.05}
        self.mask_schema['action_mask']   = {'size': outs,            'type': "Scalar", 'mutation_rate': 0.05, 'max_val':5}

		self.cuda()


    def mutate_mask(self, mask):
        original_mask = mask
        for s in self.mask_schema:
            mask_type = self.mask_schema[s]['type']
            features = self.mask_schema[s]['size']
            mutate_rate = self.mask_schema[s]['mutation_rate']
            if mask_type == "Binary":
                flip = (torch.FloatTensor(features).uniform_() < mutate_rate ).float()
				mask[s] = torch.abs(mask - flip)

            elif mask_type == "Scalar":                
				mask[s] = mask + mutate_rate * torch.normal(torch.zeros(features), torch.ones(features))
				mask[s] = torch.clamp(mask[s], 0, self.mask_schema[s]['max_val'])

            elif mask_type == "Trinary":
                mask[s]

        return mask
		

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

		v = v * mask['linear_mask_1']

		v = F.relu(self.linear2(v))
		v = v * mask['linear_mask_2']

        v = self.policy_head(v)
        v = v * mask['action_mask']

		action = -2 + 4 * F.softmax()[:,0].unsqueeze(1)

		#print("action", action)

		#action = 2 * F.tanh(self.policy_head(v))

		action = action.cpu().numpy()
		#action = torch.argmax(F.softmax(self.policy_head(v))) - 1

		#action = int(action)

		return action
