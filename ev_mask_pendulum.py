from rs_agent import *
from evaluate import *

import random
import copy

import gym
from bisect import insort, bisect_right


def gen_random_agent(features = 64, initial_noise_scale = 0.5):
	
	linear_features = features
	
	input_size = 3
	agent = FC_Agent_2Deep(input_size, linear_features)
	agent.initialize_weights(0.5)

	return agent


def gen_binary_mask(population_size, features):
	#mask = (torch.FloatTensor(population_size, features).uniform_() > 0.5).float()
	#mask = (torch.FloatTensor(features).uniform_() > 0.5).float()
	#mask = 4 * torch.FloatTensor( features).uniform_() - 2
	mask = 1.2 * torch.normal(torch.zeros(features), torch.ones(features))
	mask = torch.clamp(mask, -5, 5)
	return mask

def evolve_agents(population, num_envs, features, noise_rate = 0.1):

	roll = random.randint(0, 100)

	masks = []

	for env_idx in range(num_envs):
		if (roll > 20 and len(population) > 1):

			parent_idx = random.randint(0, len(population)-1)
			parent_mask = population[parent_idx]['mask']

			mask = copy.deepcopy(parent_mask)

			with torch.no_grad():
				mask = mask + noise_rate * torch.normal(torch.zeros(features), torch.ones(features))
				mask = torch.clamp(mask, -5, 5)
				#flip = (torch.FloatTensor(features).uniform_() < noise_rate).float()
				#	mask = torch.abs(mask - flip)
				#reroll = 4 * torch.FloatTensor(features).uniform_() - 2
				#mask = mask * (1 - flip) +  reroll * flip
			
		else:
			mask = gen_binary_mask(1, features)
		masks.append(mask)

	mask = torch.stack(masks, dim = 0)

	return mask


def update_top(agent, top_agents, top_size):
	if len(top_agents) < top_size:
		top_agents.append(agent)

	for i in range(len(top_agents)):
		if agent['score'] > top_agents[i]['score']:
			top_agents.append(agent)
			break
	
	if len(top_agents) > top_size:
		lowest = 0
		for i in range(len(top_agents)):
			if top_agents[i]['score'] < top_agents[lowest]['score']:
				lowest = i
		new_top_agents = []
		for j in range(len(top_agents)):
			if j != lowest:
				new_top_agents.append(top_agents[j])
		
		top_agents = new_top_agents

	return top_agents


def new_env():
	env = gym.make("Pendulum-v0")
	return env

if __name__ == '__main__':

	save_name = "ev_mask1"
	generations = 400
	agents = []
	top_3_agents = []
	top_5_agents = []
	top_10_agents = []
	top_25_agents = []
	scores = []
	best_agent = None
	best_score = -1000000

	#population_size = 2
	num_envs = 12
	agent_features = 128
	noise_rate = 0.2

	substrate_evo_step = 10

	initial_noise_scale = 0.4
	agent_net = gen_random_agent(features = agent_features, initial_noise_scale = initial_noise_scale)


	for gen in range(generations):
		
		#env = new_env()
		
		mask = evolve_agents(top_25_agents, num_envs, agent_features*2, noise_rate)

		with torch.no_grad():
			mask_scores = multienv_evaluate(agent_net, mask, new_env, num_envs = num_envs, trials = 10)
			#score = evaluate(agent_net, mask, new_env, trials = 10)
		for i in range(num_envs):
			age = {"mask":mask[i], "score":mask_scores[i]}
			score = mask_scores[i]

			insort(scores, score)
			rank = bisect_right(scores, score)

			top_3_agents = update_top(age, top_3_agents, 3)
			top_5_agents = update_top(age, top_5_agents, 5)
			top_10_agents = update_top(age, top_10_agents, 10)
			top_25_agents = update_top(age, top_25_agents, 4)

			pop_average = sum([a['score'] for a in top_25_agents])/len(top_25_agents)
			print("trial: ", gen, "score:",score, "rank: "+str(rank)+"/"+str(len(scores)), "pop_average:",pop_average)

			if(score > best_score):
					best_mask = mask
					best_score = score
		if gen % substrate_evo_step == 0: 
			print("mask")
			for i in range(len(top_25_agents)):
				print(i, top_25_agents[i]['score'], top_25_agents[i]['mask'], )

	torch.save(best_agent, save_name + "_top.pth")

	ensemble3 = Agent_Ensemble([a['agent'] for a in top_3_agents])

	torch.save(ensemble3, save_name + "_3_ensemble.pth")
	ensemble5 = Agent_Ensemble([a['agent'] for a in top_5_agents])

	torch.save(ensemble5, save_name + "_5_ensemble.pth")
	ensemble10 = Agent_Ensemble([a['agent'] for a in top_10_agents])

	torch.save(ensemble10, save_name + "_10_ensemble.pth")


