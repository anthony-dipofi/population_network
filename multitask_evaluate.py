import gym
import torch
from multiprocessing import Process, Pipe
import numpy as np
from collections import namedtuple
import scipy
from PIL import Image


def multitask_evaluate(agent, mask, make_env, num_procs, trials = 1):
	t_rew = np.zeros([num_procs])
	dones = np.zeros([num_procs])

	for t in range(trials):
		procs = []
		#start processes
		for env_idx in range(num_procs):
			eval_pipe, env_pipe = Pipe()
			proc = Process(target = env_proc, args=(make_env, env_pipe))
			proc.start()
			procs.append({'proc':proc, 'pipe':eval_pipe, 'done':False})

		#receive intial obs
		os = []
		active_idxs = []
		for env_idx in range(len(procs)):
			p = procs[env_idx]
			o = p['pipe'].recv()
			if p['done'] == False:
				os.append(torch.tensor(o['obs']))
				active_idxs.append(env_idx)
		obs = torch.stack(os, dim = 0).squeeze()
		ts = torch.zeros(num_procs)
		
		all_done = False
		while all_done == False:
			#Evaluate obs
			#print(obs.shape)
			m = mask[active_idxs]
			net_ins = {'obs':obs.float(), 'mask':m, 'ensemble':False, 't':ts}
			actions = agent(net_ins)
			
			#Send action
			for a_idx in range(len(active_idxs)):
				procs[active_idxs[a_idx]]['pipe'].send(actions[a_idx])

			#receive new env data
			os = []
			active_idxs = []
			rews = np.zeros([num_procs])
			ts = []
			for env_idx in range(len(procs)):
				p = procs[env_idx]

				#If the process is still enabled
				if p['done'] == False:

					o = p['pipe'].recv()
					#If the env didn't finish this step
					if o['done'] == False:
						#Process observations and rewards
						os.append(torch.tensor(o['obs']))
						rews[env_idx] = o['rew']
						ts.append(torch.tensor([o['t']]))
						#populate list for mapping between NN batch and environment processes
						active_idxs.append(env_idx)
					else: #Disable proc of env is complete
						p['done'] = True

			t_rew += rews

			#process dones
			check_done = True
			for env_idx in range(num_procs):
				if (procs[env_idx]['done'] == False):
					check_done = False
			all_done = check_done

			if (all_done == True):
				break

			obs = torch.stack(os, dim = 0).squeeze()
			ts = torch.stack(ts, dim = 0).float().squeeze()

		
		#Close env processes
		for env_idx in range(num_procs):
			procs[env_idx]['proc'].join()

	return t_rew/trials

def env_proc(make_env, pipe):
	fps = 15
	max_steps = fps * 60 * 4
	env = make_env()
	done = False
	obs = np.array(env.reset())
	#Image.fromarray(obs).resize((64,64))
	#obs = imresize(obs,(64,64))
	#print("obs",obs.shape)
	o = {"obs":obs, "rew":0, "done":False, "info":{}}
	pipe.send(o)
	step = 0
	max_x = 0
	old_info = {}
	old_info['lives'] = None
	old_info['rings'] = None
	max_rings = 0
	end_x = None
	while done == False:

		action = pipe.recv()
		
		obs, rew, done, info = env.step(action)
		obs = np.array(obs)
		#obs = scipy.misc.imresize(obs,(64,64))

		#x_loc = info['x']
		if end_x is None:
			end_x = info['screen_x_end']

		'''x_loc = info['screen_x'] / end_x
		if(x_loc > max_x):
			rew = (x_loc - max_x) * 10000
			max_x = x_loc
		else:
			rew = 0'''

		if step > max_steps:
			done = True

		if old_info['lives'] is None:
			old_info = info

		if info['lives'] < old_info['lives']:
			done = True

		rew = 0
		if info['lives'] > old_info['lives']:
			rew += 1
			max_rings = 0

		if info['rings'] > max_rings:
			rew += (info['rings'] - max_rings)
			max_rings = info['rings']
		
		old_info = info

		if info['act'] > 0:
			done = True

		#rew += info['level_end_bonus']

		#rew = 0
		#if done == True:
			#rew += info['rings'] * 100
			#rew += info['score']

		o = {"obs":obs, "rew":rew, "done":done, "info":info, "t":step}

		step += 1


		pipe.send(o)

	env.close()