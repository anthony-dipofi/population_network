import ev_mask_sonic
from ev_sonic_agent import *
from evaluate import *
from multitask_evaluate import *

import random
import copy

import gym
import retro

import time

import sonic_util
from baselines.common.atari_wrappers import WarpFrame, FrameStack


from bisect import insort, bisect_right


net_name = "ev_sonic1.pth"
#net_name = "ev_sonic3_score.pth"
net_name = "ev_sonic4_t.pth"
net_name = "ev_sonic5_noise_cyc.pth"
net_name = "ev_sonic5_noise_cyc.pth"
net_name = "ev_sonic7_noise_cyc_rings.pth"
net_name = "ev_sonic8_noise_cyc_rings.pth"

net = torch.load(net_name)

pop = net.population

best = None
best_mask = None
for p in pop:
	if best is None or p['score'] > best:
		best_mask = p['mask']
		best = p['score']

m = best_mask.unsqueeze(0)


fps = 15
max_steps = fps * 60 * 1

for p in pop:
	#m = p['mask'].unsqueeze(0)

	env = ev_mask_sonic.new_env()

	obs = env.reset()
	env.render()
	time.sleep(5)



	step = 0
	done = False
	while done == False:
		obs = torch.tensor(np.array(obs)).unsqueeze(0)
		t = torch.tensor(step)
		net_ins = {'obs':obs.float(), 'mask':m, 'ensemble':False, 't':t}


		action = net(net_ins)

		#print("a", int(action))
		obs, rew, done, info = env.step(np.array(int(action)))

		#print(step, info)

		if info['level_end_bonus'] > 0:
			done = True

		
		if step > max_steps:
			done = True

		env.render()
		step += 1
		time.sleep(0.02)
	
	env.render(close=True)
	env.close()