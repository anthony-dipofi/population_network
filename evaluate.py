import gym
import torch
from multiprocessing import Process, Pipe
import numpy as np

def evaluate(agent, mask, env, trials = 1, render = False):
    t_rew = 0
    agent.reset()
    for t in range(trials):
        done = False
        obs = env.reset()
        while done == False:
            #print("o", obs)
            obs = torch.tensor(obs).squeeze().unsqueeze(0).float()
            net_ins = {'obs':obs, 'mask':mask, 'ensemble':False}
            action = agent(net_ins)
            #print("a",action)
            obs, rew, done, info = env.step(action)
            obs = obs
            #print("o2", obs)
            if(render):
                env.render()
            t_rew += rew
        env.close()
    return t_rew/trials

def multienv_evaluate(agent, mask, make_env, num_envs, trials = 1):
    t_rew = np.zeros([num_envs])

    for t in range(trials):
        procs = []
        #start processes
        for env_idx in range(num_envs):
            eval_pipe, env_pipe = Pipe()
            proc = Process(target = env_proc, args=(make_env, env_pipe))
            proc.start()
            procs.append({'proc':proc, 'pipe':eval_pipe})

        #recieve intial obs
        obs = torch.stack([torch.tensor(p['pipe'].recv()['obs'])for p in procs], dim = 0).squeeze() 

        done = False
        while done == False:
            #Evaluate obs
            #print(obs.shape)
            net_ins = {'obs':obs.float(), 'mask':mask, 'ensemble':False}
            actions = agent(net_ins)
            
            #Send action
            for env_idx in range(num_envs):
                procs[env_idx]['pipe'].send(actions[env_idx])

            #receive new env data
            env_obs = [p['pipe'].recv() for p in procs]

            #process obs
            obs = torch.stack([torch.tensor(e['obs']) for e in env_obs], dim = 0).squeeze() 

            #process rews
            rews = np.zeros([num_envs])
            for env_idx in range(num_envs):
                rews[env_idx] += env_obs[env_idx]['rew']
            
            t_rew += rews
            
            #process dones
            for env_idx in range(num_envs):
                if (env_obs[env_idx]['done'] == True):
                    done = True

        #Close env processes
        for env_idx in range(num_envs):
            procs[env_idx]['proc'].join()

        

    return t_rew/trials

def env_proc(make_env, pipe):
    env = make_env()
    done = False
    obs = env.reset()
    o = {"obs":obs, "rew":0, "done":False, "info":{}}
    pipe.send(o)
    while done == False:

        action = pipe.recv()
        
        obs, rew, done, info = env.step(action)
        o = {"obs":obs, "rew":rew, "done":done, "info":info}

        pipe.send(o)

    env.close()


