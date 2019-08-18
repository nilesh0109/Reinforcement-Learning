import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
import gym
import load_policy
import tensorflow as tf

class Train:
    def __init__(self, alg_type=0):
        self.alg_type = alg_type
        
    def train(self, model, expert_policy_data, envname, max_episode_len):
        script_dir = os.path.dirname(__file__)
        with open(os.path.join(script_dir, expert_policy_data), 'rb') as f:
            data = pickle.load(f)
        
        x = data['observations']
        y = data['actions'].reshape(-1, data['actions'].shape[-1])
        X_train, X_test, y_train, y_test =train_test_split(x, y, test_size=0.25)
        model.create(x.shape[-1], y.shape[-1])
        model.lr_rate = 1e-2
        model.reg_lambda = 1e-4
        if self.alg_type == 0:
            self.__BehaviorCloning(model, X_train, y_train, envname)
        else:
            self.__Dagger(model, X_train, y_train, envname, max_episode_len)

    def __BehaviorCloning(self, model, X_train, y_train, envname):
        model.train(X_train, y_train, 20000, env_name=envname, lr_type='BC')

    def __Dagger(self, model, X_train, y_train, envname, max_steps):
        for _ in range(10):
            print('{:-^100} iteration {:d}, trainingsetSize={:d}'.format('Dagger model Training', _, X_train.shape[0]))
            model.train(X_train, y_train, 2000, env_name=envname, lr_type='DAgger')
            env = gym.make(envname)
            init_obs = env.reset()

            policy_observations = []
            total_policy_return = 0
            tf.reset_default_graph()
            with tf.Session() as sess:
                trained_policy_fn = model.load_trained_policy(envname, lr_type='DAgger')
                for i in range(20):
                    policy_obs, policy_act, t_return  = policy_rollout(trained_policy_fn, env, init_obs, max_steps)
                    policy_observations = policy_obs if i==0 else np.append(policy_observations,policy_obs, axis=0)
                    total_policy_return += t_return
                print('Mean Modelled Policy Return = {:f}'.format(total_policy_return/20))

                expert_observations = []
                expert_actions = []
                r_total = 0
                expert_policy_fn = load_policy.load_policy('experts/'+envname+'.pkl')
                for j,obs in enumerate(policy_observations):
                    expert_obs, expert_act, expert_rt = policy_rollout(expert_policy_fn, env, obs, max_steps)
                    expert_observations = expert_obs if j==0 else np.append(expert_observations, expert_obs, axis=0)
                    expert_actions = expert_act if j==0 else np.append(expert_actions, expert_act, axis=0)
                    r_total += expert_rt

            print('Mean Expert Policy Return = {:f}'.format(r_total/20))
            X_train = np.append(X_train, expert_observations, axis=0)
            y_train = np.append(y_train, np.squeeze(expert_actions, axis=1), axis=0)
            #squeezing action to get rid of extra dimension the expert polict added
            tf.reset_default_graph()
            model.load_graph_from_ckpt(envname, 'DAgger')

def policy_rollout(policy_fn, env, obs, max_steps, render=False):
    done = False
    totalr = 0.0
    steps = 0
    actions, observations  = [], []
    while not done:
        action = policy_fn(obs[None,:])
        observations.append(obs)
        actions.append(action)
        obs, r, done, _ = env.step(action)
        totalr += r
        steps += 1
        if render:
            env.render()
        if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
        if steps >= max_steps:
            break
    return observations, actions, totalr




