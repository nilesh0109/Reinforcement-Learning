import tensorflow as tf
import numpy as np
import pickle
import os
import gym
from sklearn.model_selection import train_test_split
from model import Model
import load_policy
import time

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_data', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()
    
    
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit
    obs = env.reset()
    action_shape = env.action_space.sample().shape
    m = Model()
    if args.train:
        script_dir = os.path.dirname(__file__)

        with open(os.path.join(script_dir,args.expert_policy_data), 'rb') as f:
            data = pickle.load(f)
        
        x = data['observations']
        y = data['actions'].reshape(-1, data['actions'].shape[-1])
        X_train, X_test, y_train, y_test =train_test_split(x, y, test_size=0.25)
        m.lr_rate = 1e-2
        m.reg_lambda = 1e-4
        m.create(obs.shape[-1], action_shape[-1])
        m.train(X_train, y_train, 5000, env_name=args.envname)
    
    else:
        expert_policy_fn = load_policy.load_policy('experts/'+args.envname+'.pkl')
        expert_policy = {'returns': []}
        trained_policy = {'returns': []}
        with tf.Session():
            for i in range(args.num_rollouts):
                print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    action = expert_policy_fn(obs[None,:])
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1
                    if args.render:
                        env.render()
                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break
                expert_policy['returns'].append(totalr)
        
        print(expert_policy)
        print('mean return', np.mean(expert_policy['returns']))
        print('std of return', np.std(expert_policy['returns']))
        save_to_disk('returns/'+args.envname+'/expert_policy.npy', expert_policy)

        trained_policy = {'returns': []}
        
        with tf.Session():
            trained_policy_fn = m.load_trained_policy(args.envname)
            for i in range(args.num_rollouts):
                print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    action = trained_policy_fn(obs[None,:])
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1
                    if args.render:
                        env.render()
                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break
                trained_policy['returns'].append(totalr)
        print(trained_policy)    
        print('mean return', np.mean(trained_policy['returns']))
        print('std of return', np.std(trained_policy['returns']))
        save_to_disk('returns/'+args.envname+'/modelled_policy.npy', trained_policy) 

def save_to_disk(filename, data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        np.save(f, data)


if __name__ == '__main__':
    main()


                
