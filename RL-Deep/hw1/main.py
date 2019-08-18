import tensorflow as tf
import numpy as np
import os
import gym
from model import Model
import load_policy
from train import Train, policy_rollout

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_data', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--num_rollouts', type=int, default=20, help='Number of expert roll outs')
    parser.add_argument('--learning_type', type=int, default=0, help='Type of Learning (0 for Behavior Cloning, 1 for DAgger)')
    args = parser.parse_args()
    
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit
    m = Model()
    if args.train:
        t = Train(args.learning_type)
        t.train(m, args.expert_policy_data, args.envname, max_steps)
    else:
        test_policies(m, env, args.envname, args.num_rollouts, max_steps, args.render)

def save_to_disk(filename, data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        np.save(f, data)

def get_policy_fn(policy_type, envname, model):
    if policy_type == 'expert':
        return load_policy.load_policy('experts/'+envname+'.pkl')
    elif policy_type == 'DAgger':
        return model.load_trained_policy(envname, 'DAgger')
    else:
        return model.load_trained_policy(envname, 'BC')
        

def test_policies(model, env, envname, num_rollouts, max_steps, render=False):
    policies = ['expert', 'BC', 'DAgger']

    #DAgger_policy_fn = model.load_trained_policy(envname, 'DAgger')
    #policy_fns = [expert_policy_fn, BC_policy_fn, DAgger_policy_fn]
    #policy_fns = [expert_policy_fn, BC_policy_fn]
    
    for ind, policy_type in enumerate(policies):
        policy_return = []
        tf.reset_default_graph()
        with tf.Session():
            policy_fn = get_policy_fn(policy_type, envname, model)
            print('______Training policy______ ',policy_type)
            for i in range(num_rollouts):
                print('iter', i)
                obs = env.reset()
                observations, actions, totalr = policy_rollout(policy_fn, env, obs, max_steps, render)
                policy_return.append(totalr)
        print('mean return', np.mean(policy_return))
        print('std of return', np.std(policy_return))
        save_to_disk('returns/'+envname+'/'+policy_type+'_Policy.npy', policy_return)


if __name__ == '__main__':
    main()
