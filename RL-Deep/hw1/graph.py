import matplotlib.pyplot as plt
import os
import numpy as np

envs = ['Hopper-v2', 'Ant-v2', 'HalfCheetah-v2', 'Humanoid-v2', 'Reacher-v2', 'Walker2d-v2']
script_dir = os.path.dirname(__file__)

fig, ax = plt.subplots(3,2, figsize=(10,10))
fig.suptitle('Behaviour Cloning Results', fontsize=16)
i,j=0,0
for env in envs:
    np_file_path = os.path.join(script_dir,'returns/'+env)
    expert_policy = np.load(np_file_path+'/expert_policy.npy' ,allow_pickle=True).item()
    cloned_policy = np.load(np_file_path+'/modelled_policy.npy', allow_pickle=True).item()
    ax[i][j].plot(expert_policy['returns'], label='Expert Policy')
    ax[i][j].plot(cloned_policy['returns'], label='Cloned Policy')
    ax[i][j].set_title(env)
    ax[i][j].legend()
    j = j+1 if j+1 < 2 else 0
    i = i+1 if j==0 else i
plt.savefig('Behavior_cloning.png')
plt.show()
    
