import matplotlib.pyplot as plt
import os
import numpy as np
from itertools import chain

envs = ['Hopper-v2', 'Ant-v2', 'HalfCheetah-v2', 'Humanoid-v2', 'Reacher-v2', 'Walker2d-v2']
script_dir = os.path.dirname(__file__)
policies = ['expert', 'BC', 'DAgger']
fig, ax = plt.subplots(3,2, figsize=(10,10))
fig.suptitle('IMITATION LEARNING Results', fontsize=16)
i,j=0,0
for env in envs:
    np_file_path = os.path.join(script_dir,'returns/'+env)
    for policy_type in policies:
        policy = np.load(np_file_path+'/'+policy_type+'_Policy.npy' ,allow_pickle=True)
        print(policy)
        ax[i][j].plot(policy, label=policy_type)
    ax[i][j].set_title(env)
    ax[i][j].legend()
    j = j+1 if j+1 < 2 else 0
    i = i+1 if j==0 else i
plt.savefig('Results/Imitation_Learning/plot.png')
plt.show()
    
