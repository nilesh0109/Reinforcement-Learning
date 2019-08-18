# CS294-112 HW 1: Imitation Learning

Dependencies:
 * Python **3.5**
 * Numpy version **1.14.5**
 * TensorFlow version **1.10.5**
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**

Once Python **3.5** is installed, you can install the remaining dependencies using `pip install -r requirements.txt`.

**Note**: MuJoCo versions until 1.5 do not support NVMe disks therefore won't be compatible with recent Mac machines.
There is a request for OpenAI to support it that can be followed [here](https://github.com/openai/gym/issues/638).

**Note**: Students enrolled in the course will receive an email with their MuJoCo activation key. Please do **not** share this key.

The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* Ant-v2.pkl
* HalfCheetah-v2.pkl
* Hopper-v2.pkl
* Humanoid-v2.pkl
* Reacher-v2.pkl
* Walker2d-v2.pkl

The name of the pickle file corresponds to the name of the gym environment.


#### ----UPDATE ----
###### Behavior Cloning Results
For the Imitiation Learning, i have used a 4 layer feedforward network with 100 units in each hidden layer and ReLu as activation function.

Figure shown below, compares the result of policy obtained through behavior cloning and expert policy on all the 6 tasks for 20 episodes.
![Behavior cloning Result](https://github.com/nilesh0109/RL-assignments/blob/master/RL-Deep/hw1/Results/Behaviour_cloning/Behavior_cloning_20000_epochs.png)
 BC is able to imitate 2 out of 6 environments very well.
 The BC policy is trained with 20 rollouts of expert policy and is trained to 20,000 epochs for all tasks
 
###### DAgger Results
Figure shown below, comapares the result of policy obtained through DAgger, behavior cloning and expert policy on all the 6 tasks for 20 episodes.
![Behavior cloning Result](https://github.com/nilesh0109/RL-assignments/blob/master/RL-Deep/hw1/Results/Imitation_Learning/plot.png)
 Clearly, DAgger is outperforming BC on all the 6 tasks and is even reaching close to expert policy asymptotically in all except one task(Humanoid-v2).
 The DAgger policy is trained initially for 2000 epochs with 20 rollouts of expert policy, then the training data is enhanced succesively for 9 more loops(i.e. 10 in total) with 20 rollout steps of DAgger policy.
