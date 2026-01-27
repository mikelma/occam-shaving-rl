
# Hyperparameters
- 37 Tricks Blog [**https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/**](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

## Brax
- There are different default params for the Brax paper compared to the PureJaxRL code which is where we got our implementation from.
- We included the values from both of these baselines in our sweep.
#### Current Values for Sweep
- environments: ["walker2d", "ant", "humanoid"]
	- Minatar?
- number of parallel runs: 30 
	- number of seeds
- vf coefficient: 0.5
	- Weighting for the value function loss
- total training timesteps: 3 million
- hidden dimension size: 256
- Number of environments: 2048
- network activation function: tanh
- Number of Steps: [5, 10]
	- Number of steps in environment between updates
- Gamma: [0.99, 0.95]
	- Discount factor
- Learning Rate: [3e-3, 3e-4, 3e-5]
	- Minatar uses 5e-3
- Entropy Coefficient: [0.0, 0.01, 0.001]
	- Weighting for the entropy term in the loss
	- Previous works mentioned in the 37 tricks blog seemed to indicate that there was no performance improvement for having entropy in continuous control environments, but I also talked with Jacob and he said his thesis work contradicted this clain.
- Update Epochs: [1, 4]
	- How many times to update with each batch of data
- Number of Minibatches: [16, 32, 64]
	- Number of minibatches each batch of data is split into
- GAE Lambda Coefficient: [0.9, 0.95]
	- Tradeoff between Monte Carlo and One Step returns
- Value for Epsilon Clip: [0.2, 1e6] (on/off)
	- PPO policy clip value
- Value Clip Epsilon: [0.2, 1e6] (on/off)
	- Clips value function estimates if too large
- Max Gradient Norm: [0.5, 1e6] (on/off)
- Anneal Learning Rate: [True, False]
- Normalize Environment: [True, False]
	- Normalizes both observations and rewards, but this could also be separated out into two different configurations
- GAE Normalization: [True, False]
	- gae = (gae - gae.mean()) / (gae.std() + 1e-8)
- Layer Norm: [True, False]
	- Add in layer norm or not
