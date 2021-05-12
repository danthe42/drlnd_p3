

[TOC]



# Project 3: Collaboration and Competition - Report

This project demonstrates training multiple agents using Multi agent reinforcement learning in the Tennis environment introduced in the [readme](README.md) file.  

I have chosen the MADDPG (multi-agent deep deterministic policy gradient) algorithm which was introduced in the 1st referenced document. 



## Architecture

I have create a hybrid solution in python using the numpy and pytorch frameworks for the agent, and it is using the UnityEnvironment module from the unityagents library for establishing the bridge between the two playing agents and the simulator (written in Unity). 

The agents can be trained using a Jupyter notebook. This notebook presents a nice, interactive UI, and it also uses a few python files behind the scenes. These code elements are:

- model.py: The neural networks implementing the policy models (Actor) and the value approximator models (Critic). Each agent have 2 Actor and 2 Critics because the MADDPG algorithm uses a target network and a "current" network.  

- p3_agent.py: Here is implemented the MADDPG agent: The "MADDPG" class implements the algorithm. Only its constructor, "act", and "train_one_episode" methods are called from outside.

- buffer.py: A simple implementation of a replay buffer which is able to store arbitrary data tuples in a size-limited storage, and return randomized samples from it on request. 

- OUNoise.py: Ornstein-Uhlenbeck Noise implementation (to be used in exploration) 

- utilities.py: A few helper procedures for making updates on a model based on the other, or making tensor from list, or transpose one. 

- Tennis.ipynb: A Jupyter notebook for demonstrating 

  - the train process of the agent, 
  - the visualized results (scores),
  - after training, the model is loaded and an episode is played as a demonstration.   

  This file is also exported here, so you can read it without an installed Jupyter notebook instance: [exported notebook](export/Continuous_Control.md)



Please follow the notebook to see the training of the MADDPG agent. 

For the detailed description of the algorithm used in the agent and the models, you can use the comments in the relevant python codes, and you can also use the following text.    



## The implemented MADDPG Agents

### Model design

An agent is using 2 different type of neural networks: 

- The Actor model is the policy network which decides the actions of the agent. It requires only the observation of the agent, which consists of 24 values. It is using 3 Fully Connected layers with RELU activations between them, and a tanh layer at the end. The layer sizes are [256,128,2]. The hyperbolic tangent function normalizes the 2 output values in the range (-1,1).  After that, because of exploration during training, we are modifying these outputs with a noise based on the Ornstein-Uhlenbeck process, as it is described in chapter 3 on the 3rd referenced document: "As detailed in the supplementary materials we used an Ornstein-Uhlenbeck process (Uhlenbeck & Ornstein, 1930) to generate temporally correlated exploration for exploration efficiency in physical control problems with inertia (similar use of autocorrelated noise was introduced in (Wawrzynski, 2015))".
- The second type of model used is for the Critic network. It's the centralized action-value function for the given agent. As explained in the MADDPG algorithm description (1st referenced document), this model needs the whole state (here: the union of the observations of each agent), and all actions as input, and only outputs one number: the Value itself. In this case it means 52 input values, and 1 output. This model also using 3 Fully Connected layers and 2 RELU activation functions between them. The layer sizes are [256,64,1]. The final output is not modified, it is the real, estimated Value of the input state + the actions.   

The MADDPG algorithm is using two instances of these two networks per agent to implement the training, and only need one actor per agent to play episodes. 

### Training process

#### Main loop

The training is happening by playing full episodes in a loop, and train the agents in each timestep when we already have enough sample data in the replay buffer for a minibatch.

As explained in the [README](README.md) , the environment is considered solved when the average score over 100 episodes reaches +0.5 .

I've noticed the training on GPU with CUDA is much faster than doing it on CPU, so I've enabled it by default. The probable reason of this is the big minibatch size, which can be parallelized very well. 

#### Agent initialization

The agents are initialized with the same, tuned hyperparameters. These are:

LR_actor = 0.00004 					   # Learning Rate used by the optimizer to train the Actor.

LR_critic = 0.0004 						 # Learning Rate used by the optimizer to train the Critic.

tau = 0.12									   # Factor used to update target network with the current one

discount_factor = 0.9999			   # discount factor used to discount future rewards. 

gradient_clip = 1.0					     # Maximum norm of the gradient (threshold) used to avoid the potential exploding gradient problem. 

replay_buffer_size = 50000 		 # the size of the replay buffer which stores the recorded transitions

noise_scale = 1.0					     # The initial (maximum) amplitude of the OU noise.

noise_reduction = 0.9999		    # The multiplication factor used to decrease the noise amplitude once per each timestep

minibatch_size = 512				  # Batch size used to take samples from the replay buffer 

weight_decay = 0.00001			# Weight Decay regularization: Here, the implementation on the L2 penalty is is modified as it was proposed in the changes in the 2nd referenced document, where a better Weight Decay regularization is described to be used with Adam optimizers.  



#### One training step

As I wrote earlier, after enough transitions are collected and inserted into the replay buffer, the training step will come. The core of the algorithm is this one training step. It consists of the following basic steps:

1. For each agent: 
   - Get a sample minibatch from the replay buffer. This is a completely random sampling. 
   - Train the given agent's current Actor and Critic networks by calling the update() function in the MADDPG agent. This step will use the previously selected transitions in the minibatch to optimize the Critic and the Agent networks of the given agent, through the following steps::
     -   We have (observations, actions, reward, next_observations)  tuples for each transitions. First, we calculate the next_actions (after the next_observations) using the Target Actor network. ( "a prime" in the first, referenced document's MADDPG algorithm description at the very beginning of the Appendix)
     -    We calculate the preferred value of the initial state/all observations ( "y" in the algorithm ) by using the Target Critic network at the next timestep ("q_next") plus the reward we got after the initial state/action ("reward") .     
     -   Then, we define the Critic loss value as the Smooth L1 Loss (see pytorch documentation's SmoothL1Loss for details) between the estimated value of the Current Critic network with the initial state/actions input and the previously calculated  "y" value. We do one backprop. step to minimize this value on the Current Critic network. Also, we are maximizing the norm of gradient with a value preconfigured as hyperparameter, using the pytorch clip_grad_norm_ function.
     -   After this, we also update the Current Agent network and maximize the value obtained by executing the Current Critic network from the initial "state/observations" and doing actions defined by this Current Agent network.        
2. Update the target networks of each agent. We are using tau (hyperparameter) to partly update the target networks of the agent with its current (previously optimized) networks. 



### Results, Conclusion

During hyperparameter tuning my experience was that there are many configurations when the agent scores just stop getting higher, it stays consistently around 0.04, or 0.15 and did not improve anymore. I could not find the reason why this phenomenon occurs.       

As it can be seen in the notebook, the environment was solved in 1385 episodes, which means that the average of the better agent scores over 100 episodes reach +0.5.  

In the final part of the jupyter notebook, after reaching score 0.5, I've demonstrated: 

- how to save the final model. The Critic network is not necessary for playback, only if we want to continue training, but I saved it, anyway,
- then I've shown how to load the networks after reinitialization,
- and how to play a couple (10) new episodes, without training.

The saved, final trained models are included in the github repo in the following files: agent0_critic_net.pth, agent0_actor_net.pth, agent1_critic_net.pth, and agent1_actor_net.pth.



## Ideas for future work

After this implementation, I think that the following ideas and promising directions would be worthwhile to work on: 

- I was thinking to try out some other Noise functions, not just the OUNoise.
- Hyperparameter tuning: There's never enough time to try out all combinations. Should execute more trainings with different values.
- Should target a harder goal: If I consider the environment solved just after score +0.5, there will still be complete episodes with very low scores ( like 0.0 or 0.1 ) It means that there is still space for improvement.   
- Would make sense to try out other optimizers, not just Adam
- It would be a good idea to try out this same MADDPG algorithm with the other, Crawler challenge. 
- I would be cool to transform this tennis challenge to 3 dimension: currently, the rackets can only move in 2, toward/away from the net and up/down (jump). I would *love* to see a Unity tennis simulator with an environment where the rackets can also move sideways. 
- There is an interesting proposal in the MADDPG document: To avoid the undesirable overfitting, where the agent learns its competitor's policies/behavior and not able to handle other agents and/or changing policies we could use many different, randomly selected sub-policies for each agent. (So-called Policy Ensembles) Should try it.
- Somehow, somewhere I've lost deterministic behavior, probably after switching to use CUDA. I've tried to initialize random generators with a seed value, even checked torch internal function calls using the "torch.use_deterministic_algorithms(True)" switch, but it is still not deterministic. Need to investigate.       



## REFERENCES

[Ryan Lowe](https://arxiv.org/search/cs?searchtype=author&query=Lowe%2C+R), [Yi Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+Y), [Aviv Tamar](https://arxiv.org/search/cs?searchtype=author&query=Tamar%2C+A), [Jean Harb](https://arxiv.org/search/cs?searchtype=author&query=Harb%2C+J), [Pieter Abbeel](https://arxiv.org/search/cs?searchtype=author&query=Abbeel%2C+P), [Igor Mordatch](https://arxiv.org/search/cs?searchtype=author&query=Mordatch%2C+I) - Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments - https://arxiv.org/abs/1706.02275

[Ilya Loshchilov](https://arxiv.org/search/cs?searchtype=author&query=Loshchilov%2C+I), [Frank Hutter](https://arxiv.org/search/cs?searchtype=author&query=Hutter%2C+F) - Decoupled Weight Decay Regularization - https://arxiv.org/abs/1711.05101

[Timothy P. Lillicrap](https://arxiv.org/search/cs?searchtype=author&query=Lillicrap%2C+T+P), [Jonathan J. Hunt](https://arxiv.org/search/cs?searchtype=author&query=Hunt%2C+J+J), [Alexander Pritzel](https://arxiv.org/search/cs?searchtype=author&query=Pritzel%2C+A), [Nicolas Heess](https://arxiv.org/search/cs?searchtype=author&query=Heess%2C+N), [Tom Erez](https://arxiv.org/search/cs?searchtype=author&query=Erez%2C+T), [Yuval Tassa](https://arxiv.org/search/cs?searchtype=author&query=Tassa%2C+Y), [David Silver](https://arxiv.org/search/cs?searchtype=author&query=Silver%2C+D), [Daan Wierstra](https://arxiv.org/search/cs?searchtype=author&query=Wierstra%2C+D) - Continuous control with deep reinforcement learning - https://arxiv.org/pdf/1509.02971.pdf











