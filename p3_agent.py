# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
import torch.nn.functional as F
from utilities import soft_update, transpose_to_tensor, transpose_list
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

class MADDPG:
    def __init__(self, obs_size, action_size, discount_factor=0.95, tau=0.02, gradient_clip=1.0, lr_actor=1.0e-4, lr_critic=1.0e-3):
        super().__init__()

        # critic input consists of the global state ( 2 local observations ) and the actions of all agents. 
        self.maddpg_agent = [DDPGAgent(obs_size, 256, 128, action_size, 2*obs_size+2*action_size, 256, 64, device, lr_actor, lr_critic), 
                             DDPGAgent(obs_size, 256, 128, action_size, 2*obs_size+2*action_size, 256, 64, device, lr_actor, lr_critic)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0
        self.obs_size = obs_size
        self.action_size = action_size
        self.gradient_clip = gradient_clip

    def init_episode(self):
        for a in self.maddpg_agent:
            a.noise.reset()

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    # Get actions of all available agents based on their local observation from the Current Actor, with a random OUNoise is preferred.
    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    # Get actions of all available agents based on their local observation from the Target Actor, with a random OUNoise is preferred.
    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions

    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """

        # need to transpose each element of the samples
        # to flip forexample obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        (obs, action, reward, next_obs, done) = map( transpose_to_tensor, transpose_list(samples) )

        # get minibatch size (it will be used in some assertion testing later)
        n_samples = len(reward[agent_number])

        # For the ability of execution on CUDA, it's necessary to upload these tensors to the GPU 
        obs[0] = obs[0].to(device)
        obs[1] = obs[1].to(device)
        obs_all = torch.cat( obs, dim=1 )
        next_obs_all = torch.cat( next_obs, dim=1 ).to(device)
        reward[agent_number] = reward[agent_number].to(device)
        done[agent_number] = done[agent_number].to(device)
        action = torch.cat(action, dim=1).to(device)

        agent = self.maddpg_agent[agent_number]

        """
          1st step: update critic network using critic_loss
            This critic loss will be the batch mean of (y- Q(s,a) from target network)^2
            Only the selected agent's critic is optimized.  
        """
        agent.critic_optimizer.zero_grad()

        # combine all the actions and observations for input to critic
        target_actions = self.target_act( next_obs )
        target_actions = torch.cat(target_actions, dim=1)
        target_critic_input = torch.cat( (next_obs_all,target_actions), dim=1).to(device)
        
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)

        # y = reward of this timestep + discount * Q(st+1,at+1) from target network
        y = reward[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1))

        critic_input = torch.cat((obs_all, action), dim=1).to(device)

        assert(critic_input.shape == torch.Size([n_samples, 2*self.obs_size + 2*self.action_size]))
        q = agent.critic(critic_input)
        assert(q.shape == torch.Size([n_samples, 1]))

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        #critic_loss = F.mse_loss( q, y.detach() )
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), self.gradient_clip)
        agent.critic_optimizer.step()

        """
          2nd step: update actor network using policy gradient
            This actor's loss will be the mean of the values (based on critic) using all observations and the actors' chosen actions for all transitions in the minibatch
            Only the selected agent's actor is optimized.  
        """

        agent.actor_optimizer.zero_grad()

        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(obs) ]


        # combine all the actions and observations for input to critic
        q_input = torch.cat(q_input, dim=1)
        q_input2 = torch.cat((obs_all, q_input), dim=1)       
        assert(q_input2.shape == torch.Size([n_samples, 2*self.obs_size + 2*self.action_size]))

        # get the policy gradient
        actor_loss = -agent.critic(q_input2).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),self.gradient_clip)
        agent.actor_optimizer.step()

    # perform a soft update on all agents from their Current networks to their Target networks 
    def update_targets(self):
        """soft update targets using self.tau as factor"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
            
            
            




