import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box

def mlp(
    sizes,
    activation=nn.Tanh,
    output_activation=nn.Identity
):  #? 为何都喜欢用Tanh，Relu不香么？ search for differences [reddit似乎有人回答过了，知乎link里面有，但还没来得及细看]
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def train(env_name='CartPole-v0',
          hidden_sizes=[32],
          lr=1e-2,
          epochs=50,
          batch_size=5000,
          render=False):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[
        0]  # TODO what's more about attributes of observation_space
    n_acts = env.action_space.n

    # make core of policy network
    #* hesy: one layer here
    logits_net = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(
            obs).sample().item()  #hesy: item get data from Tensor

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        # TODO 这里还需要好好品一下 --> 1. get_policy利用的是Categorical，which 输出是离散的把  2.这里的log_prob到底是什么？根据离散输出的值，返回去查其probability?
        logp = get_policy(obs).log_prob(act)
        #* hesy: class Categorical's special function --> get log probability of act
        return -(logp * weights).mean()  #* mean() 是因为这里是一个batch

    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)  #TODO: 似乎adam并不是特别受欢迎？
        return get_policy(obs).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)
>>>>>>> temp

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
<<<<<<< HEAD
        batch_obs = []  # for observations
        batch_acts = []  # for actions
        batch_weights = []  # for R(tau) weighting in policy gradient
        batch_rets = []  # for measuring episode returns
        batch_lens = []  # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()  # first obs comes from starting distribution
        done = False  # signal from environment that episode is over
        ep_rews = []  # list for rewards accrued throughout ep
=======
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep
>>>>>>> temp

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:
<<<<<<< HEAD
            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()
            # save obs
            batch_obs.append(
                obs.copy()
            )  #TODO copy to avoid computaion graph becomes larger ( check and record )  # 如何查看计算图中有哪些？如何看计算图?
            # act in the environment
            act = get_action(torch.as_tensor(
                obs, dtype=torch.float32))  # hesy:obs was ndarray before
=======

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
>>>>>>> temp
            obs, rew, done, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
<<<<<<< HEAD
                batch_weights += [ep_ret] * ep_len  # 果然是最naive的版本..
=======
                batch_weights += [ep_ret] * ep_len
>>>>>>> temp

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
<<<<<<< HEAD
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs,
                                                      dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts,
                                                      dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights,
                                                          dtype=torch.float32))
=======
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
>>>>>>> temp
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
<<<<<<< HEAD
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
              (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

=======
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
>>>>>>> temp

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)
