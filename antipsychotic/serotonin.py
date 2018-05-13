from __future__ import print_function

import gym
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class ModifiedTaxiEnv:
    """Modifications to the taxi environment:

    * make it so that first episode lasts forever (essentially)
    * add a NOOP so that laziness is meaningfully possible
    * amplify positive rewards (multiply by 10), since the base taxi env is so
      unforgiving

    """
    def __init__(self):
        self.env = gym.make('Taxi-v2')
        self.env._max_episode_steps = 1000 * 1000 * 1000
        self.nactions = 1 + self.env.action_space.n
        self.nstates = self.env.observation_space.n
        self.last_obs = None

    def render(self):
        return self.env.render()

    def reset(self):
        obs = self.env.reset()
        self.last_obs = obs
        return obs

    def step(self, action):
        if action == 0:
            return self.last_obs, -1, False, {'noop': True}
        obs, reward, done, info = self.env.step(action - 1)

        # amplify positive rewards, since the base taxi env is so unforgiving
        if reward > 0:
            reward *= 10

        self.last_obs = obs
        return obs, reward, done, info

def main():
    env = ModifiedTaxiEnv()

    # discount factor for Q- and F-values
    gamma = 0.95
    # discount factor for S-value
    sigma = 0.995
    # epsilon-greedy exploration parameter
    epsilon = 0.05
    # learning rate
    alpha = 0.1
    # initial S-value
    svalue = 0
    # break ties randomly by adding small amount of random noise to Q-values
    tiebreak = 0.00001
    # use an extremely pessimistic initialization to make it cautious
    optimism = -7

    # max reward desired is 10 * 10 = 100
    max_reward_desired = 100.0
    # mean reward desired is 5 utils / step
    mean_reward_desired = 5.0
    # how long are we willing to wait before achieving the next reward?
    patience = 10
    # maximum desired S-value
    max_s_desired = mean_reward_desired / (1 - sigma)
    # how low must a reward be before we consider it a "mistake"
    min_reward_desired = -5

    # cumulative reward experienced
    tot_reward = 0.0
    # tabular estimator of Q-values
    qtable = optimism + np.zeros((env.nstates, env.nactions))
    # tabular estimator of F-values
    fear_table = optimism + np.zeros((env.nstates, env.nactions))
    # keep track of how often various actions are taken
    action_histogram = np.zeros(env.nactions)

    # experience replay cache
    experience_replay = []
    # capacity of experience replay cache
    capacity = 100 * 100

    # TRAINING LOOP 
    obs = env.reset()
    # rewards experienced over time
    rewards = []
    # "mistakes" made over time
    mistakes = []
    # S-values experienced over time
    svalues = []
    for step in range(1, 40 * 1000):
        # calculate desired S_sat-value
        Ssat = max(
            # the mature case
            mean_reward_desired / (1 - sigma) + 
            gamma ** patience * max(max_s_desired, qtable.max()),
            # the immature case - bump up the desired satisfaction
            20 * 1000 - step)

        qmax = Ssat - svalue

        # if any q-value hits the qmax threshold, turn off exploration forever
        # and print out some diagnostic info
        if (qtable[obs, :] >= qmax).any():
            print('#' * 100)
            print((qtable[obs, :] >= qmax).mean())
            print('#' * 100)            
            epsilon = 0

        # PICK AN ACTION
        action = (np.random.random(env.nactions) * tiebreak + # random tiebreak for degenerate cases
                  fear_table[obs, :] + # fear negative rewards
                  (qtable[obs, :] >= qmax) + # prefer actions that hit the threshold
                  np.minimum(qtable[obs, :], qmax) # maximize q-value of selected action (up to qmax)
        ).argmax()
        # epsilon-greedy exploration
        if np.random.random() < epsilon:
            action = np.random.randint(env.nactions)
        print('ACTION', action)

        # some book-keeping and diagnostic prints
        old_obs = obs
        obs, reward, done, info = env.step(action)
        action_histogram[action] += 1
        print(action_histogram)
        env.render()

        # if done, reset env
        if done:
            obs = env.reset()

        # remember rewards and S-values
        rewards.append(reward)
        tot_reward += reward        
        svalue = sigma * svalue + reward
        svalues.append(svalue)

        # diagnostic prints
        print('STEP', step)
        print('REWARD', reward)
        print('SSAT', Ssat)
        print('SVAL', svalue)
        print('QMAX', qmax)
        print('SUM RW', np.sum(rewards[-100:]))
        print('MEAN RW', np.mean(rewards[-100:]))
        print('RECENT MISTAKES', np.sum(mistakes[-100:]))
        print('QTAB', 
              (qtable > optimism).mean(),
        )
        print('FTAB', 
              (fear_table < optimism).mean(),
        )
        if reward > mean_reward_desired:
            print('*' * 100)

        # did we make a "mistake"?
        if reward < min_reward_desired:
            print('?' * 100)
            mistakes.append(True)
        else:
            mistakes.append(False)

        # update Q-table and F-table
        experience_replay.append((old_obs, action, reward, obs))
        experience_replay = experience_replay[-capacity:]
        for s, a, r, s2 in experience_replay:
            qtable[s, a] = (1 - alpha) * qtable[s, a] + alpha * (r + gamma * qtable[s2, :].max())
            if r < min_reward_desired:
                fear_table[s, a] = ((1 - alpha) * fear_table[s, a] + 
                                    alpha * (r + gamma * fear_table[s2, :].max()))


    # produce the plot of results
    plt.plot(range(len(svalues)), np.array(svalues) / np.array(svalues).max(), 'c')
    plt.plot(range(len(rewards)), np.cumsum(rewards) / np.cumsum(rewards)[-1], 'r')
    plt.plot(range(len(mistakes)), np.cumsum(mistakes) / np.cumsum(mistakes)[-1], 'm')

    # save the plot
    plt.savefig('borgies.png')


if __name__ == '__main__':
    main()
