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

DEFAULT_GAMMA = 0.95
DEFAULT_SIGMA = 0.995
DEFAULT_INITIAL_EPSILON = 0.05
DEFAULT_ALPHA = 0.1
DEFAULT_INITIAL_SVALUE = 0
DEFAULT_TIEBREAK = 0.00001
# use an extremely pessimistic initialization to make it cautious
DEFAULT_OPTIMISM = -7
DEFAULT_MAX_REWARD_DESIRED = 100.0 # = 10 * 10 = 100
DEFAULT_MEAN_REWARD_DESIRED = 5.0
DEFAULT_PATIENCE = 10
def main(fear=True,
         # discount factor for Q- and F-values
         gamma = DEFAULT_GAMMA,
         # discount factor for S-value
         sigma = DEFAULT_SIGMA,
         # epsilon-greedy exploration parameter
         initial_epsilon = DEFAULT_INITIAL_EPSILON,
         # learning rate
         alpha = DEFAULT_ALPHA,
         # initial S-value
         initial_svalue = DEFAULT_INITIAL_SVALUE,
         # break ties randomly by adding small amount of random noise to Q-values
         tiebreak = DEFAULT_TIEBREAK,
         # optimistic or pessimistic initialization
         optimism = DEFAULT_OPTIMISM,

         # max reward desired 
         max_reward_desired = DEFAULT_MAX_REWARD_DESIRED,
         # mean reward desired in utils / step
         mean_reward_desired = DEFAULT_MEAN_REWARD_DESIRED,
         # how long are we willing to wait before achieving the next reward?
         patience = DEFAULT_PATIENCE):

    epsilon = initial_epsilon
    svalue = initial_svalue

    env = ModifiedTaxiEnv()

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

        # if any q-value hits the qmax threshold, print out some diagnostic
        # info
        if (qtable[obs, :] >= qmax).any():
            print('#' * 100)
            print((qtable[obs, :] >= qmax).mean())
            print('#' * 100)            

        # at step 20K, turn off exploration forever
        if step >= 20 * 1000:
            epsilon = 0

        # PICK AN ACTION
        if fear:
            action = (np.random.random(env.nactions) * tiebreak + # random tiebreak for degenerate cases
                      fear_table[obs, :] + # fear negative rewards
                      (qtable[obs, :] >= qmax) + # prefer actions that hit the threshold
                      np.minimum(qtable[obs, :], qmax) # maximize q-value of selected action (up to qmax)
                  ).argmax()
        else:
            action = (np.random.random(env.nactions) * tiebreak + # random tiebreak for degenerate cases
                      # don't fear negative rewards
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
    plt.figure()
    plt.plot(range(len(svalues)), np.array(svalues) / np.array(svalues).max(), 'c')
    plt.plot(range(len(rewards)), np.cumsum(rewards) / np.cumsum(rewards)[-1], 'r')
    plt.plot(range(len(mistakes)), np.cumsum(mistakes) / np.cumsum(mistakes)[-1], 'm')

    # CONSTRUCT THE OUTPUT IMAGE NAME
    # do we have fear or not?
    fname_base = 'borgies'
    if not fear:
        fname_base += '_nofear'

    # note down all non-default params
    precision = 10000 # 4 decimal places
    if gamma != DEFAULT_GAMMA:
        fname_base += '_gamma%04d' % int(precision * gamma)
    if sigma != DEFAULT_SIGMA:
        fname_base += '_sigma%04d' % int(precision * sigma)
    if initial_epsilon != DEFAULT_INITIAL_EPSILON:
        fname_base += '_epsilon%04d' % int(precision * initial_epsilon)
    if alpha != DEFAULT_ALPHA:
        fname_base += '_alpha%04d' % int(precision * alpha)
    if initial_svalue != DEFAULT_INITIAL_SVALUE:
        fname_base += '_inits%04d' % int(precision * initial_svalue)
    if tiebreak != DEFAULT_TIEBREAK:
        fname_base += '_tiebreak%04d' % int(precision * tiebreak)
    if optimism != DEFAULT_OPTIMISM:
        fname_base += '_optimism%04d' % int(precision * optimism)
    if max_reward_desired != DEFAULT_MAX_REWARD_DESIRED:
        fname_base += '_ambition%04d' % int(precision * max_reward_desired)
    if mean_reward_desired != DEFAULT_MEAN_REWARD_DESIRED:
        fname_base += '_greed%04d' % int(precision * mean_reward_desired)
    if patience != DEFAULT_PATIENCE:
        fname_base += '_patience%04d' % int(precision * patience)

    fname = fname_base + '.png'

    # save the plot
    plt.savefig(fname)


if __name__ == '__main__':
    main()
    main(fear=False)
    main(gamma=0.9999)
    main(sigma=0.9999)
    main(initial_epsilon=0)
    main(alpha=0.0001)
    main(initial_svalue=9.9999)
    main(tiebreak=0)
    main(optimism=100)
    main(max_reward_desired=9999)
    main(mean_reward_desired=1000)
    main(patience=1)
