#!/usr/bin/env python
from __future__ import print_function
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1

from __future__ import print_function

import argparse
import itertools
import random
import threading
import tensorflow as tf
from time import ctime, sleep
#from agents.hdqn import DeepQNetwork
#from utils.reward_utils import *

try:
    import hfo
except ImportError:
    print('Failed to import hfo. To install hfo, in the HFO directory' \
          ' run: \"pip install .\"')
    exit()


def player(mark):
    print('--I am player', mark, ctime())
    # Create the HFO Environment
    hfo_env = hfo.HFOEnvironment()
    hfo_env.connectToServer(hfo.LOW_LEVEL_FEATURE_SET,
                            'C:/Users/Administrator/HFO/bin/teams/base/config/formations-dt',
                            args.port, 'localhost', 'base_right', False)

    total_step = 0
    ep_rewards = []
    ep_steps = []
    ep_goals = []
    for episode in itertools.count():
        status = hfo.IN_GAME
        episode_step = 0

        while status == hfo.IN_GAME:
            total_step += 1
            episode_step += 1

            # Get the vector of state features for the current state
            state = hfo_env.getState()

            # Perform the action
            action_index = np.random.random()

            # TODO 0 for MOVE, 1 for CATCH, 2 for NOOP
            if action_index < 0.166:
                hfo_env.act(hfo.MOVE)
            elif action_index < 0.333:
                hfo_env.act(hfo.INTERCEPT)
            elif action_index < 0.5:
                hfo_env.act(hfo.GO_TO_BALL)
            elif action_index < 0.667:
                hfo_env.act(hfo.REDUCE_ANGLE_TO_GOAL)
            elif action_index == 0.833:
                hfo_env.act(hfo.REORIENT)
            else:
                hfo_env.act(hfo.NOOP)

            # Advance the environment and get the game status
            status = hfo_env.step()
            state_ = hfo_env.getState()
            reward = cal_reward(state, state_, status)

        # FIXME recording
        ep_steps.append(episode_step)
        ep_rewards.append(reward)
        if status == hfo.GOAL:
            ep_goals.append(1)
        else:
            ep_goals.append(0)
        ep_steps = ep_steps[-100:]
        ep_rewards = ep_rewards[-100:]
        ep_goals = ep_goals[-100:]

        if (episode + 1) % print_interval == 0 and mark == 0:
            print("================================================")
            print("--Agent:", mark)
            print("--Episode: ", episode)
            print("----Avg_steps: ", sum(ep_steps[-100:]) / 100.0)
            print("----Avg_reward: ", sum(ep_rewards[-100:]) / 100.0)
            print("----Goal_rate: ", sum(ep_goals[-100:]) / 100.0)
            print("------------------------------------------------")

        # Check the outcome of the episode

        # end_status = hfo_env.statusToString(status)
        # print("Episode {0:n} ended with {1:s}".format(episode, end_status))

        # Quit if the server goes down
        if status == hfo.SERVER_DOWN:
            hfo_env.act(hfo.QUIT)
            exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=6000,
                        help="Server port")
    parser.add_argument('--agent-num', type=int, default=2)
    parser.add_argument('--train-interval', type=int, default=10)
    parser.add_argument('--bef-train', type=int, default=20000)
    parser.add_argument('--print-interval', type=int, default=100)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--is-save-log', type=bool, default=False)
    args = parser.parse_args()

    # set random seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    IS_SAVE_LOG = args.is_save_log

    agent_num = args.agent_num
    train_interval = args.train_interval
    bef_train = args.bef_train
    print_interval = args.print_interval

    lock = threading.Lock()
    threads = []
    for i in range(agent_num):
        threads.append(threading.Thread(target=player, args=(i,)))
    for t in threads:
        t.setDaemon(True)
        t.start()
        sleep(5)

    [t.join() for t in threads]
    print('--Game Over.', ctime())

