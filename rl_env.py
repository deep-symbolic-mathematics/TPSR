import logging
import os
from collections import OrderedDict
from types import SimpleNamespace

from reward import compute_reward
from model import refine_for_sample

class RLEnv:
    """
    Equation Generation RL environment.

    State: a list of tokens.
    Action: a token (an integer).
    Reward: Fittness reward of the generated equation.
    """
    def __init__(self,params, samples, equation_env, model):
        self.params = params
        self.samples = samples
        self.equation_env = equation_env
        self.model = model

        self.state = [self.equation_env.equation_word2id['<EOS>']]
        self.terminal_token = self.equation_env.equation_word2id['<EOS>']

        # state -> reward
        # we may need to retrieve the states (programs) in the order they were saved, so use OrderedDict
        self.cached_reward = OrderedDict()

    def transition(self, s, a, is_model_dynamic=True):
        if a == self.terminal_token:
            done = True
        else:
            done = False

        next_state = s + [a]

        if done:
            reward = self.get_reward(next_state)
        else:
            reward = 0 # no intermediate reward
        
        return next_state, reward, done

    def step(self, action):
        self.state, reward, done = self.transition(self.state, action)

        return self.state, reward, done, {}

    def get_reward(self, s,mode='train'):
        """
        Returns:
            The reward of program in s.
        """
        if s is None:
            return 0

        if tuple(s) in self.cached_reward.keys() and mode == 'train':
            # cache rewards for training
            return self.cached_reward[tuple(s)]

        if (type(s) != list):
            s = s.tolist()
        y_pred, model_str, generations_tree = refine_for_sample(self.params, self.model,self.equation_env, s, x_to_fit = self.samples['x_to_fit'],y_to_fit = self.samples['y_to_fit']) 

        reward = compute_reward(self.params,self.samples, y_pred, model_str, generations_tree)

        if mode == 'train':
            self.cached_reward[tuple(s)] = reward

        return reward

    def equality_operator(self, s1, s2):
        return s1 == s2
    
    def tokenizer_decode(self, node_action):
        return self.equation_env.equation_id2word[node_action]

    def convert_state_to_program(self, state):
        prog = []
        if type(state) != list:
            state = state.tolist()
        for i in range(len(state)):
            prog.append(self.equation_env.equation_id2word[state[i]])
        # return prog
        return " ".join(prog)