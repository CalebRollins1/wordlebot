from tensorforce.environments import Environment
from tensorforce.agents import Agent
import numpy.random as rd
import string
from tensorforce.execution import Runner
import numpy as np


guesses = []
solutions = []

for elem in list(open('wordlist_guesses.txt')):
    guesses.append(elem.strip())

for elem in list(open('wordlist_solutions.txt')):
    solutions.append(elem.strip())


class Wordle(Environment):

    def __init__(self):
        super().__init__()
        self.state = np.zeros([26*7])
        self.key = [0,0,0,0,0]
        self.greens = []
        self.yellows = set()
        self.grays = set()
        self.letter_dict = {elem:i for i,elem in enumerate(string.ascii_lowercase)}

    def states(self):
        return dict(type = 'int',shape = (26*7,),num_values = 26*7)

    def actions(self):
        return dict(type = 'int',num_values = len(guesses))

    # Optional: should only be defined if environment has a natural fixed
    # maximum episode length; otherwise specify maximum number of training
    # timesteps via Environment.create(..., max_episode_timesteps=???)
    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    # Optional additional steps to close environment
    def close(self):
        super().close()

    def reset(self):
        answer = rd.choice(solutions)
        self.state = np.zeros([26*7])
        self.greens = []
        self.yellows = set()
        self.grays = set()
        return state

    def execute(self, actions):

        print(actions)
        quit()

        g = self.greens.copy()
        y = self.yellows.copy()
        b = self.grays.copy()

        self.key = [0,0,0,0,0]
        for i in range(5):
            if actions[i]==self.answer[i]:
                self.key[i]+=2
                g.append((i,actions[i]))
                if actions[i] in y:
                    y.remove(actions[i])
            elif actions[i] in self.answer:
                self.key[i]+=1
                y.add(guess[i])
            else:
                b.add(guess[i])

        next_state = self.state.copy()
        for i,elem in self.greens:
            next_state[26*i+self.letter_dict[elem]] = 1
        for elem in y:
            next_state[26*5+self.letter_dict[elem]] = 1
        for elem in b:
            next_state[26*6+self.letter_dict[elem]] = 1

        terminal = False  # Always False if no "natural" terminal state

        if self.key == [2,2,2,2,2]:
            reward = 1
            terminal = True
        else:
            reward = 0

        return next_state, terminal, reward

class CustomEnvironment(Environment):

    def __init__(self):
        super().__init__()

    def states(self):
        return dict(type='float', shape=(8,))

    def actions(self):
        return dict(type='int', num_values=4)

    # Optional: should only be defined if environment has a natural fixed
    # maximum episode length; otherwise specify maximum number of training
    # timesteps via Environment.create(..., max_episode_timesteps=???)
    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    # Optional additional steps to close environment
    def close(self):
        super().close()

    def reset(self):
        state = np.random.random(size=(8,))
        return state

    def execute(self, actions):
        next_state = np.random.random(size=(8,))
        terminal = False  # Always False if no "natural" terminal state
        reward = np.random.random()
        return next_state, terminal, reward



environment = Environment.create(
    environment=CustomEnvironment, max_episode_timesteps=100
)

environment = Environment.create(
    environment='gym', level='CartPole', max_episode_timesteps=500
)
print('environment done')

agent = Agent.create(
    agent='tensorforce', environment=environment, update=64,
    optimizer=dict(optimizer='adam', learning_rate=1e-3),
    objective='policy_gradient', reward_estimation=dict(horizon=20)
)
print('agent done')
runner = Runner(
    agent='agent.json',
    environment=dict(environment='gym', level='CartPole'),
    max_episode_timesteps=500,
    num_parallel=5, remote='multiprocessing'
)
print('runner done')
runner.run(num_episodes=100)

runner.close()
