import numpy.random as rd
import string
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import gym
import warnings
import tensorflow as tf
warnings.filterwarnings("ignore")




guesses = []
solutions = []

for elem in list(open('wordlist_guesses.txt')):
    guesses.append(elem.strip())

for elem in list(open('wordlist_solutions.txt')):
    solutions.append(elem.strip())


new_words = rd.choice(solutions,500)

#guesses = new_words
#solutions = new_words

class Wordle:

    def __init__(self,answer):
        self.answer = answer
        self.key = [0,0,0,0,0]
        self.current_guess = None
        self.greens = []
        self.yellows = set()
        self.grays = set()
        self.letter_dict = {elem:i for i,elem in enumerate(string.ascii_lowercase)}
        self.guesses = []
        self.state_encoded = np.ones([6,26])
        self.turns = 0
        #self.set_dict = set_dict


    def make_set_dict(self):
        self.set_dict['all'] = set(guesses)

        for elem in string.ascii_lowercase:
            for i in range(5):
                self.set_dict[(i,elem)] = set()
            self.set_dict[('y',elem)] = set()
            self.set_dict[('gray',elem)] = set(guesses)

        for elem in guesses:
            for i in range(5):
                self.set_dict[(i,elem[i])].add(elem)
                self.set_dict[('y',elem[i])].add(elem)
                self.set_dict[('gray',elem[i])].remove(elem)



    def make_guess(self,guess):
        self.key = [0,0,0,0,0]
        for i in range(5):
            if guess[i]==self.answer[i]:
                self.key[i]+=2
                self.greens.append((i,guess[i]))
                self.state_encoded[i,:] = 0
                self.state_encoded[i,self.letter_dict[guess[i]]] = 1
                if guess[i] in self.yellows:
                    self.yellows.remove(guess[i])
                    self.state_encoded[5,self.letter_dict[guess[i]]] = 0

            elif guess[i] in self.answer:
                self.key[i]+=1
                self.yellows.add(guess[i])
                self.state_encoded[5,self.letter_dict[guess[i]]] = 1
                self.state_encoded[i,self.letter_dict[guess[i]]] = 0
            else:
                self.grays.add(guess[i])
                self.state_encoded[:,self.letter_dict[guess[i]]] = 0
        self.current_guess = guess
        return self.key





    def score(self):
        s = 0
        for i,elem in enumerate(self.key):
            s+=self.score_dict[self.current_guess[i]+str(elem)]

        return s


    def possible(self,word):

        for i,elem in self.greens:
            if word[i]!=elem:
                return False

        for elem in self.yellows:
            if elem not in word:
                return False

        for elem in word:
            if elem in self.grays:
                return False



        return True

    def current_state(self):
        return self.state_encoded

    def state(self,guess):
        self.key = [0,0,0,0,0]

        g = self.greens.copy()
        y = self.yellows.copy()
        b = self.grays.copy()
        state_copy = self.state_encoded.copy()
        for i in range(5):
            if guess[i]==self.answer[i]:
                self.key[i]+=2
                g.append((i,guess[i]))
                state_copy[i,:] = 0
                state_copy[i,self.letter_dict[guess[i]]] = 1
                if guess[i] in y:
                    y.remove(guess[i])
                    state_copy[5,self.letter_dict[guess[i]]] = 0
            elif guess[i] in self.answer:
                self.key[i]+=1
                y.add(guess[i])
                state_copy[5,self.letter_dict[guess[i]]] = 1
                state_copy[i,self.letter_dict[guess[i]]] = 0
            else:
                b.add(guess[i])
                state_copy[:,self.letter_dict[guess[i]]] = 0
        return state_copy

    def reward(self,action):
        state = self.state(action)
        g = np.sum(np.sum(state[:-1,:],axis = 1)==1)
        y = np.sum(state[-1:,])
        if self.key == [2,2,2,2,2]:
            return 100-5*self.turns
        else:
            return 2*g+y

    def ongoing(self,turns):
        self.turns = turns
        if (turns==10) or (self.key == [2,2,2,2,2]):
            return False
        else:
            return True

    def encode(state,action):
        L = [0 for i in range(12*26)]
        g,y,b = state
        for i,elem in enumerate(action):
            L[26*i+self.letter_dict[elem]] = 1
        for i,elem in g:
            L[26*5+26*i+self.letter_dict[elem]] = 1
        for elem in y:
            L[26*10+self.letter_dict[elem]] = 1
        for elem in b:
            L[26*11+self.letter_dict[elem]] = 1
        return L


    def encode_state(self,state):
        return state.T.reshape(-1)



    def get_remaining_words(self):
        L = []
        for elem in solutions:
            if self.possible(elem):
                L.append(elem)


        return L

    def get_remaining_words_fast(self):
        possible = self.set_dict['all'].copy()


        for elem in self.greens:
            possible = possible.intersection(self.set_dict[elem])

        #print(possible)

        for elem in self.yellows:
            possible = possible.intersection(self.set_dict[('y',elem)])

        #print(possible)

        for elem in self.grays:
            possible = possible.intersection(self.set_dict[('gray',elem)])

        #print(possible)

        return possible

    def actions(self):
        return guesses

    def reset(self):
        self.state_encoded = np.ones([6,26])
        self.turns = 0
        self.greens = []
        self.yellows = set()
        self.grays = set()
        self.key = [0,0,0,0,0]
        self.guesses = []







class random_Q:

    def __init__(self):
        pass

    def get_value(self,state,action):
        return rd.random()

    def update(self,X_train,y_train):
        pass



class functional_Q:

    def __init__(self,model):

        self.model = model

        self.trained = False

    def get_value(self,encoded_state_action):
        if self.trained:
            return float(self.model.predict(np.array(encoded_state_action).reshape(1,-1)))
        else:
            return rd.random()

    def update(self,X_train,y_train):
        self.trained = True
        self.model.fit(X_train,y_train)

class DQN:

    def __init__(self):

        #self.model = model
        self.trained = False

        self.model = tf.keras.models.Sequential([
          tf.keras.layers.Dense(6*26, activation='relu'),
          tf.keras.layers.Dense(500,activation = 'relu'),
          tf.keras.layers.Dense(1000,activation = 'relu'),
          tf.keras.layers.Dense(len(guesses))
        ])

        loss_fn = tf.keras.losses.MeanSquaredError()

        self.model.compile(optimizer='adam',
                  loss=loss_fn)

    def get_value(self,encoded_state):
        if self.trained:
            return self.model(np.array(encoded_state).reshape(1,-1))
        else:
            return rd.random([1,len(guesses)])

    def update(self,X_train,y_train):
        self.trained = True
        self.model.fit(X_train,y_train,epochs = 25,verbose = 0)


class TSP:

    def __init__(self,num_of_points):

        self.num_of_points = num_of_points
        self.points = [[rd.random(),rd.random()] for i in range(self.num_of_points)]



class SimpleExample:

    def __init__(self,grid,location,goal):
        self.grid = grid
        self.location = location
        self.starting_location = location
        self.goal = goal
        self.turns = 0


    def state(self,action):
        x,y = action
        xc,yc = self.location
        if (abs(xc+x-2)<=2.5):
            new_x = xc+x
        else:
            new_x = xc
        if abs(yc+y-2)<=2.5:
            new_y = yc+y
        else:
            new_y = yc
        new_location = (new_x,new_y)

        return self.grid,new_location

    def current_state(self):
        return self.grid,self.location

    def encode(self,state,action):
        L = []
        grid,loc = state
        for row in grid:
            for elem in row:
                 L.append(elem)
        for elem in loc:
            L.append(elem)
        for elem in action:
            L.append(elem)
        return L

    def encode_state(self,state):
        L = []
        grid,loc = state
        g = grid.copy()

        x,y = loc

        g[x][y] = -1
        for row in g:
            for elem in row:
                 L.append(elem)

        return L

    def reward(self,action):
        r = 0
        x,y = action
        xc,yc = self.location
        if (abs(xc+x-2)<=2.55):
            new_x = xc+x
        else:
            new_x = xc
            r-=5
        if abs(yc+y-2)<=2.5:
            new_y = yc+y
        else:
            new_y = yc
            r-=5
        new_location = (new_x,new_y)

        if self.grid[new_x][new_y] == 1:
            r+=(125-5*self.turns)

        if abs(xc-self.goal[0])>abs(new_x-self.goal[0]):
            r+=0.1

        if abs(yc-self.goal[1])>abs(new_y-self.goal[1]):
            r+=0.1

        return r

    def actions(self):
        return [(0,1),(1,0),(-1,0),(0,-1)]

    def ongoing(self,turns):
        x,y = self.location
        self.turns = turns
        return ((self.grid[x][y]!=1) and (turns<25))

    def make_guess(self,action):
        x,y = action
        xc,yc = self.location
        if (abs(xc+x-2)<=2.5):
            new_x = xc+x
        else:
            new_x = xc
        if abs(yc+y-2)<=2.5:
            new_y = yc+y
        else:
            new_y = yc
        new_location = (new_x,new_y)
        self.location = new_location

        return self.grid,new_location

    def show_board(self):
        g = self.grid.copy()

        x,y = self.location

        g[x][y] = 2

        for elem in g:
            print(elem)

    def reset(self):
        self.location = self.starting_location
        self.turns = 0






class Qlearner:

    def __init__(self,env,testing,learner,alpha,gamma,epsilon):
        #environement methods:
        #state
        #encode_state
        #ongoing
        #current_state
        #make_guess
        #actions
        #reward
        self.env_list = env
        self.env_test = testing
        self.learner = learner
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        #self.Q = functional_Q(learner)
        self.Q = DQN()

    def pretrain(self):
        pass

    def train(self,rounds):
        for i in range(rounds):
            X_train = []
            y_train = []
            training_set = set()
            #for j,sol in enumerate(solutions):
            #for j in range(250):
            print(i)
            for j,W in enumerate(self.env_list):
                #print(j)
                #W = Wordle(rd.choice(solutions))
                #print(j)


                turns = 0
                while W.ongoing(turns):
                    #print(self.Q.get_value(W.encode_state(W.current_state())))
                    #print(tf.math.argmax(self.Q.get_value(W.encode_state(W.current_state())),axis = 1))
                    max_a = W.actions()[int(tf.math.argmax(self.Q.get_value(W.encode_state(W.current_state())),axis = 1).numpy())]
                    #print('a')

                    X_train.append(W.encode_state(W.current_state()))

                    new_y = []

                    for elem in W.actions():
                        new_y.append(W.reward(elem))#+self.gamma*float(tf.math.reduce_max(self.Q.get_value(W.encode_state(W.state(elem))))))
                    #print(encode(W.current_state(),max_a))
                    if rd.random()>self.epsilon:
                        W.make_guess(max_a)
                    else:
                        act = W.actions()
                        idx = rd.choice(len(act))
                        W.make_guess(W.actions()[idx])

                    #print('took action')
                    y_train.append(new_y.copy())
                    #print(u)
                    #quit()

                    #if encode(W.current_state,max_a) not in training_set:

                    turns+=1
                W.reset()

            self.Q.update(np.array(X_train),np.array(y_train))

    def test(self):
        s = 0
        n = 0
        for W in self.env_test:
            #W = Wordle(sol)
            turns = 0
            while W.ongoing(turns):
                max_a = W.actions()[int(tf.math.argmax(self.Q.get_value(W.encode_state(W.current_state())),axis = 1).numpy())]
                if False:#n==0:
                    print(W.reward(max_a))
                W.make_guess(max_a)
                turns+=1
                if False:#n==0:
                    print(self.Q.get_value(W.encode_state(W.current_state())))

                    W.show_board()
            s+=turns
            n+=1
            W.reset()

        print(s/n)


class Reinforce:

    def __init__(self,env,testing,learner,alpha,gamma,epsilon):
        #environement methods:
        #state
        #encode_state
        #ongoing
        #current_state
        #make_guess
        #actions
        #reward
        self.env_list = env
        self.env_test = testing
        self.learner = learner
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        #self.Q = functional_Q(learner)
        self.network = tf.keras.Sequential([
            tf.keras.layers.Dense(6*26, activation='relu', kernel_initializer=tf.keras.initializers.he_normal()),
            tf.keras.layers.Dense(3000, activation='relu', kernel_initializer=tf.keras.initializers.he_normal()),
            tf.keras.layers.Dense(len(guesses), activation='softmax')
        ])
        self.network.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam())

    def train(self,num_episodes):

        for episode in range(num_episodes):
            #state = env.reset()
            rewards = []
            states = []
            actions = []
            for W in self.env_list:
                W.reset()
                turn = 0
                print('working')
                while True:
                    action = self.get_action(self.network, W)
                    #new_state, reward, done, _ = env.step(action)
                    #new_state = W.encode_state(W.state(action))
                    reward = W.reward(action)
                    done = (not W.ongoing(turn))
                    states.append(W.encode_state(W.current_state()))
                    rewards.append(reward)
                    actions.append(action)
                    if done:
                        loss = self.update_network(self.network, rewards, states, actions, len(actions))
                        tot_reward = sum(rewards)
                        print(f"Episode: {episode}, Reward: {tot_reward}, avg loss: {loss:.5f}")
                        with train_writer.as_default():
                            tf.summary.scalar('reward', tot_reward, step=episode)
                            tf.summary.scalar('avg loss', loss, step=episode)
                        break
                    turn+=1
                    W.make_guess(action)

    def get_action(self,network,env):
        softmax_out = self.network(env.encode_state(env.current_state()).reshape((1, -1)))
        selected_action = np.random.choice(env.actions(), p=softmax_out.numpy()[0])
        return selected_action

    def update_network(self,network, rewards, states, actions, num_actions):
        reward_sum = 0
        discounted_rewards = []
        for reward in rewards[::-1]:  # reverse buffer r
            reward_sum = reward + self.gamma*reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        discounted_rewards = np.array(discounted_rewards)
        # standardise the rewards
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        states = np.vstack(states)
        print(tf.gradients(discounted_rewards,self.network))
        loss = self.network.train_on_batch(states, discounted_rewards)
        return loss



    def test(self):
        pass




if __name__ == '__main__':



    '''
    X_train = []
    y_train = []
    for i in range(500):
        X_train.append(rd.random(128))
        y_train.append(rd.random(10))



    model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10)
    ])

    loss_fn = tf.keras.losses.MeanSquaredError()

    model.compile(optimizer='adam',
              loss=loss_fn)

    model.fit(np.array(X_train), np.array(y_train), epochs=5)

    print(tf.math.argmax(model(np.array([rd.random(128)])),axis = 1))

    quit()



    env_train = []
    env_test = []

    for i in range(600):
        grid = np.zeros([5,5])

        x = rd.choice(5)
        y = rd.choice(5)

        grid[x,y] = 1

        goal = (x,y)

        x = rd.choice(5)
        y = rd.choice(5)

        loc = (x,y)

        if i<500:
            env_train.append(SimpleExample(grid,loc,goal))
        else:
            env_test.append(SimpleExample(grid,loc,goal))
    '''
    env_train = []
    env_test = []
    for i in range(500):

        if i<475:
            env_train.append(Wordle(rd.choice(solutions)))
        else:
            env_test.append(Wordle(rd.choice(solutions)))




    #Q = Qlearner(Wordle,MLPRegressor((12*26,60,1)),0.9,0.25)
    #Q = Qlearner(env_train,env_train,DQN(),0.5,0.5,0.1)
    Q = Reinforce(env_train,env_test,DQN(),0.95,0.95,0.95)
    for i in range(50):
        Q.train(200)
        Q.test()
