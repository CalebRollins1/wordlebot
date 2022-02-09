import numpy.random as rd
import string
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import gym
import warnings
import tensoflow as tf
warnings.filterwarnings("ignore")




guesses = []
solutions = []

for elem in list(open('wordlist_guesses.txt')):
    guesses.append(elem.strip())

for elem in list(open('wordlist_solutions.txt')):
    solutions.append(elem.strip())


new_words = rd.choice(solutions,500)

guesses = new_words
solutions = new_words

class Wordle:

    def __init__(self,answer):
        self.answer = answer
        self.key = [0,0,0,0,0]
        self.current_guess = None
        self.greens = []
        self.yellows = set()
        self.grays = set()
        self.letter_dict = {elem:i for i,elem in enumerate(string.ascii_lowercase)}
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
                if guess[i] in self.yellows:
                    self.yellows.remove(guess[i])
            elif guess[i] in self.answer:
                self.key[i]+=1
                self.yellows.add(guess[i])
            else:
                self.grays.add(guess[i])
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
        return [self.greens,self.yellows,self.grays]

    def state(self,guess):
        self.key = [0,0,0,0,0]

        g = self.greens.copy()
        y = self.yellows.copy()
        b = self.grays.copy()
        for i in range(5):
            if guess[i]==self.answer[i]:
                self.key[i]+=2
                g.append((i,guess[i]))
                if guess[i] in y:
                    y.remove(guess[i])
            elif guess[i] in self.answer:
                self.key[i]+=1
                y.add(guess[i])
            else:
                b.add(guess[i])
        return [g,y,b]

    def reward(self,action):
        g,y,b = self.state(action)
        if self.key == [2,2,2,2,2]:
            return 50
        else:
            return 2*len(g)+len(y)

    def ongoing(self,turns):
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


class TSP:

    def __init__(self,num_of_points):

        self.num_of_points = num_of_points
        self.points = [[rd.random(),rd.random()] for i in range(self.num_of_points)]



class SimpleExample:

    def __init__(self,grid,location):
        self.grid = grid
        self.location = location
        self.starting_location = location


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
            r+=2

        return r

    def actions(self):
        return [(0,1),(1,0),(-1,0),(0,-1)]

    def ongoing(self,turns):
        x,y = self.location
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

    def reset(self):
        self.location = self.starting_location






class Qlearner:

    def __init__(self,env,testing,learner,alpha,gamma,epsilon):
        #environement methods:
        #state
        #encode
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
        self.Q = functional_Q(learner)

    def pretrain(self):
        pass

    def train(self,rounds):
        for i in range(rounds):
            X_train = []
            y_train = []
            training_set = set()
            #for j,sol in enumerate(solutions):
            #for j in range(250):
            for j,W in enumerate(self.env_list):
                #print(j)
                #W = Wordle(rd.choice(solutions))
                if True:#i==0:
                    max_a = max(W.actions(), key = lambda action:self.Q.get_value(W.encode(W.state(action),action)))

                else:
                    guesses_encoded = []
                    guess_sample = rd.choice(guesses,500,replace = False)
                    for elem in guess_sample:
                        guesses_encoded.append(encode(W.state(elem),elem))
                    idx = np.argmax(self.Q.model.predict(np.array(guesses_encoded)))
                    max_a = guess_sample[idx]

                turns = 0
                while W.ongoing(turns):

                    X_train.append(W.encode(W.current_state(),max_a))
                    #print(encode(W.current_state(),max_a))
                    u = (1-self.alpha)*self.Q.get_value(W.encode(W.current_state(),max_a))+self.alpha*W.reward(max_a)
                    if rd.random()>self.epsilon:
                        W.make_guess(max_a)
                    else:
                        act = W.actions()
                        idx = rd.choice(len(act))
                        W.make_guess(W.actions()[idx])
                    max_a = max(W.actions(), key = lambda action:self.Q.get_value(W.encode(W.state(action),action)))
                    u+=self.alpha*self.gamma*self.Q.get_value(W.encode(W.current_state(),max_a))
                    y_train.append(u)
                    #print(u)
                    #quit()

                    #if encode(W.current_state,max_a) not in training_set:

                    turns+=1
                W.reset()


            self.Q.update(np.array(X_train),y_train)

    def test(self):
        s = 0
        n = 0
        for W in self.env_test:
            #W = Wordle(sol)
            turns = 0
            while W.ongoing(turns):
                max_a = max(W.actions(), key = lambda action:self.Q.get_value(W.encode(W.state(action),action)))

                W.make_guess(max_a)
                turns+=1
            s+=turns
            n+=1
            W.reset()

        print(s/n)





if __name__ == '__main__':


    env_train = []
    env_test = []

    for i in range(600):
        grid = np.zeros([5,5])

        x = rd.choice(5)
        y = rd.choice(5)

        grid[x,y] = 1

        x = rd.choice(5)
        y = rd.choice(5)

        loc = (x,y)

        if i<500:
            env_train.append(SimpleExample(grid,loc))
        else:
            env_test.append(SimpleExample(grid,loc))



    #Q = Qlearner(Wordle,MLPRegressor((12*26,60,1)),0.9,0.25)
    Q = Qlearner(env_train,env_test,MLPRegressor((29,20,1)),0.75,0.75,0.1)
    for i in range(50):
        Q.train(19)
        Q.test()
