import time
from numba import jit
from numba.experimental import jitclass
import string
import numpy.random as rd
import numpy as np

guesses = []
solutions = []

for elem in list(open('wordlist_guesses.txt')):
    guesses.append(elem.strip())

for elem in list(open('wordlist_solutions.txt')):
    solutions.append(elem.strip())


def make_set_dict():
    set_dict = {}
    set_dict['all'] = set(solutions)

    for elem in string.ascii_lowercase:
        for i in range(5):
            set_dict[(i,elem)] = set()
        set_dict[('y',elem)] = set()
        set_dict[('gray',elem)] = set(solutions)

    for elem in solutions:
        for i in range(5):
            set_dict[(i,elem[i])].add(elem)
            set_dict[('y',elem[i])].add(elem)
            if elem in set_dict[('gray',elem[i])]:
                set_dict[('gray',elem[i])].remove(elem)

    return set_dict

set_dict = make_set_dict()



#@jitclass
class Wordle:

    def __init__(self,answer):
        self.answer = answer
        self.key = [0,0,0,0,0]
        self.current_guess = None
        self.greens = []
        self.yellows = set()
        self.grays = set()
        self.set_dict = set_dict


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



#@jit(nopython=True)
def main():
    s = 0
    n = 0
    score_dict = {}


    for g in guesses[1:100]:
        s = 0
        n = 0
        for sol in solutions:

            W = Wordle(sol)
            W.make_guess(g)
            #W.make_guess('least')
            #print(W.yellows)

            s+=len(W.get_remaining_words_fast())

            n+=1

        score_dict[g] = s/n



    #print(score_dict)
    print(min(score_dict,key = lambda x: score_dict[x]))


class GeneticModel:

    def __init__(self,weights):
        self.weights = weights
        if weights == 'random':
            self.weights = {}
            for letter in string.ascii_lowercase:
                self.weights[letter] = rd.exponential()
        self.score_dict = {}


    def make_score_dict(self):

        for word in solutions:
            self.score_dict[word] = sum([self.weights[elem] for elem in word])

    def cross(self,other):

        w = {}
        for elem in self.weights:
            if rd.random()>0.2:
                w[elem] = (self.weights[elem]+other.weights[elem])/2
            else:
                w[elem] = rd.exponential()

        return GeneticModel(w)

    def cross_random(self,other,score_self,score_other):

        w = {}
        for elem in self.weights:
            if rd.random()>0.2:
                if rd.random()<np.exp(score_self)/(np.exp(score_self)+np.exp(score_other)):
                    w[elem] = self.weights[elem]
                else:
                    w[elem] = other.weights[elem]

            else:
                w[elem] = rd.exponential()

        return GeneticModel(w)


def score_genetic_model(model):

    model.make_score_dict()

    first_guess = max(model.score_dict, key = lambda x:model.score_dict[x])

    turn_list = {i+1:0 for i in range(9)}
    s = 0
    n = 0

    for i,sol in enumerate(solutions):

        turns = 1



        old_guesses = set()

        W = Wordle(sol)
        W.make_guess(first_guess)
        #W.make_guess('least')
        #print(W.yellows)
        old_guesses.add(first_guess)
        while W.key!=[2,2,2,2,2]:
            possible = W.get_remaining_words_fast()

            if len(possible.union(old_guesses))!=0:
                for elem in old_guesses:
                    if elem in possible:
                        possible.remove(elem)


            next_guess = max(possible,key = lambda x: model.score_dict[x])
            #print(possible,sol)

            W.make_guess(next_guess)

            old_guesses.add(next_guess)

            turns+=1


        turn_list[turns]+=1
        s+=turns
        n+=1
    print(turn_list)
    return s/n


def score_genetic_model_gy(model):

    model.make_score_dict()

    g_score = model.weights['a']
    y_score = model.weights['b']


    turn_list = {i+1:0 for i in range(9)}
    s = 0
    n = 0

    for i,sol in enumerate(solutions):

        turns = 0



        old_guesses = set()

        W = Wordle(sol)
        W.make_guess(first_guess)
        #W.make_guess('least')
        #print(W.yellows)
        old_guesses.add(first_guess)
        while W.key!=[2,2,2,2,2]:
            possible = W.get_remaining_words_fast()

            if len(possible.union(old_guesses))!=0:
                for elem in old_guesses:
                    if elem in possible:
                        possible.remove(elem)


            next_guess = max(possible,key = lambda x: model.score_dict[x])
            #print(possible,sol)

            W.make_guess(next_guess)

            old_guesses.add(next_guess)

            turns+=1


        turn_list[turns]+=1
        s+=turns
        n+=1
    print(turn_list)
    return s/n



w = {
'a':5.5,
'b':2.6,
'c':2.3,
'd':2.4,
'e':5.2,
'f':2.1,
'g':2.8,
'h':2.9,
'i':3.7,
'j':1.9,
'k':2.5,
'l':3.5,
'm':2.05,
'n':3.1,
'o':3.03,
'p':3,
'q':1.5,
'r':2.58,
's':4,
't':3.9,
'u':2.85,
'v':1.1,
'w':1.3,
'x':0.5,
'y':2.99,
'z':1
}

w = {'a': 0.7312658381689322, 'b': 0.6825283852364701, 'c': 0.7575540902207294, 'd': 0.8728456107209767, 'e': 0.5815954621928157, 'f': 0.45757549865634745, 'g': 0.7983745008057204, 'h': 0.9647673555432016, 'i': 0.39443551190347176, 'j': 0.5357945859123329, 'k': 0.30579282090710075, 'l': 1.2295332502338374, 'm': 0.2781897055998782, 'n': 1.0867012761875392, 'o': 0.5529405786970685, 'p': 1.0696114898224764, 'q': 0.5003602996278053, 'r': 0.9209800637180336, 's': 0.4880233693416131, 't': 1.2642882963222455, 'u': 0.5881645195236695, 'v': 0.4229886056769404, 'w': 0.4066056562894226, 'x': 0.5736988915330807, 'y': 0.416187824472656, 'z': 0.32953063871212185}

print(score_genetic_model(GeneticModel(w)))

quit()


def genetic_algorithm(rounds,pool_size,score_func):
    pool = {}
    pool[0] = {GeneticModel('random'):0 for i in range(pool_size)}
    print('pool made')
    for i in range(rounds):

        print('round {0}'.format(i,'{}'))

        for j,model in enumerate(pool[i]):
            #print('round {0} model {1}'.format(i,j,'{}'))
            pool[i][model] = score_func(model)

        top_models = sorted(pool[i],key = lambda x: pool[i][x])[:75]

        pool[i+1] = {elem:0 for elem in top_models[:10]}

        for j in range(490):
            father,mother = rd.choice(top_models,2)
            #pool[i+1][father.cross(mother,pool[i][father],pool[i][mother])] = 0
            pool[i+1][father.cross(mother)] = 0

        '''
        for father in top_models:
            for mother in top_models:
                pool[i+1][father.cross(mother)] = 0
        '''


        print(pool[i][top_models[0]])
        print(top_models[0].weights)


genetic_algorithm(50,500)

quit()


t1 = time.time()
main()
t2 = time.time()

print(t2-t1)
