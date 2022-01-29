import time
from numba import jit
from numba.experimental import jitclass

guesses = []
solutions = []

for elem in list(open('wordlist_guesses.txt')):
    guesses.append(elem.strip())

for elem in list(open('wordlist_solutions.txt')):
    solutions.append(elem.strip())

@jitclass
class Wordle:

    def __init__(self,answer):
        self.answer = answer
        self.key = [0,0,0,0,0]
        self.current_guess = None
        self.greens = []
        self.yellows = set()
        self.grays = set()


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
        y = 0
        for elem in word:
            if elem in self.grays:
                return False
            if elem in self.yellows:
                y+=1
        if y!=len(self.yellows):
            return False

        return True




    
    def get_remaining_words(self):
        L = []
        for elem in solutions:
            if self.possible(elem):
                L.append(elem)

        return L

@jit(nopython=True)
def main():
    s = 0
    n = 0
    score_dict = {}


    for g in guesses[1:10]:
        s = 0
        n = 0
        for sol in solutions:

            W = Wordle(sol)
            W.make_guess(g)
            #W.make_guess('least')
            #print(W.yellows)
            s+=len(W.get_remaining_words())
            n+=1

        score_dict[g] = s/n



    print(score_dict)
    print(min(score_dict,key = lambda x: score_dict[x]))


t1 = time.time()
main()
t2 = time.time()

print(t2-t1)
