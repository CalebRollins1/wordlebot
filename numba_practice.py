import time
from numba import jit
from numba.experimental import jitclass
import numpy as np


guesses = []
solutions = []

for elem in list(open('wordlist_guesses.txt')):
    guesses.append(elem.strip())

for elem in list(open('wordlist_solutions.txt')):
    solutions.append(elem.strip())


@jit(nopython=True)
def h(solutions):
    n = 0
    for s in solutions[:100]:
        n+=1
    return n

@jit(nopython=True)
def f(guesses,solutions):
    n = 0
    for g in guesses:
        n+=h(solutions)
    return n



@jit(nopython=True)
def main(guesses,solutions):
    best_word = ''
    score = 0
    for sol in solutions:
        n+=f(guesses,solutions)




t1 = time.time()
main(guesses,solutions)
t2 = time.time()
print(t2-t1)

t1 = time.time()
main(guesses,solutions)
t2 = time.time()
print(t2-t1)
