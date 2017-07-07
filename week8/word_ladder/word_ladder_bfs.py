import copy

import sys

sys.setrecursionlimit(15000)


class Word:
    history = []

    def __init__(self, initial, history=None):
        if history is None:
            self.history.append(initial)
        else:
            self.history = history

    def get_current(self):
        return self.history[-1]

    def change(self, new_word):
        self.history.append(new_word)

    def get_history(self):
        return self.history

    def get_initial(self):
        return self.history[0]


def replace(old, char, index):
    return old[:index] + char + old[index + 1:]


WORDS = set(i.lower().strip() for i in open('words2.txt'))


def is_valid_word(word):
    return word in WORDS


charset = "zyxwvutsrqponmlkjihgfedcba"



def dfs(frontier, explored, goal):
    if len(frontier) == 0:
        return "No Solution"
    else:
        # for element in frontier:
        #     print(element.get_current(),end=" ")
        # print()
        old_word_frontier = frontier[0]
        if old_word_frontier.get_current() == goal:
            return old_word_frontier.get_history()
        frontier.pop(0)

    for index in range(len(goal)):
        for char in charset:
            new_word = replace(old_word_frontier.get_current(), char, index)
            if new_word not in explored and is_valid_word(new_word):
                # print("add " + new_word)
                explored.add(new_word)
                history = copy.copy(old_word_frontier.get_history())
                new_word_frontier = Word(old_word_frontier.get_current(), history)
                new_word_frontier.change(new_word)
                frontier.append(new_word_frontier)


    return dfs(frontier, explored, goal)


start = "best"
target = "math"

frontier = []
explored = set()

frontier.append(Word(start))
explored.add(start)

print(dfs(frontier, explored, target))




# dfs(["doge"], "good")
