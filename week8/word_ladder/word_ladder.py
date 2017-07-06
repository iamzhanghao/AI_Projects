class Word:

    history = []

    def __init__(self,initial):
        self.history.append(initial)

    def get_current(self):
        return self.history[-1]

    def change(self,new_word):
        self.history.append(new_word)



def dfs(frontier, goal):


    pass







dfs(["doge"],"good")