#!/usr/bin/env python2
# Written by Patricia Suriana, MIT ca. 2013
# Modified by Tomas Lozano-Perez, MIT ca 2016


'''
comparison between h_G and h_FF:
          h_G                                  h_FF
sussman     32 states expanded, path cost= 6    20 states expanded, path cost= 6
large-a   1469 states expanded, path cost=12   109 states expanded, path cost=12
large-b   cannot solve in 5 mins               356 states expanded, path cost=18

'''

import pddl_parser
import search
import time
import sys
import pdb


class PlanProblem(search.Problem):
    def __init__(self, task):
        search.Problem.__init__(self, task.initial_state, task.goals)
        self.task = task

    def actions(self, state):
        return self.task.get_successor_ops(state)

    def result(self, state, action):
        return action.apply(state)

    def goal_test(self, state):
        return self.task.goal_reached(state)

    def forward(self, state):
        '''forward pass'''
        self.fl = {}
        self.al = {}
        self.ml = None

        facts = set(state)
        for fact in facts:
            self.fl[fact] = 0

        step = 0
        while True:
            if self.task.goal_reached(facts):
                self.ml = step
                break
            changed = False
            for action in self.task.get_successor_ops(facts):
                if action not in self.al:
                    self.al[action] = step
                for fact in action.apply(facts, noDel=True):
                    if fact not in self.fl:
                        self.fl[fact] = step + 1
                        facts.add(fact)
                        changed = True
            if not changed:
                break
            step += 1

    def hff(self, n):
        '''backward pass'''
        self.forward(n.state)
        if self.ml is None:
            return search.inf

        selected = set()

        G_t = []
        for i in range(self.ml + 1):
            G_t.append(set())
        for fact in self.task.goals:
            G_t[self.fl[fact]].add(fact)

        for step in range(self.ml, 0, -1):
            for fact in G_t[step]:
                for action in self.al:
                    if self.al[action] == step - 1:
                        if fact in action.add_effects:
                            selected.add(action)
                            for p in action.preconditions:
                                G_t[self.fl[p]].add(p)

        return len(selected)

    def hmax(self, n):
        self.forward(n.state)
        return max(self.fl[f] for f in self.task.goals)

    def hsum(self, n):
        self.forward(n.state)
        return sum(self.fl[f] for f in self.task.goals)

    def h_g(self, n):
        return len(self.task.goals - n.state)


def printOutputVerbose(tic, toc, path, cost, final_state, goal):
    print ("\n******************************FINISHED TEST******************************")
    print ("Goals: ")
    for state in goal:
        print("\t" + str(state))
    print('\nRunning time: ', (toc - tic), 's')
    if path == None:
        print('\tNO PATH FOUND')
    else:
        print("\nNumber of Actions: ", len(path))
        print('\nCost:', cost)
        print("\nPath: ")
        print()
        for op in path:
            print("\t" + repr(op))
        print("\nFinal States:")
        for state in final_state:
            print("\t" + str(state))
    print("*************************************************************************\n")


def printOutput(tic, toc, path, cost):
    print((toc - tic), '\t', len(path), '\t', cost)


if __name__ == "__main__":
    args = sys.argv
    if len(args) != 3:  # default task
        dirName = "painting"
        fileName = "p0"
    else:
        dirName = args[1]
        fileName = args[2]
    domain_file = dirName + '/domain.pddl'
    problem_file = dirName + '/' + fileName + '.pddl'

    # task is an instance of the Task class defined in strips.py
    task = pddl_parser.parse(domain_file, problem_file)

    # This should be commented out for larger tasks
    # print task

    print("\n******************************START TEST******************************")

    tic = time.time()

    # Define a sub-class of the Problem class, make an instance for the task and call the search
    # You should then set the variables:
    # final_state - the final state at the end of the plan
    # plan - a list of actions representing the plan
    # cost - the cost of the plan
    # Your planner_v2 here
    p = search.InstrumentedProblem(PlanProblem(task))
    # soln = search.astar_search(p, lambda n: 0)
    # soln = search.astar_search(p, lambda n: p.h_g(n))
    # soln = search.astar_search(p, lambda n: p.hmax(n))
    # soln = search.astar_search(p, lambda n: p.hsum(n))
    soln = search.astar_search(p, lambda n: p.hff(n))
    print('search stats', p)
    if soln is None:
        print ('no solution found')
        exit()

    plan = soln.solution()
    cost = soln.path_cost
    final_state = soln.state

    toc = time.time()
    printOutputVerbose(tic, toc, plan, cost, final_state, task.goals)
