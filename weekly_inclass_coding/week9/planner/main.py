# Written by Patricia Suriana, MIT ca. 2013
# Modified by Tomas Lozano-Perez, MIT ca 2016

import pddl_parser
import search
import time
import sys
import pdb


def printOutputVerbose(tic, toc, path, cost, final_state, goal):
    print "\n******************************FINISHED TEST******************************"
    print "Goals: "
    for state in goal:
        print "\t" + str(state)
    print '\nRunning time: ', (toc - tic), 's'
    if path == None:
        print '\tNO PATH FOUND'
    else:
        print "\nNumber of Actions: ", len(path)
        print '\nCost:', cost
        print "\nPath: "
        for op in path:
            print "\t" + repr(op)
        print "\nFinal States:"
        for state in final_state:
            print "\t" + str(state)
    print "*************************************************************************\n"


def printOutput(tic, toc, path, cost):
    print (toc - tic), '\t', len(path), '\t', cost


if __name__ == "__main__":
    args = sys.argv
    if len(args) != 3:  # default task
        dirName = "prodigy-bw"
        fileName = "bw-simple"
    else:
        dirName = args[1]
        fileName = args[2]
    domain_file = dirName + '/domain.pddl'
    problem_file = dirName + '/' + fileName + '.pddl'

    # task is an instance of the Task class defined in strips.py
    task = pddl_parser.parse(domain_file, problem_file)

    # This should be commented out for larger tasks


    print "\n******************************START TEST******************************"

    tic = time.time()


    # Define a sub-class of the Problem class, make an instance for the task and call the search


    # class GraphProblem(Problem):
    #
    #     """The problem of searching a graph from one node to another."""
    #
    #     def __init__(self, initial, goal, graph):
    #         Problem.__init__(self, initial, goal)
    #         self.graph = graph
    #
    #     def actions(self, A):
    #         """The actions at a graph node are just its neighbors."""
    #         return list(self.graph.get(A).keys())
    #
    #     def result(self, state, action):
    #         """The result of going to a neighbor is just that neighbor."""
    #         return action
    #
    #     def path_cost(self, cost_so_far, A, action, B):
    #         return cost_so_far + (self.graph.get(A, B) or infinity)
    #
    #     def find_min_edge(self):
    #         """Find minimum value of edges."""
    #         m = infinity
    #         for d in self.graph.dict.values():
    #             local_min = min(d.values())
    #             m = min(m, local_min)
    #
    #         return m
    #
    #     def h(self, node):
    #         """h function is straight-line distance from a node's state to goal."""
    #         locs = getattr(self.graph, 'locations', None)
    #         if locs:
    #             if type(node) is str:
    #                 return int(distance(locs[node], locs[self.goal]))
    #
    #             return int(distance(locs[node.state], locs[self.goal]))
    #         else:
    #             return infinity


    class MyProblem(search.Problem):

        """The abstract class for a formal problem.  You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

        def __init__(self, initial, goal=None):
            """The constructor specifies the initial state, and possibly a goal
          state, if there is a unique goal.  Your subclass's constructor can add
          other arguments."""
            self.initial = initial
            self.goal = goal

        def actions(self, state):
            """Return the actions that can be executed in the given
          state. The result would typically be a list, but if there are
          many actions, consider yielding them one at a time in an
          iterator, rather than building them all at once."""
            return task.get_successor_ops(state)

        def result(self, state, action):
            """Return the state that results from executing the given
          action in the given state. The action must be one of
          self.actions(state)."""
            return state.apply(state)

        def goal_test(self, state):
            """Return True if the state is a goal. The default method compares the
          state to self.goal, as specified in the constructor. Override this
          method if checking against a single self.goal is not enough."""
            return state == self.goal

        def path_cost(self, c, state1, action, state2):
            """Return the cost of a solution path that arrives at state2 from
          state1 via action, assuming cost c to get up to state1. If the problem
          is such that the path doesn't matter, this function will only look at
          state2.  If the path does matter, it will consider c and maybe state1
          and action. The default method costs 1 for every step in the path."""
            return c + 1

        def value(self, state):
            """For optimization problems, each state has a value.  Hill-climbing
          and related algorithms try to maximize this value."""
            abstract


    # You should then set the variables:
    # final_state - the final state at the end of the plan
    final_state = []
    # plan - a list of actions representing the plan
    plan = []
    # cost - the cost of the plan
    cost = None
    # Your planner_v2 here










    toc = time.time()
    printOutputVerbose(tic, toc, plan, cost, final_state, task.goals)
