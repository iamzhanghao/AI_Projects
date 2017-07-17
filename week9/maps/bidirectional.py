from week9.maps.search import *

from week9.maps.highways import *
import time

# An undirected graph of highways in USA.  The cost is defined using
# the distance function from highways.py.  The neighbors dictionary is
# also defined in highways.py. Make sure that you have downloaded the
# maps.zip file and placed it in the same directory as this file.

# NOTE THAT CREATING THIS GRAPH TAKES A BIT OF TIME, so do it only
# once if already defined in the environment.
try:
    # See if it is defined
    usa
except:
    # If it isn't, evaluate it
    usa = UndirectedGraph({id1: {id2: distance(id1, id2) for id2 in neighbors[id1]} \
                           for id1 in neighbors})


class MyFIFOQueue(FIFOQueue):
    def getNode(self, state):
        '''Returns node in queue with matching state'''
        for i in range(self.start, len(self.A)):
            if self.A[i].state == state:
                return self.A[i]

    def __contains__(self, node):
        '''Returns boolean if there is node in queue with matching
        state.  The implementation in utils.py is very slow.'''
        for i in range(self.start, len(self.A)):
            if self.A[i].state == node.state:
                return True


def bidirectional_search(problem):
    '''
    Perform bidirectional search, both directions as breadth-first
    search, should return either the final (goal) node if a path is
    found or None if no path is found.
    '''
    assert problem.goal  # a fixed goal state

    # Below is the definition of BREADTH_FIRST_SEARCH from search.py.
    # You will need to (a) UNDERSTAND and (b) MODIFY this to do
    # bidirectional search.

    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = MyFIFOQueue()
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return child
                frontier.append(child)
    return None


# Modified from search.py
def compare_searchers(problems, header,
                      h=None,
                      searchers=[breadth_first_search]):
    def do(searcher, problem):
        p = InstrumentedProblem(problem)
        print('Starting', name(searcher))
        t0 = time.time()
        if name(searcher) in ('astar_search', 'greedy_best_first_graph_search'):
            searcher(p, h)
        else:
            searcher(p)
        t1 = time.time()
        print('Completed', name(searcher))
        return p, t1 - t0

    table = [[name(s)] + [do(s, p) for p in problems] for s in searchers]
    print_table(table, header)


def test_map():
    heuristic = lambda x: distance(x.state, 25000502)
    compare_searchers(problems=[GraphProblem(20000071, 25000502, usa)],
                      h=heuristic,
                      searchers=[breadth_first_search,
                                 bidirectional_search,
                                 uniform_cost_search,
                                 astar_search],
                      header=['Searcher', 'USA(Smith Center, Cambridge)'])


test_map()
