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
    usa
except:
    # If it isn't, evaluate it
    usa = UndirectedGraph({id1: {id2: distance(id1, id2) for id2 in neighbors[id1]} \
                           for id1 in neighbors})


class MyFIFOQueue(FIFOQueue):
    def getNode(self, state):
        """Returns node in queue with matching state"""
        for i in range(self.start, len(self.A)):
            if self.A[i].state == state:
                return self.A[i]

    def __contains__(self, node):
        """Returns boolean if there is node in queue with matching
        state.  The implementation in utils.py is very slow."""
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

    """Forward Search"""
    forward_node = Node(problem.initial)
    backward_node = Node(problem.goal)
    if problem.goal_test(forward_node.state):
        return forward_node
    else:
        forward_frontier = MyFIFOQueue()
        forward_frontier.append(forward_node)
        forward_explored = set()

        backward_frontier = MyFIFOQueue()
        backward_frontier.append(backward_node)
        backward_explored = set()

        while forward_frontier and backward_frontier:
            forward_node = forward_frontier.pop()
            backward_node = backward_frontier.pop()

            forward_explored.add(forward_node.state)
            backward_explored.add(backward_node.state)

            for child in forward_node.expand(problem):
                if child.state not in forward_explored and child not in forward_frontier:
                    forward_frontier.append(child)

            for child in backward_node.expand(problem):
                if child.state not in backward_explored and child not in backward_frontier:
                    backward_frontier.append(child)

            common = set.intersection(forward_explored, backward_explored)
            if len(common) != 0:
                mid = breadth_first_search(GraphProblem(forward_node.state, backward_node.state, usa))
                ans = []
                for node in forward_node.path():
                    ans.append(node.state)
                for node in mid.path():
                    ans.append(node.state)
                for node in backward_node.path()[::-1]:
                    ans.append(node.state)
                return ans

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


# As a test, we will use uniform_cost_search to find a path from a place near the geographic center of the U.S. (
# Smith Center, KS; ID number 20000071), to Cambridge (ID number 25000502).

problem = GraphProblem(20000071, 25000502, usa)


def compare():
    heuristic = lambda x: distance(x.state, 25000502)
    compare_searchers(problems=[problem],
                      h=heuristic,
                      searchers=[breadth_first_search,
                                 bidirectional_search,
                                 uniform_cost_search,
                                 astar_search],
                      header=['Searcher', 'USA(Smith Center, Cambridge)'])


# you can create an instance of GraphProblem class from the UndirectedGraph named usa, and additionally startpoint
# and endpoint. GraphProblem is a derived class of the class Problem, and you can use any search algorithm to run it.




# map()
ans = bidirectional_search(problem)
path_objs = ans.path()
path = []
for path_obj in path_objs:
    path.append(path_obj.state)
print(path)
to_kml(path)
