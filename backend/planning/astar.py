'''
A* search algorithm for finding the shortest path in a graph.
'''

import heapq

class SearchNode:
    def __init__(self, s, A=None, parent=None, parent_action=None, cost=0):
        """
        s - the state defining the search node
        cost - cost to reach this node, i.e. cost to come

        Extra parameters are accepted for compatibility but are not stored.
        """
        self.cost = cost
        self.A = A
        self.parent = parent
        self.parent_action = parent_action
        self.state = s if isinstance(s, tuple) else tuple(s)

    def __lt__(self, other):
        return self.cost < other.cost


class PriorityQ:
    """
    Priority queue implementation with quick access for membership testing.
    Setup to work with SearchNode class.
    """

    def __init__(self):
        """
        Initialize an empty priority queue
        """
        self.l = []  # list storing the priority q
        self.s = set()  # set for fast membership testing

    def __contains__(self, x):
        """
        Test if x is in the queue
        """
        return x in self.s

    def push(self, x, cost):
        """
        Adds an element to the priority queue.
        If the state already exists, we update the cost
        """
        if x.state in self.s:
            return self.replace(x, cost)
        heapq.heappush(self.l, (cost, x))
        self.s.add(x.state)

    def pop(self):
        """
        Get the value and remove the lowest cost element from the queue
        """
        x = heapq.heappop(self.l)
        self.s.remove(x[1].state)
        return x[1]

    def peek_key(self):
        """
        Get the lowest key in the priority queue.
        """
        return self.l[0][0]

    def __len__(self):
        """
        Return the number of elements in the queue
        """
        return len(self.l)

    def replace(self, x, new_cost):
        """
        Removes element x from the q and replaces it with x with the new_cost
        """
        for y in self.l:
            if x.state == y[1].state:
                self.l.remove(y)
                self.s.remove(y[1].state)
                break
        heapq.heapify(self.l)
        self.push(x, new_cost)

    def remove_state(self, s):
        """
        Remove a state from the queue if present.

        returns - True if removed, False otherwise
        """
        if s not in self.s:
            return False
        for item in list(self.l):
            if item[1].state == s:
                self.l.remove(item)
                self.s.remove(s)
                heapq.heapify(self.l)
                return True
        return False
    
    def get_cost(self, x):
        """
        Return the cost for the search node with state x.state
        """
        for y in self.l:
            if x.state == y[1].state:
                return y[0]

class LabyrinthMap:
    def __init__(self, maze):
        self.maze = maze

    def is_goal(self, state):
        return state == self.maze.goal

    def get_actions(self, state):
        return self.maze.get_actions(state)

    def get_heuristic(self, state):
        return self.maze.get_heuristic(state)

    def get_transition_function(self, state, action):
        return self.maze.get_transition_function(state, action)

    def get_cost(self, state, action):
        return self.maze.get_cost(state, action)


def a_star_search(init_state, f, is_goal, actions, h, weight=1.0):
    """
    init_state - value of the initial state
    f - transition function takes input state (s), action (a), returns s_prime = f(s, a)
        returns s if action is not valid
    is_goal - takes state as input returns true if it is a goal state
        actions - list of actions available
    h - heuristic function, takes input s and returns estimated cost to goal
        (note h will also need access to the map, so should be a member function of GridMap)
    weight - weight value for weighted A* (default 1.0 i.e. A*)

    returns - ((path, action_path), visited) or (None, visited) if no path can be found
    path - a list of tuples. The first element is the initial state followed by all states
        traversed until the final goal state
    action_path - the actions taken to transition from the initial state to goal, i.e. the plan
    """
    frontier = PriorityQ()
    n0 = SearchNode(init_state, actions)
    n0.cost = 0
    frontier.push(n0, n0.cost + weight * h(init_state))
    visited = set()

    while len(frontier) > 0:
        n_i = frontier.pop()

        if n_i.state in visited:
            continue
        visited.add(n_i.state)

        if is_goal(n_i.state):
            return (backpath(n_i), visited)

        for a in actions:
            s_prime = f(n_i.state, a)
            if s_prime == n_i.state:
                continue

            # TODO: add a cost for the action as _ACTION_COSTS[]
            step_cost = _ACTION_COSTS[a]

            n_prime = SearchNode(s_prime, actions, n_i, a)
            n_prime.cost = n_i.cost + step_cost

            # Only push if not in frontier or new cost is lower
            if s_prime not in frontier or n_prime.cost < frontier.get_cost(n_prime):
                priority = n_prime.cost + weight * h(s_prime)
                frontier.push(n_prime, priority)

    return (None, visited)