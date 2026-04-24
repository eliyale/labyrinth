'''
A* search algorithm for finding the shortest path in a graph.

states are (Board, Player) tuples
    Board is a Board object.
    Player is a Player object.

The action is selected from [Shift or Move] where
    Shift is a tuple of (shift_location, shift_rotation)
        shift_location is a BoardLocation object
        shift_rotation is an integer from [0, 90, 180, 270]
    Move is BoardLocation object.

'''

import copy
import heapq
import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from labyrinth.model.factories import MazeCardFactory, create_maze
from labyrinth.model.game import Board, BoardLocation, Game, MazeCard, Player, Turns
from labyrinth.model.reachable import Graph

class Action:
    def __init__(self, shift_location: BoardLocation = None, shift_rotation: int = None, move_location: BoardLocation = None):
        self.shift_location = shift_location
        self.shift_rotation = shift_rotation
        self.move_location = move_location

    def is_shift(self):
        return self.shift_location is not None

    def is_move(self):
        return self.move_location is not None

class State:
    def __init__(self, board: Board, player: Player):
        self.board = board
        self.player = player

    def __eq__(self, other):
        return self.board == other.board and self.player == other.player

    def __hash__(self):
        return hash((self.board, self.player))

    def __str__(self):
        return f"State(board={self.board}, player={self.player})"

    def __repr__(self):
        return self.__str__()

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
        self.state = s

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
    def __init__(self, maze_string: str, goal_location: BoardLocation):
        self.maze = maze_string
        maze_card_factory = MazeCardFactory()
        maze = create_maze(maze_string, maze_card_factory)
        objective_maze_card = maze[goal_location]
        self.board = Board(maze, leftover_card=maze_card_factory.create_instance(MazeCard.STRAIGHT, 0), objective_maze_card=objective_maze_card)

    def _board_of(self, state):
        return state.board if isinstance(state, State) else state[0]

    def _player_of(self, state):
        return state.player if isinstance(state, State) else state[1]

    def _make_state(self, board, player, keep_wrapper=False):
        return State(board, player) if keep_wrapper else (board, player)

    def is_goal(self, state):
        board = self._board_of(state)
        player = self._player_of(state)

        goal_location = board.maze.maze_card_location(board.objective_maze_card)
        player_location = board.maze.maze_card_location(player.piece.maze_card)

        return goal_location == player_location

    def apply_action(self, state, action):
        # Clone current state to avoid mutating search tree ancestors.
        keep_wrapper = isinstance(state, State)
        copied_state = copy.deepcopy(state)
        board = self._board_of(copied_state)
        player = self._player_of(copied_state)
        try:
            if action.is_shift():
                board.shift(action.shift_location, action.shift_rotation)
            elif action.is_move():
                board.move(player.piece, action.move_location)
        except Exception:
            # Invalid actions are treated as no-op transitions by the planner.
            return copied_state
        return self._make_state(board, player, keep_wrapper=keep_wrapper)

    def move_cost(self, state: State, action: Action):
        if action.is_shift():
            return 1
        elif action.is_move():
            return 1
        return 0

    def get_actions(self, state: State):
        actions = []
        board = self._board_of(state)
        player = self._player_of(state)

        for shift_location in board.shift_locations:
            for shift_rotation in (0, 90, 180, 270):
                actions.append(Action(shift_location=shift_location, shift_rotation=shift_rotation))

        piece_location = board.maze.maze_card_location(player.piece.maze_card)
        for move_location in Graph(board.maze).reachable_locations(piece_location):
            actions.append(Action(move_location=move_location))

        return actions

    def neighbor_data(self, state: State):
        for action in self.get_actions(state):
            yield self.apply_action(state, action), action, self.move_cost(state, action)

    @classmethod
    def from_existing_board(cls):
        """
        Create a LabyrinthMap wrapper for an existing board/game state.

        This is used by the Docker/browser demo, where the board already exists
        in the backend database.
        """
        obj = cls.__new__(cls)
        obj.maze = None
        obj.board = None
        return obj

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
    best_cost = {init_state: 0}

    while len(frontier) > 0:
        n_i = frontier.pop()

        if n_i.state in visited:
            continue
        visited.add(n_i.state)

        if is_goal(n_i.state):
            return (backpath(n_i), visited)

        available_actions = actions(n_i.state) if callable(actions) else actions
        for a in available_actions:
            s_prime = f(n_i.state, a)
            if s_prime == n_i.state:
                continue

            step_cost = 1
            tentative_cost = n_i.cost + step_cost
            if tentative_cost >= best_cost.get(s_prime, float("inf")):
                continue
            best_cost[s_prime] = tentative_cost

            n_prime = SearchNode(s_prime, actions, n_i, a, tentative_cost)
            priority = tentative_cost + weight * h(s_prime)
            frontier.push(n_prime, priority)

    return (None, visited)


def backpath(node):
    path = []
    action_path = []
    current = node
    while current is not None:
        path.append(current.state)
        if current.parent_action is not None:
            action_path.append(current.parent_action)
        current = current.parent
    path.reverse()
    action_path.reverse()
    return path, action_path


DEMO_MAZE_STRING = """
###|#.#|###|###|###|#.#|###|
#..|#..|...|...|...|..#|..#|
#.#|###|#.#|#.#|#.#|###|#.#|
---------------------------|
###|###|#.#|#.#|#.#|###|#.#|
...|...|#.#|#..|#.#|...|#.#|
#.#|#.#|#.#|###|#.#|###|#.#|
---------------------------|
#.#|#.#|#.#|#.#|###|#.#|#.#|
#..|#..|#..|#..|...|#.#|..#|
#.#|###|#.#|###|#.#|#.#|#.#|
---------------------------|
#.#|#.#|#.#|###|#.#|###|###|
..#|..#|#..|...|#..|...|..#|
###|#.#|###|###|###|#.#|#.#|
---------------------------|
#.#|#.#|#.#|#.#|#.#|#.#|#.#|
#..|..#|...|#.#|..#|#..|..#|
#.#|###|###|#.#|#.#|###|#.#|
---------------------------|
###|#.#|###|###|#.#|###|#.#|
..#|#..|...|...|#.#|#..|..#|
#.#|###|###|###|#.#|#.#|#.#|
---------------------------|
#.#|###|#.#|#.#|#.#|#.#|#.#|
#..|...|...|#..|...|#..|..#|
###|###|###|###|###|###|###|
---------------------------*
"""


def _zero_heuristic(_state):
    return 0


def cheap_heuristic(state):
    """
    Very cheap heuristic:
      - 0 if already on the goal
      - 1 if goal is reachable by a single MOVE action on the current board
      - 2 otherwise

    Note: this is intentionally lightweight. It is not guaranteed admissible in all
    Labyrinth dynamics (a single SHIFT can sometimes place the player on goal).
    """
    board = state.board if isinstance(state, State) else state[0]
    player = state.player if isinstance(state, State) else state[1]

    player_location = board.maze.maze_card_location(player.piece.maze_card)
    goal_location = board.maze.maze_card_location(board.objective_maze_card)
    if player_location == goal_location:
        return 0

    if Graph(board.maze).is_reachable(player_location, goal_location):
        return 1

    return 2


def _format_action(action):
    if action.is_shift():
        return f"SHIFT at {action.shift_location} rot={action.shift_rotation}"
    if action.is_move():
        return f"MOVE to {action.move_location}"
    return "UNKNOWN ACTION"


def main():
    # Build the search map and objective.
    goal_location = BoardLocation(1, 1)
    goal_location = BoardLocation(0, 3)
    goal_location = BoardLocation(2, 4)
    labyrinth_map = LabyrinthMap(DEMO_MAZE_STRING, goal_location)

    # Create a game wrapper so the player gets a valid piece placed on the board.
    game = Game(identifier=1, board=labyrinth_map.board, turns=Turns())
    player = Player(identifier=1)
    game.add_player(player)

    init_state = State(game.board, player)
    result, visited = a_star_search(
        init_state=init_state,
        f=labyrinth_map.apply_action,
        is_goal=labyrinth_map.is_goal,
        actions=labyrinth_map.get_actions,
        h=cheap_heuristic,
        weight=1.0,
    )

    if result is None:
        print(f"No path found. Expanded {len(visited)} states.")
        return

    path, action_path = result
    print("Goal Location:", goal_location)
    print("Player Location:", game.board.maze.maze_card_location(player.piece.maze_card))
    print(f"Path found. Expanded {len(visited)} states.")
    print(f"Plan length: {len(action_path)} actions, {len(path)} states.")
    for index, action in enumerate(action_path, start=1):
        print(f"{index:03d}. {_format_action(action)}")


if __name__ == "__main__":
    main()