'''
A* search algorithm for finding the shortest path in a graph.

states are a State objects with attributes (Board, Player, goal_count, map_ref)
    Board is a Board object.
    Player is a Player object.
    goal_count is an integer representing the number of goals visited.
    map_ref is a reference to the LabyrinthMap object to avoid deepcopying the board object in transition function.

Action:
    One full player turn, represented as a combined SHIFT+MOVE:
      - shift_location: BoardLocation
      - shift_rotation: int in {0, 90, 180, 270}
      - move_location: BoardLocation reachable after that shift

The transition function applies both primitives in sequence on a copied board.
'''

import argparse
import heapq
import json
import pathlib
import sys
import time

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from labyrinth.model.factories import MazeCardFactory, create_maze
from labyrinth.model.game import Board, BoardLocation, Game, Maze, MazeCard, Piece, Player, Turns
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

    def is_turn(self):
        return self.is_shift() and self.is_move()

class State:
    def __init__(self, board: Board, player: Player, goal_count: int = 0, map_ref=None):
        self.board = board
        self.player = player
        self.goal_count = goal_count
        self.map_ref = map_ref

    def __eq__(self, other):
        return (
            self.board == other.board
            and self.player == other.player
            and self.goal_count == other.goal_count
        )

    def __hash__(self):
        return hash((self.board, self.player, self.goal_count))

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
    """
    A utility class for managing the labyrinth board and state. Provides metnods
    used by astar graph search.

    maze_string: str - the string representation of the maze
    goal_list: list[BoardLocation] - initial locations of ordered goal cards
    """
    def __init__(self, maze_string: str, goal_list: list[BoardLocation]):
        self.maze = maze_string
        self.goal_list = goal_list
        self.goal_count = len(goal_list)
        maze_card_factory = MazeCardFactory()
        maze = create_maze(maze_string, maze_card_factory)
        # Track goal cards by object identity so shifts move goals correctly.
        self.goal_cards = [maze[goal_location] for goal_location in goal_list]
        objective_maze_card = self.goal_cards[0]
        # initialize the board with the first goal location as the objective
        self.board = Board(maze, leftover_card=maze_card_factory.create_instance(MazeCard.STRAIGHT, 0), objective_maze_card=objective_maze_card)

    def _board_of(self, state):
        return state.board if isinstance(state, State) else state[0]

    def _player_of(self, state):
        return state.player if isinstance(state, State) else state[1]

    def _make_state(self, board, player, goal_count, keep_wrapper=False, map_ref=None):
        return State(board, player, goal_count, map_ref=map_ref) if keep_wrapper else (board, player)

    def _clone_state_fast(self, state):
        """Clone planner state without using deepcopy for speed."""
        keep_wrapper = isinstance(state, State)
        board = self._board_of(state)
        player = self._player_of(state)
        goal_count = state.goal_count if keep_wrapper else 0

        cloned_maze = Maze(board.maze.maze_size)
        cards_by_id = {}
        for location in board.maze.maze_locations:
            card = board.maze[location]
            cloned = MazeCard(card.identifier, card.out_paths, card.rotation)
            cloned_maze[location] = cloned
            cards_by_id[card.identifier] = cloned

        leftover = board.leftover_card
        cloned_leftover = MazeCard(leftover.identifier, leftover.out_paths, leftover.rotation)
        cards_by_id[cloned_leftover.identifier] = cloned_leftover

        objective_id = board.objective_maze_card.identifier
        cloned_objective = cards_by_id[objective_id]
        cloned_board = Board(cloned_maze, leftover_card=cloned_leftover, objective_maze_card=cloned_objective)

        piece = player.piece
        piece_card = cards_by_id[piece.maze_card.identifier]
        cloned_piece = Piece(piece.piece_index, piece_card)
        cloned_board._pieces = [cloned_piece]
        cloned_player = Player(player.identifier, piece=cloned_piece, player_name=player.player_name)
        cloned_player.score = player.score

        return self._make_state(cloned_board, cloned_player, goal_count, keep_wrapper=keep_wrapper, map_ref=self)

    def _set_objective_from_goal_count(self, board: Board, goal_count: int):
        """
        Force board objective to the next ordered target from goal_list.
        """
        if goal_count < self.goal_count:
            board._objective_maze_card = self.goal_cards[goal_count]

    def is_goal(self, state):
        # board = self._board_of(state)
        # player = self._player_of(state)
        # goal_location = self.board.maze.maze_card_location(self.board.objective_maze_card)
        # player_location = board.maze.maze_card_location(player.piece.maze_card)
        # return goal_location == player_location
        return state.goal_count == self.goal_count

    def apply_action(self, state, action):
        # Clone current state to avoid mutating search tree ancestors.
        copied_state = self._clone_state_fast(state)
        keep_wrapper = isinstance(copied_state, State)
        board = self._board_of(copied_state)
        player = self._player_of(copied_state)
        goal_count = copied_state.goal_count if keep_wrapper else 0

        # Keep board objective aligned with ordered-goal progress.
        self._set_objective_from_goal_count(board, goal_count)
        try:
            if not action.is_turn():
                return copied_state
            board.shift(action.shift_location, action.shift_rotation)
            reached_goal = board.move(player.piece, action.move_location)
            if reached_goal and goal_count < self.goal_count:
                goal_count += 1
                # Override board's internal random next objective with ordered target.
                self._set_objective_from_goal_count(board, goal_count)
        except Exception:
            # Invalid actions are treated as no-op transitions by the planner.
            return copied_state

        return self._make_state(board, player, goal_count, keep_wrapper=keep_wrapper, map_ref=self)

    def move_cost(self, state: State, action: Action):
        if action.is_turn():
            return 1
        return 0

    def get_actions(self, state: State):
        actions = []
        board = self._board_of(state)
        for shift_location in board.shift_locations:
            for shift_rotation in (0, 90, 180, 270):
                # Enumerate reachable moves on the post-shift board to build full-turn actions.
                shifted_state = self._clone_state_fast(state)
                shifted = self._board_of(shifted_state)
                shifted_player = self._player_of(shifted_state)
                try:
                    shifted.shift(shift_location, shift_rotation)
                except Exception:
                    continue
                piece_location = shifted.maze.maze_card_location(shifted_player.piece.maze_card)
                if piece_location is None:
                    continue
                for move_location in Graph(shifted.maze).reachable_locations(piece_location):
                    actions.append(
                        Action(
                            shift_location=shift_location,
                            shift_rotation=shift_rotation,
                            move_location=move_location,
                        )
                    )

        return actions

    def neighbor_data(self, state: State):
        for action in self.get_actions(state):
            yield self.apply_action(state, action), action, self.move_cost(state, action)

    def display_map(self, path=[], visited=set(), filename=None):
        """
        Visualize the map. Optionally display the resulting plan and visited nodes.
        """
        self.board.pretty_print()

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
    # For ordered-goal planning, compare against the currently active target.
    if isinstance(state, State) and hasattr(state, "goal_count") and hasattr(state, "map_ref"):
        if state.goal_count >= state.map_ref.goal_count:
            return 0
        goal_card = state.map_ref.goal_cards[state.goal_count]
        try:
            goal_location = board.maze.maze_card_location(goal_card)
        except Exception:
            # Goal card may be the current leftover card; treat as not directly reachable.
            return 2
    else:
        goal_location = board.maze.maze_card_location(board.objective_maze_card)

    player_location = board.maze.maze_card_location(player.piece.maze_card)
    if player_location == goal_location:
        return 0

    if Graph(board.maze).is_reachable(player_location, goal_location):
        return 1

    return 2


def one_turn_heuristic(state):
    """
    Turn-aware admissible heuristic for coupled (SHIFT+MOVE) actions.

      - 0 if already on current ordered goal
      - 1 if goal can be reached in one legal turn
      - 2 otherwise
    """
    board = state.board if isinstance(state, State) else state[0]
    player = state.player if isinstance(state, State) else state[1]

    # Ordered-goal context (primary path in this planner).
    if isinstance(state, State) and hasattr(state, "goal_count") and hasattr(state, "map_ref"):
        if state.goal_count >= state.map_ref.goal_count:
            return 0
        labyrinth_map = state.map_ref
        goal_card = labyrinth_map.goal_cards[state.goal_count]
    else:
        # Fallback for non-ordered contexts.
        return cheap_heuristic(state)

    piece_location = board.maze.maze_card_location(player.piece.maze_card)
    goal_location = board.maze.maze_card_location(goal_card)
    if goal_location is not None and piece_location == goal_location:
        return 0

    # Exact one-turn reachability test.
    for shift_location in board.shift_locations:
        for shift_rotation in (0, 90, 180, 270):
            shifted_state = labyrinth_map._clone_state_fast(state)
            shifted_board = shifted_state.board
            shifted_player = shifted_state.player
            try:
                shifted_board.shift(shift_location, shift_rotation)
            except Exception:
                continue

            shifted_piece_location = shifted_board.maze.maze_card_location(shifted_player.piece.maze_card)
            shifted_goal_location = shifted_board.maze.maze_card_location(goal_card)
            if shifted_goal_location is None or shifted_piece_location is None:
                continue
            if shifted_piece_location == shifted_goal_location:
                return 1
            if Graph(shifted_board.maze).is_reachable(shifted_piece_location, shifted_goal_location):
                return 1

    return 2


def manhattan_heuristic(state):
    """
    Spatial heuristic: Manhattan distance from player tile to current goal tile.

    Note: with Labyrinth shifts this is generally not admissible for turn-cost
    planning, but it is very cheap and can speed up plan discovery.
    """
    board = state.board if isinstance(state, State) else state[0]
    player = state.player if isinstance(state, State) else state[1]
    if isinstance(state, State) and hasattr(state, "goal_count") and hasattr(state, "map_ref"):
        if state.goal_count >= state.map_ref.goal_count:
            return 0
        goal_card = state.map_ref.goal_cards[state.goal_count]
    else:
        goal_card = board.objective_maze_card

    player_location = board.maze.maze_card_location(player.piece.maze_card)
    goal_location = board.maze.maze_card_location(goal_card)
    if player_location is None or goal_location is None:
        return 0
    return abs(player_location.row - goal_location.row) + abs(player_location.column - goal_location.column)


def make_hybrid_heuristic(one_turn_budget: int = 1500):
    """
    Cached hybrid heuristic for better wall-clock performance.

    Strategy:
      1) compute cheap_heuristic first
      2) if cheap result is already informative (0 or 1), use it
      3) only for ambiguous states (cheap==2), run one_turn_heuristic while budget remains
      4) memoize per-state to avoid recomputation
    """
    cache = {}
    budget = {"remaining": int(one_turn_budget)}

    def _heuristic(state):
        if state in cache:
            return cache[state]

        base = cheap_heuristic(state)
        if base < 2:
            cache[state] = base
            return base

        if budget["remaining"] > 0:
            budget["remaining"] -= 1
            value = one_turn_heuristic(state)
        else:
            value = base

        cache[state] = value
        return value

    # attach debug fields for optional printing from main()
    _heuristic._cache = cache
    _heuristic._budget = budget
    return _heuristic


def _format_action(action):
    if action.is_turn():
        return (
            f"SHIFT at {action.shift_location} rot={action.shift_rotation} "
            f"-> MOVE to {action.move_location}"
        )
    if action.is_shift():
        return f"SHIFT at {action.shift_location} rot={action.shift_rotation} (incomplete)"
    if action.is_move():
        return f"MOVE to {action.move_location} (incomplete)"
    return "UNKNOWN ACTION"


def _action_to_plan_steps(action: Action):
    """Convert one combined turn action into API replay steps."""
    if not action.is_turn():
        return []
    return [
        {
            "type": "shift",
            "row": action.shift_location.row,
            "column": action.shift_location.column,
            "leftoverRotation": action.shift_rotation,
        },
        {
            "type": "move",
            "row": action.move_location.row,
            "column": action.move_location.column,
        },
    ]


def _export_plan_json(
    maze_string: str,
    action_path,
    output_name: str = "astar_generated_plan.json",
    base_url: str = "http://127.0.0.1",
    game_id: int = 0,
    player_id: int = 1,
):
    plan_dir = pathlib.Path(__file__).resolve().parent / "plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    output_path = plan_dir / output_name

    steps = []
    for action in action_path:
        steps.extend(_action_to_plan_steps(action))

    plan = {
        "baseUrl": base_url,
        "gameId": game_id,
        "playerId": player_id,
        "mazeString": maze_string.strip(),
        "steps": steps,
    }
    output_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Run A* planner demo and export a plan JSON.")
    parser.add_argument(
        "--heuristic",
        choices=("zero", "cheap", "one-turn", "hybrid", "manhattan"),
        default="hybrid",
        help="Heuristic used by A* (default: hybrid).",
    )
    parser.add_argument(
        "--hybrid-budget",
        type=int,
        default=150,
        help="Number of one-turn checks allowed by hybrid heuristic.",
    )
    args = parser.parse_args()

    # Build the search map and objective.
    goal_list = [BoardLocation(2, 0), BoardLocation(0, 2)]
    goal_list = [BoardLocation(0, 3), BoardLocation(2, 3)]
    goal_list = [BoardLocation(0, 4), BoardLocation(2, 6)]
    goal_list = [BoardLocation(2, 0), BoardLocation(0,2), BoardLocation(1, 1)]
    labyrinth_map = LabyrinthMap(DEMO_MAZE_STRING, goal_list)

    # Create a game wrapper so the player gets a valid piece placed on the board.
    game = Game(identifier=1, board=labyrinth_map.board, turns=Turns())
    player = Player(identifier=1)
    game.add_player(player)

    init_state = State(game.board, player, map_ref=labyrinth_map)
    if args.heuristic == "zero":
        heuristic = _zero_heuristic
    elif args.heuristic == "cheap":
        heuristic = cheap_heuristic
    elif args.heuristic == "one-turn":
        heuristic = one_turn_heuristic
    elif args.heuristic == "manhattan":
        heuristic = manhattan_heuristic
    else:
        heuristic = make_hybrid_heuristic(one_turn_budget=args.hybrid_budget)

    print(f"Heuristic: {args.heuristic}")
    t_search_start = time.perf_counter()
    result, visited = a_star_search(
        init_state=init_state,
        f=labyrinth_map.apply_action,
        is_goal=labyrinth_map.is_goal,
        actions=labyrinth_map.get_actions,
        h=heuristic,
        weight=1.0,
    )
    search_seconds = time.perf_counter() - t_search_start
    print(f"A* search: {search_seconds:.3f}s ({len(visited)} states expanded)")
    if args.heuristic == "hybrid":
        print(
            "Heuristic cache: "
            f"{len(heuristic._cache)} states, one-turn calls used="
            f"{args.hybrid_budget - heuristic._budget['remaining']}"
        )

    if result is None:
        print("No path found.")
        return

    path, action_path = result
    print("Goal Location:", goal_list)
    print("Player Location:", game.board.maze.maze_card_location(player.piece.maze_card))
    print(f"Path found. Expanded {len(visited)} states.")
    print(f"Plan length: {len(action_path)} actions, {len(path)} states.")
    for index, action in enumerate(action_path, start=1):
        print(f"{index:03d}. {_format_action(action)}")

    exported_path = _export_plan_json(DEMO_MAZE_STRING, action_path)
    print(f"Exported plan JSON: {exported_path}")


if __name__ == "__main__":
    main()