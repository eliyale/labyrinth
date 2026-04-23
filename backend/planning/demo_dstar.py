from backend.planning.dstar import DStarLite
from backend.planning.labyrinth_adapter import (
    get_player_location,
    get_goal_location,
    is_goal_state,
    get_actions,
    apply_action,
)

from backend.tests.unit.factories import MazeCardFactory
from backend.labyrinth.model.factories import create_maze
from backend.labyrinth.model.game import Board, Game, Turns, Player, MazeCard, BoardLocation

MAZE_STRING = """
###|#.#|#.#|###|#.#|#.#|###|
#..|#..|...|...|#..|..#|..#|
#.#|###|###|#.#|###|###|#.#|
---------------------------|
###|###|#.#|#.#|#.#|#.#|#.#|
...|...|#.#|#..|#.#|...|..#|
#.#|#.#|#.#|###|#.#|###|#.#|
---------------------------|
#.#|#.#|#.#|#.#|#.#|#.#|#.#|
#..|#..|..#|#..|..#|#.#|..#|
#.#|#.#|#.#|#.#|#.#|#.#|#.#|
---------------------------|
#.#|#.#|#.#|###|#.#|###|###|
..#|..#|#..|...|...|...|..#|
###|#.#|###|#.#|###|#.#|#.#|
---------------------------|
###|#.#|###|#.#|###|#.#|###|
#..|..#|#..|#.#|...|#..|...|
#.#|###|#.#|#.#|#.#|###|###|
---------------------------|
###|#.#|###|#.#|#.#|#.#|#.#|
..#|#..|...|...|#.#|#..|..#|
#.#|#.#|###|###|#.#|#.#|#.#|
---------------------------|
#.#|#.#|###|###|#.#|#.#|#.#|
#..|...|...|...|#.#|...|..#|
###|###|#.#|###|#.#|###|###|
---------------------------*
"""


def make_demo_game():
    maze_card_factory = MazeCardFactory()
    maze = create_maze(MAZE_STRING, maze_card_factory)

    objective = maze[BoardLocation(1, 1)]
    board = Board(
        maze,
        leftover_card=maze_card_factory.create_instance(MazeCard.STRAIGHT, 0),
        objective_maze_card=objective,
    )

    game = Game(identifier=7, board=board, turns=Turns())
    player = Player(identifier=0, game=0)
    game.add_player(player)

    return game


def heuristic(game):
    player_loc = get_player_location(game, player_id=0)
    goal_loc = get_goal_location(game)

    # simple Manhattan distance on board coordinates
    return abs(player_loc.row - goal_loc.row) + abs(player_loc.column - goal_loc.column)


def transition(game, action):
    return apply_action(game, action, player_id=0)


def goal_check(game):
    return is_goal_state(game, player_id=0)


def choose_next_action(plan_result):
    """
    plan_result from A*/current D* baseline is:
    ((path, action_path), visited) or (None, visited)
    """
    result, visited = plan_result

    if result is None:
        return None

    path, action_path = result
    if not action_path:
        return None

    return action_path[0]


def main():
    game = make_demo_game()

    print("=== INITIAL STATE ===")
    print("Player:", get_player_location(game, 0))
    print("Goal:", get_goal_location(game))
    print(game.board.pretty_print())

    planner = DStarLite(
        f=transition,
        is_goal=goal_check,
        actions=get_actions(game, player_id=0),
        h=heuristic,
    )

    max_steps = 5

    for step in range(max_steps):
        print(f"\n=== STEP {step} ===")

        # refresh actions each time because the board changes
        planner.actions = get_actions(game, player_id=0)

        plan_result = planner.plan(game)
        next_action = choose_next_action(plan_result)

        print("Current player location:", get_player_location(game, 0))
        print("Current goal location:", get_goal_location(game))
        print("Chosen action:", next_action)

        if next_action is None:
            print("No action available. Stopping demo.")
            break

        game = transition(game, next_action)

        print("Board after action:")
        print(game.board.pretty_print())

        if goal_check(game):
            print("Goal reached!")
            break


if __name__ == "__main__":
    main()