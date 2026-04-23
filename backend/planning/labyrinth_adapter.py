from copy import deepcopy

from backend.labyrinth.model.reachable import Graph
from backend.labyrinth.model.game import BoardLocation
from backend.labyrinth.model import interactors
import backend.tests.unit.game_repository_mocks as game_repository_coach


def clone_game(game):
    return deepcopy(game)


def get_player_location(game, player_id=0):
    return game.board.maze.maze_card_location(
        game.get_player(player_id).piece.maze_card
    )


def get_goal_location(game):
    return game.board.maze.maze_card_location(
        game.board.objective_maze_card
    )


def is_goal_state(game, player_id=0):
    return get_player_location(game, player_id) == get_goal_location(game)


def get_move_actions(game, player_id=0):
    graph = Graph(game.board.maze)
    player_loc = get_player_location(game, player_id)

    reachable = graph.reachable_locations(player_loc)
    actions = []

    for loc in reachable:
        if loc != player_loc:
            actions.append(("move", loc))

    return actions


def get_shift_actions():
    edge_positions = [1, 3, 5]
    actions = []

    for c in edge_positions:
        actions.append(("shift", BoardLocation(0, c), 0))
        actions.append(("shift", BoardLocation(6, c), 0))

    for r in edge_positions:
        actions.append(("shift", BoardLocation(r, 0), 0))
        actions.append(("shift", BoardLocation(r, 6), 0))

    return actions


def get_actions(game, player_id=0):
    return get_move_actions(game, player_id) + get_shift_actions()


def apply_action(game, action, player_id=0):
    """
    This is the critical function:
    takes a game state + action → returns NEW game state
    """
    new_game = clone_game(game)

    game_repository = game_repository_coach.when_game_repository_find_by_id_then_return(new_game)
    interactor = interactors.PlayerActionInteractor(game_repository=game_repository)

    action_type = action[0]

    if action_type == "move":
        _, location = action
        interactor.perform_move(
            game_id=new_game.identifier,
            player_id=player_id,
            move_location=location
        )

    elif action_type == "shift":
        _, location, rotation = action
        interactor.perform_shift(
            game_id=new_game.identifier,
            player_id=player_id,
            shift_location=location,
            shift_rotation=rotation
        )

    return new_game