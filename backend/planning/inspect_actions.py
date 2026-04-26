from planning.astar import (
    Action,
    LabyrinthMap,
    State,
    cheap_heuristic,
    _format_action,
    DEMO_MAZE_STRING,
)

from labyrinth.model.game import BoardLocation, Game, Player, Turns


def main():
    goal_location = BoardLocation(2, 4)

    labyrinth_map = LabyrinthMap(
        maze_string=DEMO_MAZE_STRING,
        goal_location=goal_location,
    )

    game = Game(identifier=1, board=labyrinth_map.board, turns=Turns())
    player = Player(identifier=1)
    game.add_player(player)

    state = State(game.board, player)

    print("INITIAL STATE")
    state.board.pretty_print()

    actions = list(labyrinth_map.get_actions(state))

    print(f"\nNumber of actions: {len(actions)}\n")

    for i, action in enumerate(actions):
        print(i, _format_action(action))


if __name__ == "__main__":
    main()