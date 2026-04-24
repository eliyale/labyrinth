import os
import time

from planning.astar import (
    Action,
    LabyrinthMap,
    State,
    cheap_heuristic,
    _format_action,
    DEMO_MAZE_STRING,
)
from planning.dstar import DStarLite

from labyrinth.model.game import BoardLocation, Game, Player, Turns


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def pause(seconds=1.0):
    time.sleep(seconds)


def print_state(title, state, labyrinth_map, step=None, action=None, expanded=None):
    clear_screen()

    print("=" * 70)
    print(title)
    print("=" * 70)

    if step is not None:
        print(f"Simulation step: {step}")

    player_loc = state.board.maze.maze_card_location(state.player.piece.maze_card)
    goal_loc = state.board.maze.maze_card_location(state.board.objective_maze_card)

    print(f"Player location: {player_loc}")
    print(f"Goal location:   {goal_loc}")

    if action is not None:
        print(f"Action: {_format_action(action)}")

    if expanded is not None:
        print(f"Expanded states during planning: {expanded}")

    print("-" * 70)
    state.board.pretty_print()
    print("-" * 70)


def choose_next_action(plan_result):
    result, visited = plan_result

    if result is None:
        return None, visited

    path, action_path = result

    if len(action_path) == 0:
        return None, visited

    return action_path[0], visited


def apply_adversary_shift(state, labyrinth_map):
    """
    Scripted environment change for the demo.

    This simulates an adversary changing the maze after the planner has
    already committed to part of a plan.
    """
    adversary_action = Action(
        shift_location=BoardLocation(1, 6),
        shift_rotation=90,
    )

    next_state = labyrinth_map.apply_action(state, adversary_action)

    return next_state, adversary_action


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

    planner = DStarLite(
        labyrinth_map=labyrinth_map,
        heuristic=cheap_heuristic,
        weight=1.0,
    )

    print_state(
        title="INITIAL LABYRINTH STATE",
        state=state,
        labyrinth_map=labyrinth_map,
    )
    pause(2)

    max_steps = 8

    for step in range(max_steps):
        print_state(
            title="PLANNING FROM CURRENT STATE",
            state=state,
            labyrinth_map=labyrinth_map,
            step=step,
        )
        pause(1)

        plan_result = planner.plan(state)
        next_action, visited = choose_next_action(plan_result)

        if next_action is None:
            print_state(
                title="NO VALID PLAN FOUND",
                state=state,
                labyrinth_map=labyrinth_map,
                step=step,
                expanded=len(visited),
            )
            break

        print_state(
            title="NEXT ACTION SELECTED BY PLANNER",
            state=state,
            labyrinth_map=labyrinth_map,
            step=step,
            action=next_action,
            expanded=len(visited),
        )
        pause(2)

        state = labyrinth_map.apply_action(state, next_action)

        print_state(
            title="STATE AFTER EXECUTING PLANNER ACTION",
            state=state,
            labyrinth_map=labyrinth_map,
            step=step,
            action=next_action,
        )
        pause(2)

        if labyrinth_map.is_goal(state):
            print_state(
                title="GOAL REACHED BY PLANNER",
                state=state,
                labyrinth_map=labyrinth_map,
                step=step,
                action=next_action,
            )
            break

        if step == 0:
            state, adversary_action = apply_adversary_shift(state, labyrinth_map)
            planner.notify_environment_change()

            print_state(
                title="ADVERSARY / ENVIRONMENT CHANGE: MAZE SHIFTED",
                state=state,
                labyrinth_map=labyrinth_map,
                step=step,
                action=adversary_action,
            )
            pause(2)

            print_state(
                title="D* LITE REPLANNING TRIGGERED",
                state=state,
                labyrinth_map=labyrinth_map,
                step=step,
            )
            pause(2)


if __name__ == "__main__":
    main()