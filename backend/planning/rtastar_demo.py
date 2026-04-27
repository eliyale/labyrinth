"""
Demo for RTAStar on the Labyrinth project.

Copy this file to backend/planning/rtastar_demo.py and run from backend/:

    python -m planning.rtastar_demo

The planner searches DEPTH turns ahead, prints the returned partial plan,
executes only the first turn, then replans.
"""

import os
import time

from planning.astar import (
    Action,
    LabyrinthMap,
    State,
    _format_action,
    DEMO_MAZE_STRING,
)
from planning.rtastar import RTAStar, TurnAction

from labyrinth.model.game import BoardLocation, Game, Player, Turns


DEPTH = 1
MAX_STEPS = 8
GOAL_LOCATION = BoardLocation(2, 4)


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def pause(seconds=1.0):
    time.sleep(seconds)


def format_turn_action(action):
    if isinstance(action, TurnAction) or hasattr(action, "shift_action"):
        shift = _format_action(action.shift_action)
        move = "NO MOVE" if action.move_action is None else _format_action(action.move_action)
        return f"{shift} THEN {move}"

    return _format_action(action)


def print_state(title, state, step=None, action=None, expanded=None):
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
        print(f"Action: {format_turn_action(action)}")

    if expanded is not None:
        print(f"Expanded states during planning: {expanded}")

    print("-" * 70)
    state.board.pretty_print()
    print("-" * 70)


def choose_next_action(plan_result):
    result, visited = plan_result

    if result is None:
        return None, visited

    _path, action_path = result

    if len(action_path) == 0:
        return None, visited

    return action_path[0], visited


def apply_adversary_shift(state, labyrinth_map):
    adversary_action = Action(
        shift_location=BoardLocation(1, 6),
        shift_rotation=90,
    )

    next_state = labyrinth_map.apply_action(state, adversary_action)
    return next_state, adversary_action


def main():
    labyrinth_map = LabyrinthMap(
        maze_string=DEMO_MAZE_STRING,
        goal_location=GOAL_LOCATION,
    )

    game = Game(identifier=1, board=labyrinth_map.board, turns=Turns())
    player = Player(identifier=1)
    game.add_player(player)

    state = State(game.board, player)

    planner = RTAStar(
        labyrinth_map=labyrinth_map,
        depth=DEPTH,
        heuristic_weight=1.0,
        max_expansions=5000,
    )

    print_state("INITIAL LABYRINTH STATE", state)
    pause(1)

    for step in range(MAX_STEPS):
        print_state("PLANNING FROM CURRENT STATE", state, step=step)
        pause(0.5)

        print("LEFTOVER CARD:")
        print(state.board.leftover_card)
        print("rotation =", state.board.leftover_card.rotation)
        print(f"CALLING RTAStar.plan(state), depth={DEPTH}", flush=True)

        plan_result = planner.plan(state)

        print("FINISHED RTAStar.plan(state)", flush=True)

        result, visited = plan_result

        if result is not None:
            path, action_path = result
            print("\nRETURNED LOOKAHEAD PLAN:")
            for i, action in enumerate(action_path[:20]):
                print(i, format_turn_action(action))
            print(f"Total planned turns in lookahead: {len(action_path)}")

        next_action, visited = choose_next_action(plan_result)

        if next_action is None:
            if labyrinth_map.is_goal(state):
                print_state(
                    title="GOAL REACHED",
                    state=state,
                    step=step,
                    expanded=len(visited),
                )
            else:
                print_state(
                    title="NO VALID PLAN FOUND",
                    state=state,
                    step=step,
                    expanded=len(visited),
                )
            break

        print_state(
            title="NEXT TURN SELECTED BY RTA*",
            state=state,
            step=step,
            action=next_action,
            expanded=len(visited),
        )
        pause(0.5)

        state = planner.apply_turn(state, next_action)

        print_state(
            title="STATE AFTER EXECUTING FIRST TURN",
            state=state,
            step=step,
            action=next_action,
        )
        pause(0.5)

        if labyrinth_map.is_goal(state):
            print_state(
                title="GOAL REACHED BY RTA*",
                state=state,
                step=step,
                action=next_action,
            )
            break

        # potential adversary test.
        # if step == 0:
        #     state, adversary_action = apply_adversary_shift(state, labyrinth_map)
        #     print_state(
        #         title="ADVERSARY / ENVIRONMENT CHANGE: MAZE SHIFTED",
        #         state=state,
        #         step=step,
        #         action=adversary_action,
        #     )
        #     pause(0.5)


if __name__ == "__main__":
    main()
