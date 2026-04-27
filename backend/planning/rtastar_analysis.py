import csv
import os
import time

from planning.astar import LabyrinthMap, State, _format_action
from planning.rtastar import RTAStar, TurnAction

from labyrinth.model.game import BoardLocation, Game, Player, Turns


CARD_PATTERNS = {
    "V": ["#.#", "#.#", "#.#"],
    "H": ["###", "...", "###"],
    "NE": ["#.#", "#..", "###"],
    "ES": ["###", "#..", "#.#"],
    "SW": ["###", "..#", "#.#"],
    "WN": ["#.#", "..#", "###"],
    "NES": ["#.#", "#..", "#.#"],
    "ESW": ["###", "...", "#.#"],
    "NSW": ["#.#", "..#", "#.#"],
    "NEW": ["#.#", "...", "###"],
}


LARGE_MAZE_SIZE = 8
GOAL_LOCATION = BoardLocation(5, 4)

DEPTH_VALUES = range(1, 11)
MAX_STEPS = 40
MAX_EXPANSIONS = 30000
DEPTH_VALUES = range(1, 7)

RESULT_DIR = os.path.join(
    os.path.dirname(__file__),
    "rtastar_experiment_results",
)


def build_large_maze_string(size=8):
    """
    Deterministic mixed-piece square maze generator.

    Produces a repeatable maze using:
        - corridor pieces
        - corner (L) pieces
        - T pieces

    This avoids the overly symmetric all-corner layouts that caused poor
    experimental behavior.

    Start cell:
        (0, 0)

    Goal cell:
        (size-1, size-1)
    """

    palette = [
        "V", "H",          # corridors
        "NE", "ES", "SW", "WN",   # L pieces
        "NES", "ESW", "NSW", "NEW",  # T pieces
    ]

    grid = []

    for r in range(size):
        row = []

        for c in range(size):

            # deterministic pseudo-random mix
            idx = (
                r * 13
                + c * 17
                + r * c * 7
                + r * r
                + c * c
            ) % len(palette)

            card = palette[idx]

            # Ensure reasonable start / goal cards
            if (r, c) == (0, 0):
                card = "ES"

            elif (r, c) == (size - 1, size - 1):
                card = "WN"

            row.append(card)

        grid.append(row)

    lines = [""]
    separator = "-" * (4 * size - 1)

    for r in range(size):

        for subrow in range(3):

            line = "|".join(
                CARD_PATTERNS[grid[r][c]][subrow]
                for c in range(size)
            ) + "|"

            lines.append(line)

        if r == size - 1:
            lines.append(separator + "*")
        else:
            lines.append(separator + "|")

    return "\n".join(lines)


LARGE_MAZE_STRING = build_large_maze_string(LARGE_MAZE_SIZE)


def build_initial_state():
    labyrinth_map = LabyrinthMap(
        maze_string=LARGE_MAZE_STRING,
        goal_location=GOAL_LOCATION,
    )

    game = Game(
        identifier=1,
        board=labyrinth_map.board,
        turns=Turns(),
    )

    player = Player(identifier=1)
    game.add_player(player)

    state = State(game.board, player)

    return labyrinth_map, state


def apply_turn(labyrinth_map, state, turn_action):

    shifted_state = labyrinth_map.apply_action(
        state,
        turn_action.shift_action,
    )

    if turn_action.move_action is None:
        return shifted_state

    return labyrinth_map.apply_action(
        shifted_state,
        turn_action.move_action,
    )


def choose_next_action(plan_result):
    result, visited = plan_result

    if result is None:
        return None, visited

    _state_path, action_path = result

    if not action_path:
        return None, visited

    return action_path[0], visited


def format_turn_action(action):
    if isinstance(action, TurnAction) or hasattr(action, "shift_action"):
        shift_text = _format_action(action.shift_action)

        if action.move_action is None:
            move_text = "NO MOVE"
        else:
            move_text = _format_action(action.move_action)

        return f"{shift_text} THEN {move_text}"

    return _format_action(action)


def turn_step_cost(planner, state, turn_action):
    """
    Recompute the cost of the executed first turn for logging.

    This mirrors the current RTAStar successor cost:
        turn_penalty + path length from shifted player location to move target.

    The working rtastar.py computes this inside turn_successors(), but does not
    expose a public cost method. 
    """

    shifted_state = planner.labyrinth_map.apply_action(
        state,
        turn_action.shift_action,
    )

    if turn_action.move_action is None:
        return planner.turn_penalty

    player_after_shift = planner.player_location(shifted_state)
    move_location = turn_action.move_action.move_location

    graph = planner.__class__.graph_path

    # Call the instance method from rtastar.py.
    path = planner.graph_path(
        __import__("labyrinth.model.reachable", fromlist=["Graph"]).Graph(
            shifted_state.board.maze
        ),
        player_after_shift,
        move_location,
    )

    move_cost = len(path) - 1

    return planner.turn_penalty + move_cost


def run_depth(depth):
    labyrinth_map, state = build_initial_state()

    planner = RTAStar(
        labyrinth_map=labyrinth_map,
        depth=depth,
        heuristic_weight=1.0,
        max_expansions=MAX_EXPANSIONS,
    )

    print(f"\n[depth={depth}] starting run")

    total_cost = 0.0
    total_runtime = 0.0
    total_expanded = 0
    turns = 0
    solved = False
    last_step_cost = None
    seen_player_goal_pairs = {}
    cycle_detected = False

    for step in range(MAX_STEPS):
        if step % 5 == 0:
            print(f"[depth={depth}] step {step}: planning...")

        start_time = time.perf_counter()
        plan_result = planner.plan(state)
        runtime = time.perf_counter() - start_time

        next_action, visited = choose_next_action(plan_result)

        total_runtime += runtime
        total_expanded += len(visited)

        if next_action is None:
            print(f"[depth={depth}] stopped: planner returned no action")
            break

        step_cost = turn_step_cost(planner, state, next_action)

        total_cost += step_cost
        last_step_cost = step_cost
        turns += 1

        if step % 5 == 0:
            print(
                f"[depth={depth}] step {step}: "
                f"player={planner.player_location(state)}, "
                f"goal={planner.goal_location(state)}, "
                f"cost={step_cost:.2f}, "
                f"expanded={len(visited)}, "
                f"runtime={runtime:.4f}s, "
                f"action={format_turn_action(next_action)}"
            )

        state = apply_turn(labyrinth_map, state, next_action)
        # pair = (
        #     planner.player_location(state),
        #     planner.goal_location(state),
        # )

        # if pair in seen_player_goal_pairs:
        #     cycle_detected = True
        #     print(
        #         f"[depth={depth}] cycle detected at step {step}: "
        #         f"player={pair[0]}, goal={pair[1]}, "
        #         f"previous_step={seen_player_goal_pairs[pair]}"
        #     )
        #     break

        # seen_player_goal_pairs[pair] = step
        # if step < 3:
        #     print("before leftover:", state.board.extra_maze_card)

        # state = apply_turn(labyrinth_map, state, next_action)

        # if step < 3:
        #     print("after leftover: ", state.board.extra_maze_card)

        if labyrinth_map.is_goal(state):
            solved = True
            print(f"[depth={depth}] solved at step {step}, turns={turns}")
            break

    avg_step_cost = total_cost / turns if turns else float("inf")

    result = {
        "depth": depth,
        "solved": solved,
        "turns": turns,
        "total_cost": total_cost,
        "avg_step_cost": avg_step_cost,
        "last_step_cost": last_step_cost if last_step_cost is not None else float("inf"),
        "runtime_sec": total_runtime,
        "expanded_states": total_expanded,
    }

    print(
        f"[depth={depth}] step {step}: "
        f"player={planner.player_location(state)}, "
        f"goal={planner.goal_location(state)}, "
        f"cost={step_cost:.2f}, "
        f"expanded={len(visited)}, "
        f"runtime={runtime:.4f}s, "
        f"action={format_turn_action(next_action)}"
    )

    return result


def write_csv(results):
    os.makedirs(RESULT_DIR, exist_ok=True)

    out_path = os.path.join(
        RESULT_DIR,
        "rtastar_depth_results.csv",
    )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "depth",
                "solved",
                "turns",
                "total_cost",
                "avg_step_cost",
                "last_step_cost",
                "runtime_sec",
                "expanded_states",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nWrote CSV: {out_path}")


def make_plots(results):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots.")
        return

    os.makedirs(RESULT_DIR, exist_ok=True)

    depths = [r["depth"] for r in results]

    plot_specs = [
        ("total_cost", "Total Cost"),
        ("turns", "Number of Turns"),
        ("avg_step_cost", "Average Step Cost"),
        ("runtime_sec", "Runtime (seconds)"),
    ]

    for key, title in plot_specs:
        plt.figure()
        plt.plot(depths, [r[key] for r in results], marker="o")
        plt.xlabel("Depth Limit")
        plt.ylabel(title)
        plt.title(f"{title} vs Depth Limit")
        plt.grid(True)

        out_path = os.path.join(
            RESULT_DIR,
            f"{key}_vs_depth.png",
        )

        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

        print(f"Wrote plot: {out_path}")

    fig, ax1 = plt.subplots()

    ax1.plot(
        depths,
        [r["total_cost"] for r in results],
        marker="o",
    )
    ax1.set_xlabel("Depth Limit")
    ax1.set_ylabel("Total Cost")

    ax2 = ax1.twinx()
    ax2.plot(
        depths,
        [r["runtime_sec"] for r in results],
        marker="s",
    )
    ax2.set_ylabel("Runtime (seconds)")

    plt.title("Total Cost and Runtime vs Depth Limit")

    out_path = os.path.join(
        RESULT_DIR,
        "total_cost_and_runtime_vs_depth.png",
    )

    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Wrote plot: {out_path}")


def main():
    print("RTA* depth experiment")
    print(f"maze size: {LARGE_MAZE_SIZE}x{LARGE_MAZE_SIZE}")
    print(f"goal location: {GOAL_LOCATION}")
    print(f"depth values: {list(DEPTH_VALUES)}")
    print(f"max steps per run: {MAX_STEPS}")
    print(f"max expansions per plan call: {MAX_EXPANSIONS}")

    print("\nGenerated analysis maze:")
    print(LARGE_MAZE_STRING)

    print("\nStarting depth sweep...")

    results = []

    for depth in DEPTH_VALUES:
        results.append(run_depth(depth))

    write_csv(results)
    make_plots(results)

    print("\nSummary:")
    for r in results:
        print(
            f"depth={r['depth']:2d} | "
            f"solved={str(r['solved']):5s} | "
            f"turns={r['turns']:3d} | "
            f"cost={r['total_cost']:8.2f} | "
            f"avg_step={r['avg_step_cost']:7.2f} | "
            f"runtime={r['runtime_sec']:8.3f}s | "
            f"expanded={r['expanded_states']}"
        )


if __name__ == "__main__":
    main()