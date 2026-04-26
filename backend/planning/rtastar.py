"""
Real-Time A* / bounded-horizon planner for Labyrinth.

This planner treats one Labyrinth turn as:
    1. mandatory shift/insert of leftover card with chosen rotation
    2. optional movement through reachable free space

Each complete shift+optional-move turn has unit cost. This makes the
planner optimize number of Labyrinth turns rather than number of squares
walked after a shift.

At each call to plan(), it searches `depth` turns ahead, returns the best
partial plan, and the demo should execute only the first TurnAction before
replanning. This is the "limited search horizon + commit one move" pattern
from Real-Time Heuristic Search / RTA*.
"""

import heapq
from dataclasses import dataclass

from labyrinth.model.reachable import Graph


INF = float("inf")


@dataclass(frozen=True)
class TurnAction:
    shift_action: object
    move_action: object | None


def _row_col(location):
    """Support this repo's BoardLocation row/column API."""
    try:
        return location.row, location.column
    except AttributeError:
        return location.getRow(), location.getColumn()


class RTAStar:
    def __init__(
        self,
        labyrinth_map,
        depth: int = 1,
        heuristic_weight: float = 1.0,
        max_expansions: int = 5000,
    ):
        """
        depth:
            Lookahead depth in Labyrinth turns. depth=1 evaluates all legal
            next turns and picks the best. depth=2 searches two turns ahead,
            then still returns only the first turn.

        heuristic_weight:
            Score is g + heuristic_weight * h.

        max_expansions:
            Safety cap to keep the planner from exploding.
        """
        if depth < 1:
            raise ValueError("RTAStar depth must be >= 1")

        self.labyrinth_map = labyrinth_map
        self.depth = depth
        self.heuristic_weight = heuristic_weight
        self.max_expansions = max_expansions

    def player_location(self, state):
        return state.board.maze.maze_card_location(state.player.piece.maze_card)

    def goal_location(self, state):
        return state.board.maze.maze_card_location(state.board.objective_maze_card)

    def is_goal(self, state) -> bool:
        return self.labyrinth_map.is_goal(state)

    def heuristic(self, state) -> float:
        """Manhattan distance from player location to current objective location."""
        player_loc = self.player_location(state)
        goal_loc = self.goal_location(state)

        if player_loc is None or goal_loc is None:
            return INF

        pr, pc = _row_col(player_loc)
        gr, gc = _row_col(goal_loc)
        return abs(pr - gr) + abs(pc - gc)

    def turn_cost(self, shifted_state, move_action) -> int:
        """
        Cost of one complete Labyrinth turn.

        A turn is the atomic action RTA* commits to: one mandatory shift plus
        an optional move. Using unit cost here makes the bounded search prefer
        plans that reach the goal in fewer turns.

        We still validate move reachability on the shifted board, because
        invalid move actions should not be expanded.
        """
        if move_action is None or move_action.move_location is None:
            return 1

        start = self.player_location(shifted_state)
        goal = move_action.move_location

        if start is None or goal is None:
            return INF

        if not Graph(shifted_state.board.maze).is_reachable(start, goal):
            return INF

        return 1

    def apply_turn(self, state, turn_action: TurnAction):
        """
        Apply one legal Labyrinth turn:
            mandatory shift, then optional movement.
        """
        shifted_state = self.labyrinth_map.apply_action(
            state,
            turn_action.shift_action,
        )

        if turn_action.move_action is None:
            return shifted_state

        return self.labyrinth_map.apply_action(
            shifted_state,
            turn_action.move_action,
        )

    def turn_successors(self, state):
        """
        Generate legal turn successors:
            state -> shift -> optional move

        Yields:
            (next_state, turn_action, turn_cost)
        """

        # get all available actions
        raw_actions = list(self.labyrinth_map.get_actions(state))

        # Isolate the shift actions
        shift_actions = [
            action for action in raw_actions
            if getattr(action, "shift_location", None) is not None
        ]

        # Determine all available successors after all available shifts
        for shift_action in shift_actions:
            shifted_state = self.labyrinth_map.apply_action(state, shift_action)

            # Always allow "shift and do not move". This is still a full
            # Labyrinth turn, so it has unit cost.
            stay_turn = TurnAction(shift_action=shift_action, move_action=None)
            yield shifted_state, stay_turn, 1

            # Get all move actions available after shift
            move_actions = [
                action for action in self.labyrinth_map.get_actions(shifted_state)
                if getattr(action, "move_location", None) is not None
            ]

            # Search over all move actions after shift
            for move_action in move_actions:
                turn = TurnAction(shift_action=shift_action, move_action=move_action)
                next_state = self.labyrinth_map.apply_action(shifted_state, move_action)
                cost = self.turn_cost(shifted_state, move_action)
                yield next_state, turn, cost

    def plan(self, start_state):
        """
        Bounded-horizon RTA* / minimin lookahead.

        Returns:
            ((state_path, action_path), visited)

        The caller should execute only action_path[0], then replan.

        QUEUE IS A WORK IN PROGRESS, THIS IS A POTENTION IMPLEMENTATION
        Queue entries:
          score:
              Primary A*/RTA* priority, computed as:
                  g_turns + heuristic_weight * heuristic(state)
        
          g_walk:
              Secondary tie-breaker: total player movement distance accumulated
              along this lookahead path. This is used to prefer shorter movement
              among plans with the same turn score.
        
          depth_used:
              Number of Labyrinth turns simulated so far in the lookahead tree.
              Used to enforce the search horizon.
        
          counter:
              Unique insertion ID used as a final tie-breaker so heapq never tries
              to compare State objects directly.
        
          g_turns:
              Number of complete Labyrinth turns accumulated so far.
        
          state:
              Current simulated state at this queue node.
        
          path:
              State path from start_state to this node.
        
          actions:
              TurnAction path from start_state to this node.
        """

        best_goal = None
        best_goal_key = None

        if self.is_goal(start_state):
            return (([start_state], []), {start_state})

        counter = 0
        expansions = 0
        visited = set()

        # Queue entries CURRENTLY:
        #   cost = g + w*h
        #   depth_used
        #   counter: an insertion ID (tie breaker), garenteed to be unique even when cost and depth are equal
        #   g = turn cost accumulated in the lookahead tree
        #   state
        #   path
        #   actions
        open_heap = []

        start_score = self.heuristic_weight * self.heuristic(start_state)
        heapq.heappush(
            open_heap,
            (start_score, 0, counter, 0, start_state, [start_state], []),
        )
        counter += 1

        best_frontier = None
        best_score = INF

        while open_heap and expansions < self.max_expansions:
            score, depth_used, _, g_cost, state, path, actions = heapq.heappop(open_heap)
            visited.add(state)

            if self.is_goal(state):
                return ((path, actions), visited)
            # if self.is_goal(state):
            #     key = (g_turns, g_walk)
            #     if best_goal_key is None or key < best_goal_key:
            #         best_goal_key = key
            #         best_goal = (path, actions)

                # Safe stop only when remaining heap cannot beat this goal
                if not open_heap or open_heap[0][0] > score:
                    return best_goal, visited

                continue

            # Horizon reached: evaluate this frontier node.
            if depth_used >= self.depth:
                frontier_score = g_cost + self.heuristic_weight * self.heuristic(state)
                if frontier_score < best_score:
                    best_score = frontier_score
                    best_frontier = (path, actions)
                continue

            expansions += 1

            for next_state, turn_action, step_cost in self.turn_successors(state):
                if step_cost == INF:
                    continue

                new_g = g_cost + step_cost
                new_score = new_g + self.heuristic_weight * self.heuristic(next_state)

                heapq.heappush(
                    open_heap,
                    (
                        new_score,
                        depth_used + 1,
                        counter,
                        new_g,
                        next_state,
                        path + [next_state],
                        actions + [turn_action],
                    ),
                )
                counter += 1

        # If no goal found within the horizon, execute the first action of the
        # best frontier path discovered.
        if best_frontier is not None:
            return best_frontier, visited

        return (None, visited)
