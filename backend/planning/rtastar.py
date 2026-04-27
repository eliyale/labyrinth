import heapq
from dataclasses import dataclass
from labyrinth.model.reachable import Graph

INF = float("inf")


@dataclass(frozen=True)
class TurnAction:
    shift_action: object
    move_action: object | None


def _row_col(location):
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
        turn_penalty: int = 100,
        max_expansions: int = 50000,
    ):
        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.labyrinth_map = labyrinth_map
        self.depth = depth
        self.heuristic_weight = heuristic_weight
        self.turn_penalty = turn_penalty
        self.max_expansions = max_expansions

    def player_location(self, state):
        return state.board.maze.maze_card_location(state.player.piece.maze_card)

    def goal_location(self, state):
        return state.board.maze.maze_card_location(state.board.objective_maze_card)

    def is_goal(self, state):
        return self.labyrinth_map.is_goal(state)

    def manhattan(self, a, b):
        if a is None or b is None:
            return INF

        ar, ac = _row_col(a)
        br, bc = _row_col(b)
        return abs(ar - br) + abs(ac - bc)

    def heuristic(self, state):
        return self.manhattan(
            self.player_location(state),
            self.goal_location(state),
        )
    
    def apply_turn(labyrinth_map, state, turn_action):
        """
        Helper to apply turn
        """

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
    
    def graph_path(self, graph, source, target):
        """
        Use Graph's internal neighbor structure to reconstruct a path.
        Utility function, used to compute movement cost from current location to a node
        """
        reached = {source: source}
        frontier = [source]

        while frontier:
            current = frontier.pop(0)

            if current == target:
                break

            for neighbor in graph._neighbors(current):
                if neighbor not in reached:
                    reached[neighbor] = current
                    frontier.append(neighbor)

        if target not in reached:
            return []

        path = [target]
        node = target

        while reached[node] != node:
            node = reached[node]
            path.append(node)

        path.reverse()
        return path
    

    def turn_successors(self, state):
        """
        Generate possible next turns.

        One successor = one full turn:
            shift + optional move

        Cost:
            turn_penalty + step cost to a new location + heuristic cost to goal
        """

        for shift_action in self.labyrinth_map.get_actions(state):
            if getattr(shift_action, "shift_location", None) is None:
                continue

            shifted_state = self.labyrinth_map.apply_action(state, shift_action)
            player_after_shift = self.player_location(shifted_state)

            # Shift + no move.
            yield (
                shifted_state,
                TurnAction(shift_action, None),
                self.turn_penalty,
            )

            for move_action in self.labyrinth_map.get_actions(shifted_state):
                move_location = getattr(move_action, "move_location", None)

                if move_location is None:
                    continue

                next_state = self.labyrinth_map.apply_action(
                    shifted_state,
                    move_action,
                )

                graph = Graph(shifted_state.board.maze)
                path = self.graph_path(graph, player_after_shift, move_location)
                move_cost = len(path) - 1

                yield (
                    next_state,
                    TurnAction(shift_action, move_action),
                    self.turn_penalty + move_cost,
                )

    def plan(self, start_state):
        """
        RTA*/minimin lookahead.

        g is local to this call:
            g_x(n) = cost from current state x to lookahead node n

        f_x(n) = g_x(n) + h(n)

        Search is over complete-turn successors.
        """

        if self.is_goal(start_state):
            return (([start_state], []), {start_state})

        open_heap = []
        visited = set()

        counter = 0
        expansions = 0

        best_frontier = None
        best_frontier_key = (INF, INF)

        start_h = self.heuristic_weight * self.heuristic(start_state)

        heapq.heappush(
            open_heap,
            (
                start_h,        # f = local g + h
                start_h,        # tie-break: lower h means more progress
                counter,
                0,              # depth_used
                0,              # local g
                start_state,
                [start_state],
                [],
            ),
        )
        counter += 1

        while open_heap and expansions < self.max_expansions:
            (
                f_score,
                h_score,
                _,
                depth_used,
                g_cost,
                state,
                path,
                actions,
            ) = heapq.heappop(open_heap)

            visited.add(state)

            if self.is_goal(state):
                return ((path, actions), visited)

            if depth_used >= self.depth:
                frontier_key = (f_score, h_score)

                if frontier_key < best_frontier_key:
                    best_frontier_key = frontier_key
                    best_frontier = (path, actions)

                continue

            expansions += 1

            for next_state, turn_action, step_cost in self.turn_successors(state):
                if step_cost == INF:
                    continue

                next_g = g_cost + step_cost
                next_h = self.heuristic_weight * self.heuristic(next_state)
                next_f = next_g + next_h

                heapq.heappush(
                    open_heap,
                    (
                        next_f,
                        next_h,
                        counter,
                        depth_used + 1,
                        next_g,
                        next_state,
                        path + [next_state],
                        actions + [turn_action],
                    ),
                )
                counter += 1

        if best_frontier is not None:
            return best_frontier, visited

        return (None, visited)




