"""
D* Lite planner for Labyrinth.

This implementation runs D* Lite over the discovered Labyrinth state graph.
It uses the existing LabyrinthMap interface:

    map.get_actions(state)
    map.apply_action(state, action)
    map.is_goal(state)

Return format matches a_star_search:
    ((path, action_path), visited) or (None, visited)
"""

import heapq
import math
from collections import defaultdict


INF = float("inf")


class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.entry_finder = {}
        self.counter = 0

    def insert(self, state, key):
        self.entry_finder[state] = key
        heapq.heappush(self.heap, (key[0], key[1], self.counter, state))
        self.counter += 1

    def remove(self, state):
        self.entry_finder.pop(state, None)

    def top_key(self):
        while self.heap:
            k1, k2, _, state = self.heap[0]
            key = (k1, k2)

            if state in self.entry_finder and self.entry_finder[state] == key:
                return key

            heapq.heappop(self.heap)

        return (INF, INF)

    def pop(self):
        while self.heap:
            k1, k2, _, state = heapq.heappop(self.heap)
            key = (k1, k2)

            if state in self.entry_finder and self.entry_finder[state] == key:
                del self.entry_finder[state]
                return state, key

        return None, (INF, INF)

    def contains(self, state):
        return state in self.entry_finder


class DStarLite:
    def __init__(
        self,
        labyrinth_map,
        heuristic,
        weight=1.0,
        max_discovery=30000,
        max_compute_steps=100000,
        max_path_steps=200,
    ):
        self.labyrinth_map = labyrinth_map
        self.heuristic = heuristic
        self.weight = weight

        self.max_discovery = max_discovery
        self.max_compute_steps = max_compute_steps
        self.max_path_steps = max_path_steps

        self.g = defaultdict(lambda: INF)
        self.rhs = defaultdict(lambda: INF)
        self.open = PriorityQueue()

        self.successors = {}
        self.predecessors = defaultdict(set)
        self.action_from_to = {}

        self.start = None
        self.last_start = None
        self.goal = None
        self.km = 0

        self.environment_changed = False
        self.last_result = None
        self.visited = set()

    def notify_environment_change(self):
        self.environment_changed = True

    def state_distance(self, a, b):
        """
        D* Lite key heuristic between two states.
        Uses player-board locations as a cheap distance.
        """
        try:
            a_loc = a.board.maze.maze_card_location(a.player.piece.maze_card)
            b_loc = b.board.maze.maze_card_location(b.player.piece.maze_card)

            if a_loc is None or b_loc is None:
                return 0

            return abs(a_loc.row - b_loc.row) + abs(a_loc.column - b_loc.column)

        except Exception:
            return 0

    def transition_cost(self, _state, _action, _next_state):
        return 1

    def calculate_key(self, state):
        value = min(self.g[state], self.rhs[state])
        return (
            value + self.weight * self.state_distance(self.start, state) + self.km,
            value,
        )

    def get_successors(self, state):
        if state in self.successors:
            return self.successors[state]

        result = []

        for action in self.labyrinth_map.get_actions(state):
            next_state = self.labyrinth_map.apply_action(state, action)

            if next_state == state:
                continue

            cost = self.transition_cost(state, action, next_state)

            result.append((next_state, action, cost))
            self.predecessors[next_state].add(state)
            self.action_from_to[(state, next_state)] = action

        self.successors[state] = result
        return result

    def discover_reachable_graph_until_goal(self, start):
        """
        Labyrinth has an implicit goal condition rather than one fixed goal node.
        So first we discover the state graph forward until a goal state is found.
        Then D* Lite repairs values backward from that discovered goal.
        """
        frontier = []
        counter = 0
        visited = set()

        heapq.heappush(frontier, (self.heuristic(start), counter, start))
        counter += 1

        found_goal = None

        while frontier and len(visited) < self.max_discovery:
            _, _, state = heapq.heappop(frontier)

            if state in visited:
                continue

            visited.add(state)

            if self.labyrinth_map.is_goal(state):
                found_goal = state
                break

            for next_state, _action, _cost in self.get_successors(state):
                if next_state not in visited:
                    heapq.heappush(
                        frontier,
                        (self.heuristic(next_state), counter, next_state),
                    )
                    counter += 1

        self.visited = visited
        return found_goal, visited

    def initialize(self, start):
        self.g = defaultdict(lambda: INF)
        self.rhs = defaultdict(lambda: INF)
        self.open = PriorityQueue()

        self.successors = {}
        self.predecessors = defaultdict(set)
        self.action_from_to = {}

        self.start = start
        self.last_start = start
        self.km = 0

        goal, visited = self.discover_reachable_graph_until_goal(start)
        self.goal = goal

        if self.goal is None:
            return False, visited

        self.rhs[self.goal] = 0
        self.open.insert(self.goal, self.calculate_key(self.goal))

        return True, visited

    def update_vertex(self, state):
        if state != self.goal:
            best_rhs = INF

            for next_state, _action, cost in self.get_successors(state):
                best_rhs = min(best_rhs, cost + self.g[next_state])

            self.rhs[state] = best_rhs

        if self.open.contains(state):
            self.open.remove(state)

        if self.g[state] != self.rhs[state]:
            self.open.insert(state, self.calculate_key(state))

    def compute_shortest_path(self):
        steps = 0

        while (
            self.open.top_key() < self.calculate_key(self.start)
            or self.rhs[self.start] != self.g[self.start]
        ):
            steps += 1

            if steps > self.max_compute_steps:
                break

            state, old_key = self.open.pop()

            if state is None:
                break

            new_key = self.calculate_key(state)

            if old_key < new_key:
                self.open.insert(state, new_key)

            elif self.g[state] > self.rhs[state]:
                self.g[state] = self.rhs[state]

                for pred in list(self.predecessors[state]):
                    self.update_vertex(pred)

            else:
                self.g[state] = INF
                self.update_vertex(state)

                for pred in list(self.predecessors[state]):
                    self.update_vertex(pred)

    def extract_path(self):
        if self.goal is None:
            return None

        if self.g[self.start] == INF and self.rhs[self.start] == INF:
            return None

        path = [self.start]
        action_path = []

        current = self.start

        for _ in range(self.max_path_steps):
            if self.labyrinth_map.is_goal(current):
                return path, action_path

            best = None
            best_value = INF

            for next_state, action, cost in self.get_successors(current):
                value = cost + self.g[next_state]

                if value < best_value:
                    best_value = value
                    best = (next_state, action)

            if best is None:
                return None

            next_state, action = best

            if next_state == current:
                return None

            path.append(next_state)
            action_path.append(action)
            current = next_state

        return None

    def plan(self, state):
        """
        Plan from the current state.

        If the environment changed, we rebuild the discovered graph because a
        Labyrinth shift can globally change the transition structure.
        """
        if (
            self.start is None
            or self.goal is None
            or self.environment_changed
        ):
            initialized, visited = self.initialize(state)

            if not initialized:
                self.last_result = (None, visited)
                self.environment_changed = False
                return self.last_result

        else:
            self.km += self.state_distance(self.last_start, state)
            self.last_start = state
            self.start = state

        self.start = state

        self.compute_shortest_path()

        path_result = self.extract_path()

        if path_result is None:
            self.last_result = (None, self.visited)
        else:
            self.last_result = (path_result, self.visited)

        self.environment_changed = False
        return self.last_result