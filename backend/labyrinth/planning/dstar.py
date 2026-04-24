"""
D* Lite planner interface for Labyrinth.

Current implementation: repeated-A* replanning baseline.
The demo shows the D* Lite-style replanning pipeline:
plan → act → environment changes → replan.
"""

from labyrinth.planning.astar import a_star_search


class DStarLite:
    def __init__(self, labyrinth_map, heuristic, weight=1.0):
        self.labyrinth_map = labyrinth_map
        self.heuristic = heuristic
        self.weight = weight
        self.environment_changed = False
        self.last_result = None

    def notify_environment_change(self):
        self.environment_changed = True

    def plan(self, state):
        self.last_result = a_star_search(
            init_state=state,
            f=self.labyrinth_map.apply_action,
            is_goal=self.labyrinth_map.is_goal,
            actions=self.labyrinth_map.get_actions,
            h=self.heuristic,
            weight=self.weight,
        )
        self.environment_changed = False
        return self.last_result