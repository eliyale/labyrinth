"""
D* / replanning module for Labyrinth.

For now, this is a repeated-A* baseline:
whenever called, it simply runs A* from the current state.
Later, we will replace the internals with true D* Lite logic
without changing the external interface.
"""

from backend.planning.astar import a_star_search


def dstar_search(init_state, f, is_goal, actions, h, weight=1.0):
    """
    Temporary replanning baseline that reuses A*.
    This keeps the same interface we want for D*.
    """
    # Current behavior: replan from scratch every time this function is called.
    # This is a repeated-A* baseline, not true incremental D* Lite yet.
    return a_star_search(init_state, f, is_goal, actions, h, weight)

class DStarLite:
    def __init__(self, f, is_goal, actions, h, weight=1.0):
        self.f = f
        self.is_goal = is_goal
        self.actions = actions
        self.h = h
        self.weight = weight

        self.last_result = None
        self.environment_changed = False

        # Placeholder for future D* state
        self.last_state = None

    def plan(self, current_state):
        """
        For now: just call A* (same as baseline).
        Later: replace with incremental D* logic.
        """
        self.last_state = current_state
        self.last_result = a_star_search(
            current_state, self.f, self.is_goal, self.actions, self.h, self.weight
        )
        return self.last_result
    
    def notify_environment_change(self):
        """
        Mark that something in the environment changed.
        Later, true D* will use this to update the search incrementally.
        """
        self.environment_changed = True