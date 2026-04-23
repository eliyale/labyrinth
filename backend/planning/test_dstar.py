from backend.planning.dstar import DStarLite

blocked = False

def get_player_piece_location(state):
    return state

def dummy_transition(state, action):
    global blocked
    if action != "go":
        return state

    if state == (0,) and not blocked:
        return (1,)

    return state

def dummy_is_goal(state):
    return get_player_piece_location(state) == (1,)

def dummy_heuristic(state):
    return 0

planner = DStarLite(
    f=dummy_transition,
    is_goal=dummy_is_goal,
    actions=["go"],
    h=dummy_heuristic
)

def run_once():
    return planner.plan((0,))

print("=== First planning call ===")
result1, visited1 = run_once()
print("Result 1:", result1)
print("Visited 1:", visited1)

planner.notify_environment_change()
blocked = True

print("=== Second planning call after environment change ===")
result2, visited2 = run_once()
print("Result 2:", result2)
print("Visited 2:", visited2)

assert result1 is not None, "First planning call should find a path"
assert result1[0] == [(0,), (1,)], "First path is incorrect"
assert result2 is None, "Second planning call should fail after blocking the path"
assert visited2 == {(0,)}, "Visited states after blocking are incorrect"

planner.notify_environment_change()
blocked = False

print("=== Third planning call after environment opens again ===")
result3, visited3 = run_once()
print("Result 3:", result3)
print("Visited 3:", visited3)

assert result3 is not None, "Third planning call should find a path again"
assert result3[0] == [(0,), (1,)], "Third path is incorrect"