# Labyrinth Planning Algorithms

To run our planning module, we suggest executing all commands below from
the backend directory.

## To run some illustrative examples run 
    python -m planning.examples

## To run the A* graph search on an example maze run
    python -m planning.astar


# Notes on planning in the labyrinth board

## Heuristics

As a reminder a Heuristic is admissible if it always underestimates the true
cost to go to the goal, i.e. h(s) <= h*(s)

## Cost structures:

## Multi-goal:

There are two ways to think about multi-goal, visting n subgoal states in the order
given, or in any order. For our problem we have chosen to vist the goals in order,
which is how the original game is played. This scenario is a bit easier to solve,
there are less combinatorix in the state space. The augmented state space must now 
also includes an integer k which tracks the number of goals visited. is_goal returns
true only when k == n. Simply running A* to G1, G2 and so on will not 
automatically be globally optimal, as one set of actions to reach G1 may leave the 
board in a more optimal state to reach g2 than a differnt set of actions. So we must run A* on the full augmented state space.

Furthermore, if one of the goals is shifted by an action, the planner must be able
to account for this. WE keep references to the goals as instances of the card itself
which the board shifting actions automatically update.

