# Labyrinth Planning Algorithms

To run our planning module, we suggest executing all commands below from
the backend directory.

## To run some illustrative examples run 
    python -m planning.examples

## To run the A* graph search on an example maze run
    python -m planning.astar

## To replay a plan, translate the plan to JSON format found under the plans directory and run
    python backend/planning/run_trial.py \
    --plan backend/planning/plans/example_plan.json \
    --base-url http://127.0.0.1 \
    --game-id 0 \
    --trials 5 \
    --step-delay 0.2 \
    -v

The backend still enforces a shift action then a move action, i.e. plans of the form shift move shift move etc.
I aim to disable this by setting the environment variable below. I reccommend putting this in a .env file at the root
    ALLOW_ARBITRARY_ACTION_ORDER=true

Also this run_trial.py script does not support adding a maze string or the goal objective yet, I am working on that next.


# Notes on planning in the labyrinth board

Here are some notes I've taken about the complexities of what I've implemented

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

