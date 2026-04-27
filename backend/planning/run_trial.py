#!/usr/bin/env python3
"""
Run a plan reliably over API for repeated trials.

This script removes turn-management pain by polling /state and only sending
actions when nextAction matches the expected player/action.

It can reset the game each trial, create a fresh player, replay the plan,
and repeat.

Plan format matches replay_plan.py, plus optional layout:

{
  "baseUrl": "http://127.0.0.1",
  "gameId": 0,
  "playerId": 1,
  "mazeString": "\\n###|...",
  "steps": [
    {"type": "shift", "row": 0, "column": 1, "leftoverRotation": 90},
    {"type": "move", "row": 0, "column": 1}
  ]
}

If "mazeString" or "MAZE_STRING" is set, run_trial sends PUT /api/games/{id} with {"mazeString": ...}
before executing steps (each trial if --trials > 1). Same format as labyrinth.model.factories.create_maze.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


def _load_plan(path: pathlib.Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _state_url(base: str, game_id: int) -> str:
    return f"{base.rstrip('/')}/api/games/{game_id}/state"


def _add_player_url(base: str, game_id: int) -> str:
    return f"{base.rstrip('/')}/api/games/{game_id}/players"


def _delete_player_url(base: str, game_id: int, player_id: int) -> str:
    return f"{base.rstrip('/')}/api/games/{game_id}/players/{player_id}"


def _change_game_url(base: str, game_id: int) -> str:
    return f"{base.rstrip('/')}/api/games/{game_id}"


def _shift_url(base: str, game_id: int, player_id: int) -> str:
    return f"{base.rstrip('/')}/api/games/{game_id}/shift?p_id={player_id}"


def _move_url(base: str, game_id: int, player_id: int) -> str:
    return f"{base.rstrip('/')}/api/games/{game_id}/move?p_id={player_id}"


def _raise_for_non_ok(response: requests.Response, context: str) -> None:
    if response.ok:
        return
    raise RuntimeError(f"{context} -> HTTP {response.status_code}: {response.text!r}")


def _fetch_state(session: requests.Session, base: str, game_id: int) -> requests.Response:
    return session.get(_state_url(base, game_id), timeout=10)


def _step_to_payload(step: Dict[str, Any]) -> Tuple[str, Dict[str, Any], str]:
    step_type = (step.get("type") or "").lower()
    if step_type == "shift":
        payload = {
            "location": {"row": int(step["row"]), "column": int(step["column"])},
            "leftoverRotation": int(step.get("leftoverRotation", step.get("rotation", 0))),
        }
        return step_type, payload, "SHIFT"
    if step_type == "move":
        payload = {"location": {"row": int(step["row"]), "column": int(step["column"])}}
        return step_type, payload, "MOVE"
    raise ValueError(f"Unknown step type: {step!r}")


def _ensure_game_and_optional_fresh_player(
    session: requests.Session,
    base: str,
    game_id: int,
    fresh_player: bool,
    explicit_player_id: Optional[int],
    verbose: bool,
) -> int:
    state_response = _fetch_state(session, base, game_id)

    # Game missing: create by adding first player.
    if state_response.status_code == 404:
        create = session.post(_add_player_url(base, game_id), json={}, timeout=10)
        _raise_for_non_ok(create, f"POST {_add_player_url(base, game_id)}")
        created_player = create.json().get("id")
        if created_player is None:
            raise RuntimeError("Add player succeeded but response had no 'id'")
        if verbose:
            print(f"Created game {game_id} with first player id={created_player}")
        return int(explicit_player_id) if explicit_player_id is not None else int(created_player)

    _raise_for_non_ok(state_response, f"GET {_state_url(base, game_id)}")
    state = state_response.json()
    players = state.get("players", [])

    if fresh_player:
        # Remove all existing players first for deterministic turn order.
        for p in players:
            pid = int(p["id"])
            delete = session.delete(_delete_player_url(base, game_id, pid), timeout=10)
            _raise_for_non_ok(delete, f"DELETE {_delete_player_url(base, game_id, pid)}")
        create = session.post(_add_player_url(base, game_id), json={}, timeout=10)
        _raise_for_non_ok(create, f"POST {_add_player_url(base, game_id)}")
        created_player = create.json().get("id")
        if created_player is None:
            raise RuntimeError("Add player succeeded but response had no 'id'")
        if verbose:
            print(f"Reset players; created fresh player id={created_player}")
        return int(explicit_player_id) if explicit_player_id is not None else int(created_player)

    # Reuse existing players.
    if explicit_player_id is not None:
        return int(explicit_player_id)
    if players:
        return int(players[0]["id"])

    # No players but game exists: add one.
    create = session.post(_add_player_url(base, game_id), json={}, timeout=10)
    _raise_for_non_ok(create, f"POST {_add_player_url(base, game_id)}")
    created_player = create.json().get("id")
    if created_player is None:
        raise RuntimeError("Add player succeeded but response had no 'id'")
    return int(created_player)


def _reset_game(session: requests.Session, base: str, game_id: int, maze_size: int) -> None:
    response = session.put(_change_game_url(base, game_id), json={"mazeSize": maze_size}, timeout=10)
    _raise_for_non_ok(response, f"PUT {_change_game_url(base, game_id)}")


def _put_game_maze_string(session: requests.Session, base: str, game_id: int, maze_string: str) -> None:
    """PUT /api/games/{id} with mazeString — server restarts game with that layout."""
    response = session.put(_change_game_url(base, game_id), json={"mazeString": maze_string}, timeout=60)
    _raise_for_non_ok(response, f"PUT {_change_game_url(base, game_id)} (mazeString)")


def _wait_for_turn(
    session: requests.Session,
    base: str,
    game_id: int,
    player_id: int,
    expected_action: str,
    timeout_s: float,
    poll_interval_s: float,
) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        response = _fetch_state(session, base, game_id)
        _raise_for_non_ok(response, f"GET {_state_url(base, game_id)}")
        state = response.json()
        next_action = state.get("nextAction")
        if (
            next_action
            and int(next_action.get("playerId", -1)) == player_id
            and next_action.get("action") == expected_action
        ):
            return
        time.sleep(poll_interval_s)
    raise TimeoutError(
        f"Timed out waiting for nextAction {expected_action} for player {player_id} "
        f"in game {game_id}"
    )


def _current_player_location(
    session: requests.Session,
    base: str,
    game_id: int,
    player_id: int,
) -> Dict[str, int]:
    response = _fetch_state(session, base, game_id)
    _raise_for_non_ok(response, f"GET {_state_url(base, game_id)}")
    state = response.json()
    players = state.get("players", [])
    player = next((p for p in players if int(p.get("id", -1)) == player_id), None)
    if player is None:
        raise RuntimeError(f"Player {player_id} not found in state for game {game_id}")
    player_card_id = int(player.get("mazeCardId"))

    maze_cards = state.get("maze", {}).get("mazeCards", [])
    card = next((c for c in maze_cards if int(c.get("id", -1)) == player_card_id), None)
    if card is None or card.get("location") is None:
        raise RuntimeError(
            f"Could not resolve current location for player {player_id} (card id {player_card_id})"
        )
    location = card["location"]
    return {"row": int(location["row"]), "column": int(location["column"])}


def _execute_plan_steps(
    session: requests.Session,
    base: str,
    game_id: int,
    player_id: int,
    steps: List[Dict[str, Any]],
    wait_timeout_s: float,
    poll_interval_s: float,
    step_delay_s: float,
    ignore_turn_state: bool,
    auto_noop_move: bool,
    verbose: bool,
) -> None:
    previous_step_type: Optional[str] = None
    for index, step in enumerate(steps, start=1):
        step_type, payload, expected_action = _step_to_payload(step)

        # Turn normalizer: inject a no-op MOVE between consecutive SHIFTs.
        if auto_noop_move and not ignore_turn_state and previous_step_type == "shift" and step_type == "shift":
            noop_location = _current_player_location(session, base, game_id, player_id)
            _wait_for_turn(
                session,
                base,
                game_id,
                player_id,
                expected_action="MOVE",
                timeout_s=wait_timeout_s,
                poll_interval_s=poll_interval_s,
            )
            noop_url = _move_url(base, game_id, player_id)
            noop_payload = {"location": noop_location}
            if verbose:
                print(f"[inject] POST {noop_url} :: {json.dumps(noop_payload)}")
            noop_response = session.post(noop_url, json=noop_payload, timeout=10)
            _raise_for_non_ok(noop_response, f"POST {noop_url}")

        if not ignore_turn_state:
            _wait_for_turn(
                session,
                base,
                game_id,
                player_id,
                expected_action=expected_action,
                timeout_s=wait_timeout_s,
                poll_interval_s=poll_interval_s,
            )
        url = _shift_url(base, game_id, player_id) if step_type == "shift" else _move_url(base, game_id, player_id)
        if verbose:
            print(f"[{index}/{len(steps)}] POST {url} :: {json.dumps(payload)}")
        response = session.post(url, json=payload, timeout=10)
        _raise_for_non_ok(response, f"POST {url}")
        previous_step_type = step_type
        if step_delay_s > 0 and index < len(steps):
            time.sleep(step_delay_s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=pathlib.Path, required=True, help="Path to plan JSON")
    parser.add_argument("--base-url", default=None, help="Override plan baseUrl")
    parser.add_argument("--game-id", type=int, default=None, help="Override plan gameId")
    parser.add_argument("--player-id", type=int, default=None, help="Force player id for execution")
    parser.add_argument("--trials", type=int, default=1, help="Number of reset/replay runs")
    parser.add_argument("--maze-size", type=int, default=7, help="Maze size for reset via PUT /games/{id} when no mazeString")
    parser.add_argument("--no-reset", action="store_true", help="Do not reset game each trial")
    parser.add_argument(
        "--reuse-players",
        action="store_true",
        help="Keep existing players instead of deleting and creating one fresh player each trial",
    )
    parser.add_argument("--wait-timeout", type=float, default=20.0, help="Seconds to wait for expected nextAction")
    parser.add_argument("--poll-interval", type=float, default=0.1, help="State polling interval in seconds")
    parser.add_argument("--step-delay", type=float, default=0.0, help="Sleep between successful steps")
    parser.add_argument(
        "--ignore-turn-state",
        action="store_true",
        help="Post plan steps in listed order without waiting for nextAction; use with backend "
        "ALLOW_ARBITRARY_ACTION_ORDER=True",
    )
    parser.add_argument(
        "--no-auto-noop-move",
        action="store_true",
        help="Disable automatic insertion of MOVE(current_location) between consecutive SHIFT steps",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if not args.plan.is_file():
        sys.exit(f"Plan file not found: {args.plan}")

    plan = _load_plan(args.plan)
    if "steps" not in plan or not isinstance(plan["steps"], list):
        sys.exit("Plan JSON must contain a list 'steps'")

    base_url = (args.base_url or plan.get("baseUrl") or "http://127.0.0.1").rstrip("/")
    game_id = int(args.game_id if args.game_id is not None else plan.get("gameId", 0))
    plan_player_id = plan.get("playerId")
    explicit_player_id = args.player_id if args.player_id is not None else plan_player_id
    steps: List[Dict[str, Any]] = plan["steps"]
    maze_string = plan.get("mazeString") or plan.get("MAZE_STRING")
    reset_each_trial = not args.no_reset
    fresh_player = not args.reuse_players

    session = requests.Session()

    for trial in range(1, args.trials + 1):
        print(f"=== Trial {trial}/{args.trials} ===")
        player_id = _ensure_game_and_optional_fresh_player(
            session,
            base_url,
            game_id,
            fresh_player=fresh_player,
            explicit_player_id=explicit_player_id,
            verbose=args.verbose,
        )
        if reset_each_trial:
            if maze_string:
                _put_game_maze_string(session, base_url, game_id, maze_string)
                if args.verbose:
                    print(f"Applied mazeString to game {game_id} (PUT /api/games/{game_id})")
            else:
                _reset_game(session, base_url, game_id, maze_size=args.maze_size)
                if args.verbose:
                    print(f"Reset game {game_id} to mazeSize={args.maze_size}")
        elif trial == 1 and maze_string:
            _put_game_maze_string(session, base_url, game_id, maze_string)
            if args.verbose:
                print(f"Applied mazeString to game {game_id} (one-shot, no --trials reset)")

        _execute_plan_steps(
            session,
            base_url,
            game_id,
            player_id,
            steps,
            wait_timeout_s=args.wait_timeout,
            poll_interval_s=args.poll_interval,
            step_delay_s=args.step_delay,
            ignore_turn_state=args.ignore_turn_state,
            auto_noop_move=not args.no_auto_noop_move,
            verbose=args.verbose,
        )
        time.sleep(1)
        print(f"Trial {trial} complete (game={game_id}, player={player_id}, steps={len(steps)}).")

    print("All trials completed.")


if __name__ == "__main__":
    main()
