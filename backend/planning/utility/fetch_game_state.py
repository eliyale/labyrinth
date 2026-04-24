#!/usr/bin/env python3
"""
Print game id, player ids, and whose turn it is — same data the browser polls from GET /state.

The bundled web client (web-client/src/services/game-api.js) always uses game id 0 for all
API paths. If your replay plan used another gameId, shift/move can fail or hit the wrong game.

Usage (Docker nginx on port 80):

  python backend/planning/fetch_game_state.py --base-url http://127.0.0.1 --game-id 0

Local Flask (no Docker):

  python backend/planning/fetch_game_state.py --base-url http://127.0.0.1:5000 --game-id 0

Optional: dump full JSON

  python backend/planning/fetch_game_state.py --game-id 0 --json

If you get GAME_NOT_FOUND for game 0, no row exists yet. GET /state does not create a game.
Create one the same way the UI does — first player added — then state exists:

  curl -X POST http://127.0.0.1/api/games/0/players -H 'Content-Type: application/json' -d '{}'

Or use this script (creates the game row on 404, and adds a player if the game has zero players):

  python backend/planning/fetch_game_state.py --base-url http://127.0.0.1 --init-game

Note: labyrinth/__init__.py runs init_database() on app startup, which drops the games table,
so the DB is empty after each server restart until someone adds a player again.

On macOS, AirPlay Receiver often listens on port 5000 and answers random HTTP with HTTP 403 and an
empty body. If you see that, do not use :5000 for Flask; use Docker nginx on port 80, or run Flask
on another port (e.g. 5001) and disable AirPlay for receiver in System Settings.
"""

from __future__ import annotations

import argparse
import json
import sys

import requests


def _print_game_not_found_help(game_id: int, base: str) -> None:
    print(
        "  No game row in SQLite for this id. GET /api/games/{id}/state only loads; it never creates.",
        file=sys.stderr,
    )
    print(
        f"  Create game {game_id} (and first player) with:",
        file=sys.stderr,
    )
    print(
        f"    curl -X POST {base}/api/games/{game_id}/players "
        "-H 'Content-Type: application/json' -d '{{}}'",
        file=sys.stderr,
    )
    print(
        "  Or open the web UI, choose online play, and add a player (same POST under the hood).",
        file=sys.stderr,
    )
    if game_id != 0:
        print(
            "  Hint: the bundled web client always uses game id 0; try --game-id 0 if unsure.",
            file=sys.stderr,
        )


def _post_first_player(session: requests.Session, base: str, game_id: int) -> requests.Response:
    url = f"{base}/api/games/{game_id}/players"
    return session.post(url, json={}, headers={"Content-Type": "application/json"}, timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1", help="Origin (no trailing slash), e.g. http://localhost")
    parser.add_argument("--game-id", type=int, default=0, help="Path segment in /api/games/{id}/state")
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of a summary")
    parser.add_argument(
        "--init-game",
        action="store_true",
        help="Ensure a playable game: on 404 POST /players to create the row; if GET succeeds but "
        "players is empty, POST /players once to add the first human player, then GET again",
    )
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    session = requests.Session()
    url = f"{base}/api/games/{args.game_id}/state"
    response = session.get(url, timeout=10)

    if response.status_code == 404 and args.init_game:
        body = response.text
        try:
            err = response.json()
            if err.get("key") != "GAME_NOT_FOUND":
                print(f"GET {url} -> HTTP 404 {body!r}", file=sys.stderr)
                sys.exit(1)
        except Exception:
            print(f"GET {url} -> HTTP 404 {body!r}", file=sys.stderr)
            sys.exit(1)
        create_url = f"{base}/api/games/{args.game_id}/players"
        created = _post_first_player(session, base, args.game_id)
        if not created.ok:
            print(f"POST {create_url} -> HTTP {created.status_code} {created.text!r}", file=sys.stderr)
            sys.exit(1)
        print(f"Created game {args.game_id} (first player POST ok). Fetching state again…", file=sys.stderr)
        response = session.get(url, timeout=10)

    if response.ok and args.init_game:
        try:
            probe = response.json()
        except Exception:
            probe = {}
        if not probe.get("players"):
            create_url = f"{base}/api/games/{args.game_id}/players"
            added = _post_first_player(session, base, args.game_id)
            if not added.ok:
                print(
                    f"Game {args.game_id} had no players; POST {create_url} -> "
                    f"HTTP {added.status_code} {added.text!r}",
                    file=sys.stderr,
                )
                sys.exit(1)
            print("Added first player (game had zero players in /state). Fetching again…", file=sys.stderr)
            response = session.get(url, timeout=10)

    if not response.ok:
        print(f"GET {url} -> HTTP {response.status_code} {response.text!r}", file=sys.stderr)
        if response.status_code == 404:
            _print_game_not_found_help(args.game_id, base)
            if not args.init_game:
                print("  Or re-run with:  --init-game", file=sys.stderr)
        sys.exit(1)

    data = response.json()
    if args.json:
        print(json.dumps(data, indent=2))
        return

    game_id = data.get("id", args.game_id)
    players = data.get("players", [])
    player_ids = [p.get("id") for p in players]
    next_action = data.get("nextAction")
    print(f"GET {url}")
    print(f"  game id (JSON 'id'):     {game_id}")
    print(f"  player ids:              {player_ids}")
    if next_action:
        print(f"  nextAction.playerId:   {next_action.get('playerId')}")
        print(f"  nextAction.action:     {next_action.get('action')}")
    else:
        print("  nextAction:              (null — no players yet, or turns not started)")
    print()
    if not player_ids:
        print(
            "WARNING: no players in this game. POST /api/games/{}/players again or re-run with --init-game.".format(
                game_id
            )
        )
    else:
        print(
            "Use gameId =",
            game_id,
            "and for replay_plan use --player-id matching nextAction.playerId when it is that player's turn "
            "(often 1 for the first human player).",
        )


if __name__ == "__main__":
    main()
