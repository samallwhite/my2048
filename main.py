"""2048 game — human-playable CLI entry point."""

import argparse
from game.game import Game
from game.board import UP, DOWN, LEFT, RIGHT

_KEY_MAP = {
    "w": UP,
    "a": LEFT,
    "s": DOWN,
    "d": RIGHT,
}

_DIR_NAMES = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT"}


def main():
    parser = argparse.ArgumentParser(description="2048 Game")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    args = parser.parse_args()

    game = Game(seed=args.seed)
    game.reset()

    print("2048 — WASD to move, Q to quit\n")

    while True:
        game.render()
        if game.board.is_game_over():
            print("Game Over!")
            break

        key = input("Move (w/a/s/d, q=quit): ").strip().lower()
        if key == "q":
            break
        if key not in _KEY_MAP:
            print(f"  Invalid key. Use: {', '.join(_KEY_MAP.keys())} or q")
            continue

        action = _KEY_MAP[key]
        state, reward, done, info = game.step(action)

        if info.get("invalid"):
            print("  Cannot move in that direction!")
        else:
            print(f"  -> {_DIR_NAMES[action]}, +{reward} pts")

    print(f"\nFinal score: {game.score}  Max tile: {game.max_tile}")


if __name__ == "__main__":
    main()
