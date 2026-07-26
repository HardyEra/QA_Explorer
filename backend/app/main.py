"""Backward-compatible command-line entrypoint."""

from runner import run_exploration


if __name__ == "__main__":
    run_exploration(start_url="https://www.saucedemo.com")
