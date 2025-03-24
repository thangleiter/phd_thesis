#!/usr/bin/env python3
import subprocess
import sys


def update_submodules() -> None:
    """Updates submodules according to branch specifications in .gitmodules"""
    print("Switched branches, updating submodules...")
    subprocess.run(["git", "submodule", "update"], check=False)


def main(args: list[str]) -> int:
    # Git passes 3 arguments to post-checkout:
    # $1: previous HEAD ref
    # $2: current HEAD ref
    # $3: flag indicating if this is a branch checkout (1) or a file checkout (0)

    if len(args) < 4:
        print("Error: Not enough arguments provided")
        return 1

    # Check if this is a branch checkout
    is_branch_checkout = args[3]
    if is_branch_checkout == "1":
        update_submodules()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

