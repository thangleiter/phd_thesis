import argparse
import concurrent.futures
import pathlib
import subprocess
import sys

import tqdm
from tqdm import tqdm


def execute_file(file):
    """
    Execute a Python file in a separate process using sys.executable.
    Returns a tuple with the file path, exit code, stdout, and stderr.
    """
    result = subprocess.run(
        # [sys.executable, file],
        [sys.executable, '-c', 'import time; time.sleep(1)'],
        capture_output=True,
        text=True
    )
    return file, result.returncode, result.stdout, result.stderr


def gather_files(root, args):
    # If neither option is provided, default to running all subdirectories.
    if not args.all and not args.dirs:
        args.all = True

    if args.all:
        # Gather all Python files in subdirectories (excluding those directly in the root).
        files = [p for p in root.rglob('*.py') if p.parent != root]
    else:
        # Calculate absolute paths for the selected directories.
        selected_dirs = {(root / d).resolve() for d in args.dirs}
        files = []
        for p in root.rglob('*.py'):
            # Exclude files in the root.
            if p.parent == root:
                continue
            # Check if the file is located in one of the selected subdirectories.
            # We use p.relative_to(selected_dir) to see if p is under that directory.
            for d in selected_dirs:
                try:
                    # If no exception, then p is under d.
                    p.relative_to(d)
                    files.append(p)
                    break  # Found a matching directory; no need to check others.
                except ValueError:
                    continue

    return files


def main():
    # Set up CLI argument parsing.
    parser = argparse.ArgumentParser(
        description="Execute plotting scripts in selected subdirectories."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--all",
        action="store_true",
        help="Run all scripts (default if no other option is provided)."
    )
    group.add_argument(
        "--dirs",
        nargs="+",
        help="List of subdirectories (relative to the project root) to run."
    )
    args = parser.parse_args()

    files = gather_files(root := pathlib.Path(__file__).parent.resolve(), args)

    # Execute files in parallel using a process pool.
    with (
            concurrent.futures.ProcessPoolExecutor() as executor,
            tqdm(total=len(files), unit='files') as pbar
    ):
        futures = {executor.submit(execute_file, file): file for file in files}
        for future in concurrent.futures.as_completed(futures):
            file, returncode, stdout, stderr = future.result()

            if returncode == 0:
                pbar.set_description(f'{(file.relative_to(root))} ran successfully.')
                if stdout:
                    print(stdout)
            else:
                print(f"\nFile {file} failed with exit code {returncode}.")
                pbar.set_description(f'{(file.relative_to(root))} failed with exit code '
                                     f'{returncode}.')
                if stderr:
                    print(stderr)

            pbar.update()


if __name__ == '__main__':
    main()
