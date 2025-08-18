import argparse
import concurrent.futures
import os
import pathlib
import sys

from tqdm import tqdm


def execute_file(file, timeout=None):
    """
    Execute a Python file in a separate process using sys.executable.
    Returns a tuple with the file path, exit code, stdout, and stderr.
    """
    import subprocess
    import time

    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, file],
        # [sys.executable, '-c', 'import time; time.sleep(1)'],
        timeout=timeout,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=dict(os.environ, PYTHONUTF8="1")
    )
    ptime = time.perf_counter() - start
    return file, ptime, result.returncode, result.stdout, result.stderr


def gather_files(root, args):
    def select(p):
        if p.parent == root:
            return False
        for parent in p.relative_to(root).parents:
            if parent.name.startswith('.'):
                return False
        return True

    # If neither option is provided, default to running all subdirectories.
    if not args.all and not args.dirs:
        args.all = True

    if args.all:
        files = list(filter(select, root.rglob('*.py')))
    else:
        # Calculate absolute paths for the selected directories.
        selected_dirs = {(root / d).resolve() for d in args.dirs}
        files = []
        for p in root.rglob('*.py'):
            if not select(p):
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


def run_sequential(files, root):
    results = {}
    with tqdm(total=len(files), unit='files') as pbar:
        for file in files:
            pbar.set_description(f'Running {(file.relative_to(root))}')
            pbar.update()

            results[file] = execute_file(file)
            file, utime, returncode, stdout, stderr = results[file]

            if returncode == 0:
                if stdout:
                    print(stdout)
            else:
                print(f'{(file.relative_to(root))} failed with exit code {returncode} after '
                      f'{utime:.1f} s.')
                if stderr:
                    print(stderr)

    total = pbar.format_dict['elapsed']
    print(f'Processed {len(files)} scripts in {total:.1f} seconds.')
    return results


def run_parallel(files, root, workers, timeout):
    results = {}
    with (
        concurrent.futures.ProcessPoolExecutor(workers) as executor,
        tqdm(total=len(files), unit='file') as pbar
    ):
        futures = {executor.submit(execute_file, file, timeout): file for file in files}
        pbar.set_description(f'Running {len(files)} scripts')
        pbar.update(n=0)

        for future in concurrent.futures.as_completed(futures):
            results[future] = future.result()
            file, ptime, returncode, stdout, stderr = results[future]

            if returncode == 0:
                pbar.set_description(f'{(file.relative_to(root))} ran successfully in '
                                     f'{ptime:.1f} s')
                if stdout:
                    print(stdout)
            else:
                pbar.set_description(f'{(file.relative_to(root))} failed with exit code '
                                     f'{returncode} after {ptime:.1f} s')
                if stderr:
                    print(stderr)

            pbar.update()

    total = pbar.format_dict['elapsed']
    ptimes = sum(result[1] for result in results.values())
    print(f'Processed {len(files)} scripts with {workers} processes in {total:.1f} seconds. '
          f'Speedup: {round((1 - total / ptimes) * 100)} %')
    return results


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
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run scripts in parallel processes."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of workers if --parallel is True"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help="Timeout in seconds for jobs."
    )
    args = parser.parse_args()

    files = gather_files(root := pathlib.Path(__file__).parent.resolve(), args)
    if not files:
        print('No files found. Exiting.')
        sys.exit(0)
    else:
        print('Found the following files:')
        print('\n'.join(str(file) for file in files))

    if args.parallel:
        results = run_parallel(files, root, args.workers, args.timeout)
    else:
        results = run_sequential(files, root)

    return results


if __name__ == '__main__':
    main()
