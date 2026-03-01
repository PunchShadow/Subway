# Repository Guidelines

## Project Structure & Module Organization
- `shared/` contains core graph data structures, CUDA kernels, partitioning, and utilities used by all applications (e.g., `graph.cu`, `gpu_kernels.cu`, `subgraph_generator.cu`).
- `subway/` contains algorithm entry points and kernels for BFS, CC, SSSP, SSWP, and PageRank in sync/async variants (e.g., `bfs-async.cu`).
- `tools/` provides the `converter` utility for turning `.el`/`.wel` graphs into binary CSR (`.bcsr`/`.bwcsr`).
- Root `Makefile` orchestrates builds; `README.md` describes input formats and runtime flags.

## Build, Test, and Development Commands
- `make` builds shared objects, all Subway apps, and tools using `nvcc` and `g++`.
- `make -C tools` builds only the graph `converter`.
- `make -C shared` or `make -C subway` builds only libraries or app objects.
- `make clean` removes object files and compiled binaries.
- Run an app from the repo root, for example: `./sssp-async --input path/to/graph.bwcsr --source 10` or `./bfs-sync --input path/to/graph.bcsr`.

## Coding Style & Naming Conventions
- Language: C++11/CUDA (`.cu`, `.cuh`, `.cpp`, `.hpp`).
- Indentation is predominantly tabs in `.cu` files; follow the existing style of the file you touch.
- Brace style is K&R with the opening brace on a new line for functions.
- Executable names follow `<algo>-<sync|async>` (e.g., `pr-async`).

## Testing Guidelines
- There is no formal test framework or coverage target in this repo.
- `shared/test.cu` is a simple helper, not a runnable test suite.
- Validate changes by running algorithms on small graphs and checking for expected output/stability.
- If you add tests, document the command in this file and wire it into the root `Makefile`.

## Commit & Pull Request Guidelines
- Commit messages are short, imperative, and sentence-case (examples from history: “Add PageRank algorithm”, “Update README.md”).
- Keep commits focused; avoid mixing formatting-only changes with functional changes.
- For PRs, include a clear description, the commands you ran, and any required GPU/CUDA assumptions (e.g., `-arch=sm_60`).

## Configuration Tips
- The CUDA architecture flag is set in `NFLAGS` (`-arch=sm_60`) across Makefiles; update it for your GPU if needed.
- Input graph formats are extension-sensitive: `.el`/`.wel` for text, `.bcsr`/`.bwcsr` for binary.
