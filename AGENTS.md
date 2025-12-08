# Repository Guidelines

## Project Structure & Module Organization
- Code lives in `src/geo_forge/`, with NuScenes helpers in `nuscenes.py` and preprocessing pipeline components in `preprocess/` (`preprocess.py`, `sam3_preprocessor.py`, adapters, and configs in `dataclass.py`). 
- Package is Python 3.12+, packaged via `pyproject.toml`; outputs such as masks and visualizations default to `src/geo_forge/preprocess/datasets/` unless overridden.
- No formal `tests/` directory yet; keep data-heavy artifacts out of the repo and use `.gitignore` for large outputs.

## Setup, Build, and Development Commands
- Install deps with uv: `uv sync` (resolves `uv.lock`, fetches SAM/HF and NuScenes dependencies). 
- Run the NuScenes sample iterator demo: `uv run python -m geo_forge.nuscenes`.
- Run the preprocessing demo (process first mini-scene, writes masks/visualizations): `uv run python -m geo_forge.preprocess.preprocess`.
- Use `uv run python` for ad-hoc scripts to ensure the locked environment is respected.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indents; keep functions small and typed (use `typing` hints and dataclasses for configs).
- Prefer descriptive, lower_snake_case names for functions/variables and UpperCamelCase for classes; keep module-level constants upper_snake.
- When adding utilities, prefer pure functions inside existing modules before creating new top-level packages; colocate dataset-specific adapters under `preprocess/`.
- Include brief docstrings for public functions; avoid heavy inline comments unless clarifying non-obvious logic.

## Testing Guidelines
- No automated suite is present yet; add `pytest` tests under a new `tests/` directory as you extend functionality.
- For data-dependent code, stub or fixture minimal frames/point clouds rather than hitting full NuScenes to keep tests light.
- Before opening a PR, at minimum run the demo scripts above to confirm end-to-end preprocessing still works.

## Data & Configuration Tips
- Set `NUSCENES_DATAROOT` to point to your local dataset root; defaults to `/data/nuscenes` if unset.
- SAM/SAM3 models expect a CUDA device by default (`SAM3PreprocessorConfig.device`), so override to `cpu` when necessary.
- Outputs include `.npy` masks and `.jpg` visualizations; keep paths configurable via `SAM3PreprocessorConfig.output_dir`.

## Commit & Pull Request Guidelines
- Git history mixes imperative summaries and Conventional Commit prefixes (`feat:`, `refactor:`); prefer concise, imperative subjects (≤72 chars) and add a type prefix when it clarifies intent.
- In PRs, include: what changed, how to run/verify (commands above), dataset/configs used (e.g., camera names, scene IDs), and sample output paths or counts.
- Attach screenshots or file listings for generated visualizations when relevant; link issues or TODOs you addressed.
