# Humanoid Vision

Detection and Tracking of Humans

## Getting Started

1. Clone this repository.

```
git clone https://github.com/howird/humanoid_vision.git
```

2. Install
   [uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods)

3. Create, activate, and sync your uv virtualenv

```
uv venv
source .venv/bin/activate
uv sync
uv pip install setuptools numpy torch
uv sync --no-build-isolation-package detectron2 --no-build-isolation-package neural-renderer-pytorch
```

## References

- Built off of the following amazing repositories:
  - [PHALP](https://github.com/brjathu/PHALP.git)
    - [deep_sort](https://github.com/nwojke/deep_sort)
  - [4D-Humans](https://github.com/shubham-goel/4D-Humans)
