## Contributing to MSDYOLO

Issues and pull requests are welcome. Please keep changes focused on rotated
object detection, DOTA preparation, training reliability, or documentation
that makes the cloud workflow reproducible.

## Repository boundaries

- `msdyolo/` is the only implementation package. The root command files are
  compatibility wrappers and should stay small.
- `configs/` contains model and experiment configuration; `scripts/` contains
  cloud and operational entry points; `tests/` contains deterministic checks.
- `dataset/`, `runs/`, `training.log`, `wandb/`, and model checkpoints are
  runtime artifacts. They are ignored and must not be committed.
- DOTA labels use pixel coordinates in each patch (`0..1024`), followed by the
  class name and difficult flag. Do not normalize them to `[0, 1]`.

## Local development

From a fresh checkout:

```bash
python -m pip install -e .
python -m pytest -q
bash -n scripts/setup.sh scripts/ddp_train.sh
git diff --check
```

Use the fixture-based baseline command in the README for a CPU smoke test.
Full training requires a CUDA host and a prepared DOTA v1.5 dataset; do not
download that dataset as part of a unit test.

## Pull requests

1. Create a focused branch from `master`.
2. Keep commits small and describe the user-visible reason for the change.
3. Run the relevant tests and include the exact commands in the PR body.
4. For cloud-training changes, report the config, device, dataset layout, and
   the first-epoch diagnostic. A real full-MSD run must show `match > 0`.
5. Never include cloud credentials, DOTA files, generated logs, or weights.

For bug reports, include a minimal reproduction, the Python/PyTorch/CUDA
versions, and the relevant log excerpt. Redact credentials and private paths.

## License

By contributing, you agree that your contributions will be licensed under the
[GNU GPL v3](LICENSE).
