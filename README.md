# Trading

NBA data and modeling workspace.

## Environment

Create the local conda environment under the project directory:

```bash
/share/apps/software/anaconda3/bin/conda env create \
  --prefix /home/ayf7/trading/env \
  --file /home/ayf7/trading/environment.yml
```

Activate it:

```bash
source /share/apps/software/anaconda3/bin/activate /home/ayf7/trading/env
```

If you prefer to create the base Python environment first and install from the spec afterward:

```bash
/share/apps/software/anaconda3/bin/conda create \
  --prefix /home/ayf7/trading/env \
  python=3.12.3

source /share/apps/software/anaconda3/bin/activate /home/ayf7/trading/env
conda env update --prefix /home/ayf7/trading/env --file /home/ayf7/trading/environment.yml
```

The environment is ignored by git via `env/`. The `environment.yml` file is the tracked, reproducible description of what should be installed.

Conda is used for Python, `sqlite`, and `pip`. Python analysis libraries are installed through pip to avoid slow conda solves on the cluster.

## VS Code SQLite

The `alexcvzz.vscode-sqlite` extension needs a `sqlite3` executable. After activating the environment, confirm the path:

```bash
which sqlite3
```

It should be:

```text
/home/ayf7/trading/env/bin/sqlite3
```

In VS Code settings JSON, set:

```json
"sqlite.sqlite3": "/home/ayf7/trading/env/bin/sqlite3"
```

Then open:

```text
/home/ayf7/trading/data/artifacts/nba_raw.sqlite
```

## Data

See [data/README.md](data/README.md) for raw, core, and feature database artifacts.
