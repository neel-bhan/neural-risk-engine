# Generated dataset workspace

M6 labels are produced by the trusted C++ pricing engine. No Python pricing formula or Python
dependency is involved. Generated CSV and manifest files belong under `data/generated/`, which is
ignored by Git.

Regenerate and independently verify the small evidence dataset from a clean checkout:

```bash
make clean
make CXXFLAGS='-std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror' dataset-small
make dataset-verify
```

`make dataset-reproduce` generates the small dataset twice and requires byte-identical CSV and
manifest files. The larger preset is deliberately optional because it is materially more
expensive:

```bash
make clean
make CXXFLAGS='-std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror' dataset-large
```

To change scale or tolerances, copy `data/config/m6-large.cfg`, edit the copy before running the
experiment, and use:

```bash
make dataset-large DATASET_CONFIG=path/to/config.cfg DATASET_OUTPUT=data/generated/name
```

The committed configuration is part of provenance. Do not tune a quality threshold after looking
at generated labels merely to hide a rejected row. Rejected rows stay in `labels.csv` with
`included_for_training=false` and explicit flags.

See [`DATASET_SCHEMA_V1.md`](DATASET_SCHEMA_V1.md) for the schema and
[`../docs/M6_DATASET_GENERATION.md`](../docs/M6_DATASET_GENERATION.md) for measured evidence.
