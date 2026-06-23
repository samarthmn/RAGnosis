# RAGnosis Dataset Explorer

An interactive, single-file visualizer for this synthetic FHIR dataset — see every
resource type, its schema, and **how the resources link to each other**.

## The dataset

Synthetic FHIR R4 patient records (Massachusetts population), generated with **Synthea**:

- Generator: https://github.com/synthetichealth/synthea
- Configured via the Synthea Patient Generator (SPT) customizer: https://synthetichealth.github.io/spt/#/customizer

The data is entirely synthetic — no real patients. It lives as newline-delimited
JSON (`.ndjson`) files in `dataset/`, one resource type per file, with expected
record counts in [`metadata.json`](metadata.json).

> **Note:** `dataset/` is **git-ignored** — it's several GB, too large for GitHub.
> Regenerate it locally with the steps below.

## Generating the dataset

The dataset is produced by [Synthea](https://github.com/synthetichealth/synthea)
in **bulk FHIR (NDJSON)** mode. Quickest path:

1. **Design a population** with the SPT customizer
   (https://synthetichealth.github.io/spt/#/customizer) — pick the state, population size, and modules, then download the generated `synthea.properties`.

2. **Run Synthea** with NDJSON output enabled:

   ```bash
   git clone https://github.com/synthetichealth/synthea
   cd synthea
   # one resource type per .ndjson file (bulk export)
   ./run_synthea -c /path/to/synthea.properties \
       --exporter.fhir.bulk_data=true \
       --exporter.baseDirectory=./output \
       -p 1000
   ```

   Equivalent settings in `src/main/resources/synthea.properties`:

   ```properties
   exporter.fhir.export=true
   exporter.fhir.bulk_data=true
   exporter.baseDirectory=./output
   ```

3. **Place the output** so the files land in `data/dataset/*.ndjson`
   (Synthea writes them under `output/fhir/`):

   ```bash
   mkdir -p /path/to/data/dataset
   cp output/fhir/*.ndjson /path/to/data/dataset/
   ```

Counts will vary with population size and seed — `metadata.json` reflects the
reference run used here (1,158 patients, Massachusetts).

## Run it

```bash
./serve.sh
```

This serves the folder locally and opens the explorer in your browser. It then
**auto-loads `dataset/` with no manual selection**. Press `Ctrl-C` to stop.

> Set a custom port with `PORT=9000 ./serve.sh`.

## How it works

Everything is one self-contained `index.html` (vanilla JS + canvas — no frameworks,
no build step). The multi-gigabyte files are never loaded fully into the browser:

- **Schema + reference graph** — reads only the head of each file (HTTP Range
  requests) and scales counts to the totals in `metadata.json`.
- **Record browsing** — streams records on demand, paginated, so a 1 GB file
  browses as smoothly as a small one.
- **Follow a reference** — clicking a `Type/id` reference streams the target file
  with early-exit to locate that exact record.

## What you can explore

1. **Overview** — all resource types with record counts and on-disk sizes.
2. **Relationship Graph** — an interactive graph of how resource types reference
   each other (drag, hover to isolate, click to drill in).
3. **Type detail** — incoming/outgoing references, a coverage-ranked field schema,
   and a lazy record browser with a collapsible JSON tree where every reference is
   clickable.
