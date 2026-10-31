# Maritime Benchmark v1.0

A 200-item, English-only, four-option maritime multiple-choice benchmark for evaluating
domain adaptation of small language models. The benchmark is rebuilt from published
primary sources and is designed for local evaluation on a 12 GB GPU.

> **Supersedes the legacy 163-item bilingual release.** The legacy MCQ scores are invalid:
> the old harness never inserted the four answer choices into the prompt. The legacy files
> are retained under `archive/legacy_163/` for provenance only. Do not use them.

## Overview

- **200 items**, English only, four options each.
- **160 test / 40 dev** items.
- Six categories: COLREGs, SOLAS, MARPOL, STCW, GMDSS, and Port/PSC.
- Every item carries `source`, `source_url`, and `evidence` strings.
- Primary metric: token-normalised option likelihood. Secondary metric: choices-in-prompt
  generation, with parse failures retained as `?`.
- The benchmark is **not contamination-free**; both audits are included.

## Composition

| Category | Full | Test | Dev |
|---|---:|---:|---:|
| COLREGs | 50 | 40 | 10 |
| SOLAS | 35 | 28 | 7 |
| MARPOL | 35 | 28 | 7 |
| STCW | 25 | 20 | 5 |
| GMDSS | 25 | 20 | 5 |
| Port/PSC | 30 | 24 | 6 |
| **Total** | **200** | **160** | **40** |

Split composition: 190 `public_core` items (150 test, 40 dev) and 10 `port_local` items
(all test). The full answer key is A 51 / B 49 / C 51 / D 49; the test key is
A 42 / B 38 / C 40 / D 40. The correct option is uniquely longest in 14/200 items (7.0%).

## Schema

```json
{
  "id": "COL-001",
  "category": "colregs",
  "question": "Under the COLREGs, what must every vessel maintain at all times?",
  "choices": ["A. A proper lookout by sight and hearing", "B. A visual watch only", "..."],
  "answer": "A",
  "correct_text": "A proper lookout by sight and hearing",
  "source": "COLREGs Rule 5",
  "source_url": "https://www.navcen.uscg.gov/navrules",
  "evidence": "Rule 5 requires a proper lookout by sight and hearing ...",
  "split": "public_core",
  "difficulty": "easy",
  "eval_split": "test",
  "meta": {"correct_len": 37, "option_lens": [37, 19, 18, 42],
           "correct_is_longest": false, "correct_is_unique_longest": false}
}
```

`split` is `public_core` or `port_local`; `eval_split` is `test` or `dev`.

## Baselines (160-item test split)

| Strategy | Accuracy |
|---|---:|
| Constant A | 26.25 |
| Constant B | 23.75 |
| Constant C | 25.00 |
| Constant D | 25.00 |
| First option | 26.25 |
| Longest option | 7.50 |
| Shortest option | 26.875 |
| Uniform random | 25.00 |

The strongest surface baseline is the shortest-option heuristic at 26.9%, not the
constant-letter heuristic. Report both when comparing models.

## Reference results

Token-normalised option likelihood on the 160-item test split. Dev accuracies are from
the 40-item dev split. These are the values reported in the paper.

| Configuration | Test acc. | Correct | Dev acc. |
|---|---:|---:|---:|
| Base | 33.8 | 54/160 | 37.5 |
| Pilot CPT | 49.4 | 79/160 | 55.0 |
| Pilot merge (α=0.60) | 45.0 | 72/160 | 50.0 |
| S-304 raw | 45.0 | 72/160 | 52.5 |
| S-304 α=0.30 | 36.9 | 59/160 | 47.5 |
| S-304 α=0.45 | 38.8 | 62/160 | 50.0 |
| S-304 α=0.60 | 41.9 | 67/160 | 47.5 |
| S-304 α=0.75 | 41.9 | 67/160 | 47.5 |
| S-608 raw | 51.2 | 82/160 | 57.5 |
| S-608 α=0.30 | 35.6 | 57/160 | 50.0 |
| S-608 α=0.45 | 40.0 | 64/160 | 47.5 |
| S-608 α=0.60 | 44.4 | 71/160 | 50.0 |
| S-608 α=0.75 | 48.8 | 78/160 | 50.0 |

Paired McNemar tests and Wilson confidence intervals are in the paper. The headline
includes the 10 `port_local` items; they do not drive it (public_core 51.3%,
port_local 50.0% for S-608 raw).

## Validation and contamination

- Items were authored and verified by the authors against the cited primary sources.
  **No independent domain expert was available**; ambiguous items were removed. This is
  single-annotator validation with a documented second pass, not independent expert
  adjudication.
- 13-gram audit: 0/173 auditable exact matches; 27 items are shorter than 13 tokens;
  all 622 candidate documents were exact-verified.
- 9-gram audit: 15/200 items have at least one exact 9-gram (10 answer/boundary,
  5 question-only); on the test split, 13 items overlap (10 answer/boundary,
  3 question-only), leaving 147 no-match items. All 1,018 candidate documents were
  exact-verified.
- The split is descriptive, not a causal estimate. The benchmark is not
  contamination-free.

## Evaluation

Primary protocol (option likelihood; choices are not shown as labels):

```bash
python evaluate.py \
  --model /path/to/model \
  --benchmark benchmark/maritime_bench_v1.json \
  --output runs/test.jsonl \
  --scoring likelihood \
  --eval-split test
```

Secondary protocol (choices in the prompt; parse failures retained):

```bash
python evaluate.py \
  --model /path/to/model \
  --benchmark benchmark/maritime_bench_v1.json \
  --output runs/test_generation.jsonl \
  --scoring generation \
  --eval-split test
```

Use `--scoring both` to write both records in one pass. The harness writes one JSONL
record per item and protocol; `pred` is `A`–`D` or `?`.

Analyse a completed run:

```bash
python analysis/analyze.py \
  --input-dir runs \
  --benchmark benchmark/maritime_bench_v1.json \
  --output runs/analysis.json
```

## Files

| Path | Contents |
|---|---|
| `benchmark/maritime_bench_v1.json` | 200-item benchmark (SHA-256 `ad69f775…18254`) |
| `benchmark/SOURCES.md` | Per-item source, URL, and evidence table |
| `benchmark/validate_benchmark.py` | Schema, key-balance, and cue audit |
| `evaluate.py` | Likelihood and generation harness |
| `analysis/analyze.py` | Accuracy, Wilson CIs, McNemar, category and contamination splits |
| `analysis/contamination.json` | 13-gram audit |
| `analysis/contamination_9gram.json` | 9-gram sensitivity audit |
| `archive/legacy_163/` | Superseded legacy release (invalid MCQ scores) |

## Reproducing

See `REPRODUCING.md` for the exact commands, hashes, and expected checks. The benchmark
SHA-256 is
`ad69f7755d3d5ff50617f8d551be5412f44ff29d8fc97aff82667cc667e18254`.
The verified test and dev result archives have SHA-256
`7a19244d58bdb44af4a7389fc629090b74e547c7a445c833f314679e64cfba13` and
`d5edba6127ec96d3c512d530da304072475f55f418bd1c3d53dbbc7e955d2085`.

## License

Apache 2.0. See `LICENSE`.
