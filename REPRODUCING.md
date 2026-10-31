# Reproducing Maritime Benchmark v1.0

This file records the commands and checks used for the released benchmark and the
paper's maritime evaluation. The evaluation was run on an RTX 5070 with 12 GB VRAM under
WSL2; the harness also runs on CPU for smoke tests.

## 1. Verify the release

```bash
sha256sum benchmark/maritime_bench_v1.json
# ad69f7755d3d5ff50617f8d551be5412f44ff29d8fc97aff82667cc667e18254

python benchmark/validate_benchmark.py
# Expected: 200 items, 0 errors, 0 warnings
```

The validator checks the schema, the 160/40 split, category counts, answer-key balance,
duplicate IDs/questions, option-label consistency, and surface cues.

## 2. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`evaluate.py` imports `torch` and `transformers` lazily, so the parser and scoring
functions can be tested without a GPU stack. `analysis/analyze.py` uses `numpy`; the
Wilson intervals and exact McNemar tests are implemented without SciPy (SciPy is used
only if present).

## 3. Run the primary protocol

```bash
python evaluate.py \
  --model /path/to/model \
  --model-name my-model \
  --benchmark benchmark/maritime_bench_v1.json \
  --output runs/my-model/test_likelihood.jsonl \
  --scoring likelihood \
  --eval-split test
```

The likelihood prompt is the question followed by each option text; the four choices
are **not** shown as labelled alternatives. The model's score for an option is the
length-normalised log-likelihood of that continuation.

## 4. Run the secondary protocol

```bash
python evaluate.py \
  --model /path/to/model \
  --model-name my-model \
  --benchmark benchmark/maritime_bench_v1.json \
  --output runs/my-model/test_generation.jsonl \
  --scoring generation \
  --eval-split test \
  --max-new-tokens 8
```

All four labelled choices are shown. A generation that cannot be mapped to A–D is
retained as `?` and counted as a parse failure; the analysis reports the rate.

## 5. Analyse

```bash
python analysis/analyze.py \
  --input-dir runs/my-model \
  --benchmark benchmark/maritime_bench_v1.json \
  --output runs/my-model/analysis.json \
  --eval-split test
```

The analysis reports overall and per-category accuracy with Wilson 95% CIs, macro and
balanced accuracy, constant-letter/first/longest/shortest baselines, paired exact
McNemar tests, and Holm-corrected category tests. Add
`--contamination analysis/contamination_9gram.json` for the descriptive 9-gram split.

## 6. Expected reference values

The paper's 13 configurations are listed in `README.md`. For the 160-item test split:

- Base: 54/160 = 33.8%.
- Step-608 raw CPT: 82/160 = 51.2%.
- Step-608 α=0.45 merge: 64/160 = 40.0%.
- Strongest surface baseline: shortest option, 43/160 = 26.875%.

Paired McNemar, step-608 raw vs. base: b=32, c=4, p=1.9×10⁻⁶. Step-608 raw vs.
α=0.45: b=20, c=2, p=1.2×10⁻⁴. Step-608 raw vs. α=0.75: b=6, c=2, p=0.289.

## 7. Contamination audits

The audits were run against the cleaned maritime corpus used for CPT. They are included
as `analysis/contamination.json` (13-gram) and `analysis/contamination_9gram.json`
(9-gram). Summary:

| Audit | Verified matches | Auditable | Notes |
|---|---:|---:|---|
| 13-gram | 0 | 173/200 | 27 items shorter than 13 tokens; 622 candidate documents exact-verified |
| 9-gram | 15/200 | 200/200 | 10 answer/boundary, 5 question-only; 1,018 candidate documents exact-verified |

The 9-gram split is descriptive, not a causal contamination estimate. The benchmark is
not contamination-free.

## 8. Provenance and validation

`benchmark/SOURCES.md` lists the primary source, URL, and evidence string for every
item. Items were authored and verified by the authors against those sources. No
independent domain expert was available; ambiguous items were removed. This is
single-annotator validation with a documented second pass, not independent expert
adjudication.

## 9. Result archives

The verified paper-run archives are:

- Test: `7a19244d58bdb44af4a7389fc629090b74e547c7a445c833f314679e64cfba13`
- Dev: `d5edba6127ec96d3c512d530da304072475f55f418bd1c3d53dbbc7e955d2085`

Each archive contains 13 configuration files, one JSONL record per item and protocol,
and a benchmark-hash check. The analysis reproduces the paper's tables from those
records.
