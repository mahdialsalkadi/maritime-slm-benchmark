# Legacy 163-item release (invalid MCQ scores)

These files are retained only for provenance. **Do not use them for evaluation.**

The legacy harness never inserted the four answer choices into the MCQ prompt, so its
MCQ scores measured letter priors rather than maritime knowledge. The legacy benchmark
is also bilingual and includes a 15-item open-ended section that is not part of
Maritime Benchmark v1.0.

Use `benchmark/maritime_bench_v1.json` and `evaluate.py` instead.
