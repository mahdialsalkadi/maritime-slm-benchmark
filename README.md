# Maritime SLM Benchmark: 163-Question Bilingual Evaluation Suite

This repository provides the complete dataset and evaluation code for the **163-Question Bilingual Maritime Language Model Benchmark** introduced in:

> **Reliable Continued Pre-Training of a Maritime Language Model: Capability Drift and Weight-Space Recovery**  
> *Under peer review for ICAICE / DSIS 2026.*

---

## Overview

Adapting small language models (SLMs) to maritime operations requires rigorous evaluation across both general reasoning and specialized operational knowledge. This benchmark evaluates models in both **English** and **Bahasa Malaysia** (the primary administrative language of Port Klang Authority and Jabatan Laut Malaysia).

The benchmark comprises **163 test items**:
1. **148 Multiple-Choice Questions (MCQ)** covering six core operational and regulatory categories.
2. **15 Free-Form Open-Ended Questions** evaluating contextual regulatory reasoning and domain terminology recovery.

---

## Category Breakdown

| Category | Questions ($n$) | Description | Primary Sources |
|:---|:---:|:---|:---|
| **Navigation Rules** | 30 | International Regulations for Preventing Collisions at Sea (COLREGs 1972), steering, sailing rules, lights/shapes, sound signals. | IMO COLREGs, UK MCA |
| **SOLAS & MARPOL** | 20 | Safety of Life at Sea (fire safety, life-saving appliances) & Marine Pollution (Annex I--VI, sulfur caps, ballast water). | IMO Conventions, Port State Control |
| **Notice-to-Mariners (NTM)** | 20 | Interpretation of navigational warnings, coordinate reading, temporary hazard avoidance, buoyage status. | Jabatan Laut Malaysia (JLM), UKHO |
| **Port Operations** | 39 | Vessel Traffic Services (VTS), pilotage boarding, berthing guidelines, marine circulars (20 OOD items + 19 Port Klang Authority notices). | Port Klang Authority (PKA), Marine Dept. |
| **Emergency Procedures** | 20 | Global Maritime Distress and Safety System (GMDSS), Search and Rescue (SAR), distress alerts (MAYDAY, PAN PAN), fire protocols. | IAMSAR Manual, MMEA, IMO |
| **Maritime Bahasa Malaysia** | 19 | Standard Malaysian maritime operational terms, navigation circular terminology, vessel inspection vocabulary. | JLM circulars, MMEA reports |
| **Total Multiple-Choice** | **148** | Complete domain assessment with 4-choice format (A, B, C, D). | |
| **Open-Ended Evaluation** | **15** | Scenario-based drafting and incident response scored by domain keyword recovery. | Maritime handbooks, casualty reports |
| **Total Benchmark Suite** | **163** | | |

---

## Dataset Format

The dataset is provided in `maritime_eval_163.json`. Each MCQ item follows this structure:

```json
{
  "id": "nav_01",
  "q": "Under Rule 15 of COLREGs, when two power-driven vessels are crossing so as to involve risk of collision, which vessel must keep out of the way?",
  "options": {
    "A": "The vessel which has the other on her own port side",
    "B": "The vessel which has the other on her own starboard side",
    "C": "Both vessels must alter course to starboard",
    "D": "The vessel with the lower maneuvering capability"
  },
  "answer": "B",
  "category": "navigation_rules"
}
```

---

## Quickstart & Evaluation

### Installation
```bash
git clone https://github.com/mahdialsalkadi/maritime-slm-benchmark.git
cd maritime-slm-benchmark
pip install -r requirements.txt
```

### Run Evaluation
```bash
python evaluate.py \
  --model "Qwen/Qwen3.5-0.8B-Base" \
  --benchmark "maritime_eval_163.json" \
  --output "eval_results.json"
```

---

## Reference Baseline Results

As reported in the paper, evaluated across reference models:

| Benchmark Dimension | Base (0.8B) | Raw CPT (S-608) | Merged ($\alpha=0.45$) | $\Delta$ vs. Base |
|:---|:---:|:---:|:---:|:---:|
| Emergency Procedures (%) | 25.0 | 55.0 | **50.0** | **+25.0** |
| Maritime Bahasa Malaysia (%) | 26.3 | 0.0 | **36.8** | **+10.5** |
| Notice-to-Mariners (NTM) (%) | 55.0 | 20.0 | **65.0** | **+10.0** |
| Navigation Rules / COLREGs (%) | **66.7** | 26.7 | 46.7 | -20.0 |
| SOLAS / MARPOL Recall (%) | 15.0 | 20.0 | **20.0** | **+5.0** |
| Port Operations (%) | **51.3** | 46.2 | 48.7 | -2.6 |
| **Maritime MCQ Overall (%)** | 43.2 | 30.4 | **45.9** | **+2.7** |
| **Maritime Open-Ended (/10)** | 3.69 | 3.73 | **4.30** | **+16.5%** |

---

## License

This benchmark dataset and code are released under the [Apache 2.0 License](LICENSE).
