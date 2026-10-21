# Class-balanced frame filtering for text-referred objects in simulated videos

Semi-online referring video object segmentation (see "A Filtering Framework for Semi-online Referring Video Object Segmentation", 2025) must decide which frames to process when compute is limited. Filtering frames in which the text-referred object is visible is the key step, so a better frame filter can keep more useful frames under the same budget. This experiment asks whether a learned, class-balanced frame scorer beats a standard change-threshold keyframe rule.

**Question:** Can a class-balanced LightGBM frame filter keep more frames where a text-referred object is visible than a standard change-threshold keyframe filter under the same frame budget?

**Dataset:** Simulated videos generated in code (numpy); no download. This is not the paper's own dataset (same_as_paper = false). Train and test videos use different random seeds.

**Baseline:** a change-threshold keyframe filter that keeps a frame when its scene descriptor changes more than a threshold tuned on training videos, capped at the per-video budget. This stands in for the paper's unspecified filter.

**Improved model:** a class-balanced LightGBM per-frame scorer using causal (past-only) rolling features, with hyper-parameters chosen by grouped cross-validation on the training videos. It keeps the top-scoring frames within each video under the same budget.

**Metrics:** main = recall of referred-object frames kept on the test set; secondary = precision, F1, and frames kept per video.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results are written to results.json (and figures/recall_bar.png).

## Reference

This project is motivated by the following paper:

> X Hu, H Neumann, J Lang (2025). *A Filtering Framework for Semi-online Referring Video Object Segmentation*. Proceedings of the 33rd ACM International. https://dl.acm.org/doi/abs/10.1145/3746027.3755018

Paper author: Prof. Heiko Neumann, Ulm University (heiko.neumann@uni-ulm.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
