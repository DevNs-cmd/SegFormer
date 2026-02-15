🚜 Terrain Segmentation for Robust Off-Road Navigation
🚀 Executive Summary

This project delivers a domain-generalized terrain segmentation system designed specifically for off-road autonomous navigation. Unlike traditional semantic segmentation models that optimize only for mean IoU, this system converts pixel-wise predictions into planner-ready traversability cost maps, enabling risk-aware decision-making under severe domain shift.

The pipeline is engineered for robustness, deployment feasibility, and direct integration into navigation stacks.

🧠 Problem Statement

Off-road environments are significantly more complex than structured road scenes. Terrain appearance varies drastically across:

Weather conditions

Illumination changes

Geographic locations

Sensor noise and motion artifacts

Core Challenge

Accurately classify terrain types (drivable, loose surface, obstacle, etc.) at the pixel level in unstructured environments while maintaining reliability under domain shift.

Why Synthetic-to-Real Generalization Fails

Synthetic datasets often lack:

Real sensor noise

Illumination extremes

Motion blur artifacts

Natural texture variability

As a result, models may overfit to synthetic color/texture statistics and degrade significantly in real-world conditions.

Why Mean IoU Is Not Enough

Two models can have similar mean IoU yet behave very differently in safety-critical scenarios.

A small false negative on an obstacle can cause mission failure.

Navigation systems require risk-aware outputs, not just segmentation accuracy.

This project reframes segmentation as a decision-support module, not just a vision task.

🔬 Core Contributions
1️⃣ Domain Generalization by Design

Aggressive train-time augmentation simulates real-world variability:

Brightness/contrast shifts

Gaussian noise

Motion blur

Color jitter

Perspective distortion

Random scaling & cropping

Horizontal flips

Optional synthetic shadows

Objective: reduce dependency on synthetic-only image statistics.

2️⃣ Hybrid Loss for Robust Overlap Learning
𝐿
𝑜
𝑠
𝑠
=
0.5
⋅
𝐶
𝐸
+
0.3
⋅
𝐷
𝑖
𝑐
𝑒
+
0.2
⋅
𝐹
𝑜
𝑐
𝑎
𝑙
Loss=0.5⋅CE+0.3⋅Dice+0.2⋅Focal

CrossEntropy → stable multi-class optimization

Dice Loss → improves region overlap quality

Focal Loss → emphasizes hard pixels and minority obstacle classes

This balances global accuracy and safety-critical precision.

3️⃣ Multi-Scale + Flip Test-Time Augmentation (TTA)

Inference across multiple scales with horizontal-flip averaging:

Improves robustness to object scale variation

Reduces boundary ambiguity

Stabilizes predictions under domain shift

4️⃣ Traversability Cost Mapping (Navigation-Ready Output)

Segmentation masks are transformed into configurable terrain cost maps:

Class-to-cost dictionary maps terrain labels to normalized traversal risk

Produces planner-compatible heatmaps

Direct integration with A*, D*, or MPC-based planners

This bridges perception and decision-making.

📊 Performance Benchmarks

Replace TBD with actual experimental results before submission.

Metric	Value
Mean IoU	TBD
Per-class IoU	TBD
Pixel Accuracy	TBD
Inference Time (ms/image)	TBD
FPS	TBD
Model Size (MB)	TBD
Parameter Count	TBD
🏗 Architecture & Design Rationale
Why SegFormer (Instead of UNet)

Hierarchical transformer encoder

Efficient MLP decoder

Large receptive field without heavy high-resolution decoding

Better modeling of global terrain context

Transformer-based backbones help capture long-range spatial relationships critical in large, texture-ambiguous terrain regions.

Why Aggressive Augmentation

Domain shift is the primary failure mode in synthetic-to-real transfer.
Augmentation explicitly injects variability observed in field conditions.

Why Post-Processing

Morphological closing

Small-region removal

These reduce noisy isolated predictions and improve mask coherence before cost-map conversion.

⚙ Deployment & Real-Time Considerations
Memory Control

Adjustable input resolution

Configurable batch size

Mixed precision (torch.amp)

Edge Device Suitability

SegFormer-B2/B3 variants

Optional TTA disablement

Resolution scaling

Speed–Accuracy Tradeoff Modes
Mode	Configuration
Fast	Single-scale, no TTA
Robust	Multi-scale + flip TTA
🧭 Decision Intelligence Layer
Traversability Mapping

Converts semantic output → normalized traversal risk

Produces interpretable heatmaps

Planner Integration

Cost maps can directly feed:

A*

D*

MPC cost terms

Enables risk-aware trajectory optimization instead of shortest-path-only behavior.

System Integration

Designed to fuse with:

Localization

Depth estimation

Control modules

Full autonomous navigation stacks

📁 Project Structure
terrain_segmentation/
  configs/config.yaml
  models/segformer.py
  utils/dataset.py
  utils/augmentations.py
  utils/losses.py
  utils/metrics.py
  utils/postprocessing.py
  utils/traversability.py
  utils/visualization.py
  train.py
  evaluate.py
  inference.py
  validate_env.py
  requirements.txt
  README.md


Modular, reproducible, and configuration-driven.

📂 Dataset Layout
<dataset_root>/
  train/
    images/
    masks/
  val/
    images/
    masks/
  test/
    images/
    masks/


Masks must be single-channel class-index maps

Paths configured via configs/config.yaml

No hardcoded dataset paths

🛠 Setup
cd terrain_segmentation
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

🚂 Training
python train.py --config configs/config.yaml


Outputs:

outputs/checkpoints/best_model.pt

TensorBoard logs in outputs/logs/

📈 Evaluation
python evaluate.py --config configs/config.yaml --split val
python evaluate.py --config configs/config.yaml --split test


Outputs:

Metrics YAML

Confusion matrix

Worst IoU examples

🔍 Inference
python inference.py --config configs/config.yaml --input /path/to/images --output outputs/inference


Outputs:

Predicted masks

Overlay visualizations

Traversability cost maps

🧪 Suggested Ablation Studies

Hybrid loss vs CE-only

With vs without domain randomization

With vs without TTA

With vs without post-processing

SegFormer-B2 vs B3

Report:

Mean IoU

Per-class IoU (especially obstacles)

Inference latency

🔮 Future Extensions

Uncertainty estimation for safety thresholds

ONNX / TensorRT export pipeline

Temporal consistency for video

Online domain adaptation

Planner-in-the-loop trajectory-level evaluation
