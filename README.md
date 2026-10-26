# VesselGNN-Label

Inference code and pretrained models for automatic anatomical labelling of vessel centerline segments with graph neural networks (GNNs). Pretrained models are provided for three vascular territories:

| Region | Labels |
|---|---|
| Coronary arteries | 18 segments (LM, LAD p/m/d, LCx p/d, RI, D1–D3, OM1–OM3, Sep, RCA p/m/d, AM) |
| Intracranial arteries | 22 segments (ICA, M1, M2, A1, A2, AComm, VA, BA, P1, P2, PComm, OA; left/right) |
| Abdominal arteries | 5 segments (abdominal aorta, renal L/R, iliac L/R) |

This repository contains only the inference code and trained weights. No patient data is included.

## Method overview

The centerline segments of a case are joined into a graph by their connectivity, and a GNN predicts an anatomical label for each segment. Predictions are averaged over five cross-validation fold models.

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 1.13.1, PyTorch Geometric 2.6.1 and VTK 9.3.1. Inference runs on CPU or GPU.

## Input format

Each case is a folder containing one centerline file per segment (`.vtk` legacy or `.vtp` XML polydata) and a `topology.json` file:

```
case_001/
├── seg_A.vtk
├── seg_B.vtk
├── ...
└── topology.json
```

```json
{
  "segments": ["seg_A", "seg_B", "seg_C", "seg_D", "seg_E"],
  "connections": [["seg_A", "seg_B"], ["seg_A", "seg_C"], ["seg_A", "seg_D"], ["seg_A", "seg_E"]]
}
```

- `segments`: segment names (file names without extension). List the root of the vessel tree first, i.e. the most proximal segment where path distances start. During training the root was the left main (coronary), the left ICA (intracranial) and the abdominal aorta (abdominal). Segments are internally reordered so that parents precede their children.
- `connections`: `[parent, child]` pairs. Each pair links the last point of the parent to the first point of the child. A segment may have more than one parent (e.g. a communicating artery).
- Centerline points should be ordered from proximal to distal, in millimetres, in the LPS patient coordinate system (DICOM convention: +x patient left, +y posterior, +z superior), as used for training. Each case is translated so that its bounding-box minimum lies at the origin.

### Connectivity conventions

The models were trained on graphs with the connectivity below. Input topologies should follow the same conventions: a segment that borders several parent segments is connected to each of them.

| Region | Root | Connections (parent → child) |
|---|---|---|
| Coronary | LM | LM → LAD_p, LCx_p, RI; LAD_p → LAD_m, D1; LAD_m → LAD_d, D2; LAD_d → D3; LAD_p, LAD_m, LAD_d → Sep; LCx_p → LCx_d, OM1; LCx_d → OM2, OM3. Second tree: RCA_p → RCA_m → RCA_d; RCA_m → AM |
| Intracranial | ICA_L | ICA → M1, A1, PComm, OA; M1 → M2; A1 → A2, AComm (AComm joins both A1); VA_L, VA_R → BA; BA → P1; PComm → P1; P1 → P2 (left and right) |
| Abdominal | abdominal aorta | aorta → renal L/R, iliac L/R |

Predictions do not depend on the order in which connections are listed. For the intracranial model, the path-distance feature in the Circle of Willis loops depends on the traversal order, which was label-based during training, so intracranial results may differ slightly from those obtained with the training graph construction.

## Usage

```bash
# single case
python predict.py --region abdominal --input examples/synthetic_abdominal

# folder of case folders
python predict.py --region coronary --input path/to/cases --output predictions.csv
```

`--region` is one of `coronary`, `intracranial` or `abdominal`. The output CSV has one row per segment: `case, segment, predicted_label, probability`.

Example output for the included synthetic case:

```
synthetic_abdominal   seg_A   abdominal_aorta   1.000
synthetic_abdominal   seg_B   renal_left        0.910
synthetic_abdominal   seg_C   renal_right       1.000
synthetic_abdominal   seg_D   iliac_left        0.861
synthetic_abdominal   seg_E   iliac_right       0.998
```

From Python:

```python
from vessel_labelling import load_case, load_models, predict_case

models = load_models('coronary')
names, segments, connections = load_case('path/to/case')
predictions = predict_case(segments, connections, models, 'coronary')  # [(label, probability), ...]
```

## Repository structure

```
├── predict.py                  # command-line inference
├── vessel_labelling/
│   ├── config.py               # label sets and model settings per region
│   ├── graph.py                # graph construction and node features
│   ├── models.py               # GNN classifiers
│   └── inference.py            # I/O, model loading, prediction
├── weights/
│   ├── coronary/               # fold1.pt ... fold5.pt
│   ├── intracranial/
│   └── abdominal/
└── examples/
    └── synthetic_abdominal/    # analytic (synthetic) centerlines, not patient data
```

Each weight file contains only the model parameters and the feature standardisation statistics (mean and standard deviation) for that fold.
