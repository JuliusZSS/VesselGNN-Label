"""Load centerline segments, build the graph and predict an anatomical label per segment."""
import glob
import json
import os

import numpy as np
import torch
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from .config import NUM_NODE_FEATURES, REGIONS
from .graph import build_graph
from .models import GCN, GraphSAGE

CENTERLINE_EXTENSIONS = ('.vtk', '.vtp')
TOPOLOGY_FILE = 'topology.json'
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'weights')


def read_centerline(path):
    """Return the (N, 3) points of a centerline stored as legacy .vtk or XML .vtp polydata."""
    reader = vtk.vtkXMLPolyDataReader() if path.lower().endswith('.vtp') else vtk.vtkPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    points = reader.GetOutput().GetPoints()
    if points is None or points.GetNumberOfPoints() < 2:
        raise ValueError(f'{path}: centerline must contain at least 2 points')
    return vtk_to_numpy(points.GetData()).astype(np.float64)


def _segment_file(case_dir, name):
    for ext in CENTERLINE_EXTENSIONS:
        path = os.path.join(case_dir, name + ext)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f'No {"/".join(CENTERLINE_EXTENSIONS)} file for segment "{name}" in {case_dir}')


def load_case(case_dir):
    """Load a case folder described by its topology.json.

    Returns (names, segments, connections), with segments in the order listed in topology.json
    and connections as (parent_index, child_index) pairs in the order listed.
    """
    topo_path = os.path.join(case_dir, TOPOLOGY_FILE)
    with open(topo_path) as fh:
        topo = json.load(fh)
    names = list(topo['segments'])
    index = {n: i for i, n in enumerate(names)}
    unknown = {n for pair in topo['connections'] for n in pair} - index.keys()
    if unknown:
        raise ValueError(f'{topo_path}: connections refer to unlisted segment(s) {sorted(unknown)}')
    segments = [read_centerline(_segment_file(case_dir, n)) for n in names]
    return names, segments, [(index[p], index[c]) for p, c in topo['connections']]


def load_models(region, device='cpu'):
    """Load the five cross-validation fold models (and their feature statistics) for a region."""
    cfg = REGIONS[region]
    arch = {'GCN': GCN, 'GraphSAGE': GraphSAGE}[cfg['model']]
    models = []
    for path in sorted(glob.glob(os.path.join(WEIGHTS_DIR, region, 'fold*.pt'))):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model = arch(NUM_NODE_FEATURES, cfg['hidden_dim'], len(cfg['labels']), cfg['num_layers'])
        model.load_state_dict(ckpt['state_dict'])
        models.append((model.to(device).eval(), ckpt['mean'].to(device), ckpt['std'].to(device)))
    if not models:
        raise FileNotFoundError(f'No weights found in {os.path.join(WEIGHTS_DIR, region)}')
    return models


@torch.no_grad()
def predict_case(segments, connections, models, region, device='cpu'):
    """Predict a label for each segment by averaging the segment-level class probabilities of
    all fold models. Returns a list of (label, probability) aligned with `segments`."""
    cfg = REGIONS[region]
    labels = cfg['labels']
    data, order = build_graph(segments, connections, cfg.get('double_secondary_root_edges', False))
    data = data.to(device)
    batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

    probs = 0
    for model, mean, std in models:
        _, region_out, _ = model((data.x - mean) / std, data.edge_index, batch, data.region_batch)
        probs = probs + region_out.exp()
    probs = (probs / len(models)).cpu().numpy()

    result = [None] * len(segments)
    for k, s in enumerate(order):
        c = int(probs[k].argmax())
        result[s] = (labels[c], float(probs[k, c]))
    return result
