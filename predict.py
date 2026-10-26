"""Label vessel centerline segments with a pretrained GNN.

Usage:
    python predict.py --region coronary --input path/to/case_dir
    python predict.py --region coronary --input path/to/dir_of_cases --output predictions.csv
"""
import argparse
import csv
import os

import torch

from vessel_labelling import REGIONS, load_case, load_models, predict_case
from vessel_labelling.inference import TOPOLOGY_FILE


def find_cases(path):
    """A case is a folder containing topology.json; `path` may be a case or a folder of cases."""
    if os.path.isfile(os.path.join(path, TOPOLOGY_FILE)):
        return [path]
    cases = sorted(os.path.join(path, d) for d in os.listdir(path)
                   if os.path.isfile(os.path.join(path, d, TOPOLOGY_FILE)))
    if not cases:
        raise FileNotFoundError(f'No case folders with {TOPOLOGY_FILE} found under {path}')
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--region', required=True, choices=sorted(REGIONS))
    parser.add_argument('--input', required=True, help='case folder, or folder of case folders')
    parser.add_argument('--output', default='predictions.csv', help='output CSV file')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    models = load_models(args.region, args.device)

    rows = []
    for case_dir in find_cases(args.input):
        names, segments, connections = load_case(case_dir)
        preds = predict_case(segments, connections, models, args.region, args.device)
        case = os.path.basename(os.path.normpath(case_dir))
        for name, (label, prob) in zip(names, preds):
            rows.append([case, name, label, f'{prob:.4f}'])
            print(f'{case}\t{name}\t{label}\t{prob:.3f}')

    with open(args.output, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['case', 'segment', 'predicted_label', 'probability'])
        writer.writerows(rows)
    print(f'Saved {len(rows)} predictions to {args.output}')


if __name__ == '__main__':
    main()
