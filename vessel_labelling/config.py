"""Per-region label sets and model configurations (index order = model output order)."""

REGIONS = {
    'coronary': {
        'labels': ['LM', 'LAD_p', 'LAD_m', 'LAD_d', 'LCx_p', 'LCx_d', 'RI', 'D1', 'D2', 'D3',
                   'OM1', 'OM2', 'OM3', 'Sep', 'RCA_p', 'RCA_m', 'RCA_d', 'AM'],
        'model': 'GCN', 'hidden_dim': 64, 'num_layers': 2,
        # The right coronary tree (second root) was joined with a doubled edge during training.
        'double_secondary_root_edges': True,
    },
    'intracranial': {
        'labels': ['ICA_L', 'ICA_R', 'M1_L', 'M1_R', 'M2_L', 'M2_R', 'A1_L', 'A1_R', 'A2_L', 'A2_R',
                   'AComm', 'VA_L', 'VA_R', 'BA', 'P1_L', 'P1_R', 'P2_L', 'P2_R',
                   'PComm_L', 'PComm_R', 'OA_L', 'OA_R'],
        'model': 'GraphSAGE', 'hidden_dim': 64, 'num_layers': 4,
    },
    'abdominal': {
        'labels': ['abdominal_aorta', 'renal_left', 'renal_right', 'iliac_left', 'iliac_right'],
        'model': 'GraphSAGE', 'hidden_dim': 64, 'num_layers': 4,
    },
}

NUM_NODE_FEATURES = 9
