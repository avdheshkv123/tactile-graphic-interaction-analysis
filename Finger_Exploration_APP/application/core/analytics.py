import json
from collections import defaultdict


def generate_analysis(sequence):
    visit_count = defaultdict(int)

    for region in sequence:
        visit_count[region] += 1

    transitions = []

    for i in range(len(sequence) - 1):
        transitions.append(f"{sequence[i]}->{sequence[i + 1]}")

    result = {
        "sequence": sequence,
        "unique_sequence": list(dict.fromkeys(sequence)),
        "visits": dict(visit_count),
        "transitions": transitions,
        "num_steps": len(sequence)
    }

    return result


def save_analysis_json(output_path, analysis_data):
    with open(output_path, "w") as f:
        json.dump(analysis_data, f, indent=4)