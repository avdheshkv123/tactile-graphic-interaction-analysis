import json


def load_regions(region_json_path):
    with open(region_json_path, "r") as f:
        regions = json.load(f)

    return regions


def get_region(x, y, regions):
    for region_name, (x1, y1, x2, y2) in regions.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return region_name

    return None


def build_region_sequence(cleaned_points, regions):
    sequence = []
    previous_region = None

    for point in cleaned_points:
        x = point["x"]
        y = point["y"]

        current_region = get_region(x, y, regions)

        if current_region and current_region != previous_region:
            sequence.append(current_region)
            previous_region = current_region

    return sequence