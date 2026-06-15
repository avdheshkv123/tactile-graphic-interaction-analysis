import os
from datetime import datetime


def create_run_folder(base_output_folder="outputs"):
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    run_folder = os.path.join(base_output_folder, f"run_{timestamp}")
    json_folder = os.path.join(run_folder, "json")
    plots_folder = os.path.join(run_folder, "plots")
    models_folder = os.path.join(run_folder, "models")
    logs_folder = os.path.join(run_folder, "logs")

    os.makedirs(json_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)
    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    return {
        "run_folder": run_folder,
        "json_folder": json_folder,
        "plots_folder": plots_folder,
        "models_folder": models_folder,
        "logs_folder": logs_folder,
    }