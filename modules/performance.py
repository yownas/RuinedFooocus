import os
import shutil
import json
from csv import DictReader, reader

DEFAULT_PERFORMANCE_FILE = "performance.default"
PERFORMANCE_FILE = "performance.json"


def load_performance():
    perf_options = {}

    if not os.path.isfile(PERFORMANCE_FILE):
        shutil.copy(DEFAULT_PERFORMANCE_FILE, PERFORMANCE_FILE)

    with open(PERFORMANCE_FILE) as f:
        data = json.load(f)
        for name, settings in data.items():
            perf_options[name] = settings

    return perf_options


def save_performance(perf_options):
    with open(PERFORMANCE_FILE, "w") as f:
        json.dump(perf_options, f, indent=2)


PERFORMANCE = load_performance()
performance_options = {f"{k}": v for k, v in PERFORMANCE.items()}
