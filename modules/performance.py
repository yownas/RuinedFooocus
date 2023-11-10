import os
import shutil
import json

DEFAULT_PERFORMANCE_FILE = "settings/performance.default"
PERFORMANCE_FILE = "settings/performance.json"
NEWPERF = "Custom..."


def load_performance():
    perf_options = {}

    with open(DEFAULT_PERFORMANCE_FILE) as f:
        default_data = json.load(f)

    if not os.path.isfile(PERFORMANCE_FILE):
        shutil.copy(DEFAULT_PERFORMANCE_FILE, PERFORMANCE_FILE)
    else:
        with open(PERFORMANCE_FILE) as f:
            data = json.load(f)
            for name, settings in default_data.items():
                if name not in data:
                    data[name] = settings
            perf_options = data

        with open(PERFORMANCE_FILE, "w") as f:
            json.dump(perf_options, f, indent=2)

    return perf_options


def save_performance(perf_options):
    global PERFORMANCE, performance_options
    with open(PERFORMANCE_FILE, "w") as f:
        json.dump(perf_options, f, indent=2)
    PERFORMANCE = perf_options
    performance_options = {f"{k}": v for k, v in PERFORMANCE.items()}


def get_perf_options(name):
    return performance_options[name]


PERFORMANCE = load_performance()
performance_options = {f"{k}": v for k, v in PERFORMANCE.items()}
