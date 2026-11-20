import os
from pathlib import Path


def get_all_files(directory, extension):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(os.path.join(root, filename))
    return files


def get_visual_paths(directory_path):
    return [str(path) for path in sorted(Path(directory_path).glob("*/*"))
            if path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}]
