import os


def get_project_root() -> str:
    return os.path.abspath(os.curdir)


def get_datasets_dir() -> str:
    return os.path.join(get_project_root(), "datasets")
