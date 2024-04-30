import pathlib

from transfergraph.dataset.task import TaskType


def get_root_path_string() -> str:
    return pathlib.Path(__file__).parent.parent.parent.resolve().__str__()


def get_directory_experiments(task_type: TaskType) -> str:
    return f"{get_root_path_string()}/resources/experiments/{task_type.value}"
