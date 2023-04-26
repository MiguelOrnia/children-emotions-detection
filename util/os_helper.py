import os


def get_path(target_path, file):
    absolute_path = os.path.dirname(file)
    relative_path = target_path
    full_path = os.path.join(absolute_path, relative_path)
    return full_path
