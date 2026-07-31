def override_dict(source: dict, target: dict) -> dict:
    return _override_dict(source, target, path="")


def check_conflict(source: dict, target: dict):
    _check_conflict(source, target, path="")


def _override_dict(source: dict, target: dict, path: str) -> dict:
    unknown_keys = sorted(set(source) - set(target))
    if unknown_keys:
        raise ValueError(f"unknown arguments: {_format_keys(unknown_keys, path)}")

    result = dict(target)
    for name, value in source.items():
        target_value = target[name]
        child_path = _join_path(path, name)

        if isinstance(value, dict) and isinstance(target_value, dict):
            result[name] = _override_dict(value, target_value, child_path)
        elif isinstance(value, dict) or isinstance(target_value, dict):
            raise ValueError(f"argument type mismatch: {child_path}")
        else:
            result[name] = value

    return result


def _check_conflict(source: dict, target: dict, path: str):
    unknown_keys = sorted(set(source) - set(target))
    if unknown_keys:
        raise ValueError(f"unknown arguments: {_format_keys(unknown_keys, path)}")

    for name, value in source.items():
        target_value = target[name]
        child_path = _join_path(path, name)

        if isinstance(value, dict) and isinstance(target_value, dict):
            _check_conflict(value, target_value, child_path)
        elif isinstance(value, dict) or isinstance(target_value, dict):
            raise ValueError(f"argument type mismatch: {child_path}")
        elif value != target_value:
            raise ValueError(
                f"argument mismatch: {child_path}={value!r}, target has {target_value!r}"
            )


def _format_keys(keys: list[str], path: str) -> str:
    return ", ".join(_join_path(path, key) for key in keys)


def _join_path(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key
