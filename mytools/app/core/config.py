from dataclasses import dataclass
from pathlib import Path

import yaml

from .paths import now_stamp, norm_path


MODES = ("check_dataset", "train", "evaluate", "export", "predict")

DIR_FIELD_NAMES = {
    "data_dir",
    "dataset_dir",
    "save_model_dir",
    "save_inference_dir",
    "model_dir",
    "output",
}
FILE_FIELD_NAMES = {
    "character_dict_path",
    "weight_path",
    "pretrain_weight_path",
    "resume_path",
    "pretrained_model",
    "checkpoints",
    "input",
    "infer_img",
}
WEIGHT_FIELD_NAMES = {
    "weight_path",
    "pretrain_weight_path",
    "resume_path",
    "pretrained_model",
    "checkpoints",
}
IMAGE_FIELD_NAMES = {"input", "infer_img"}


@dataclass(frozen=True)
class FieldSpec:
    path: tuple[str, ...]
    section: str
    label: str
    editor_kind: str
    value_type: str
    browse_kind: str | None = None


def deep_copy(data):
    return yaml.safe_load(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))


def read_config(path, _model_type=None):
    path = Path(norm_path(path))
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def write_config(path, config, backup=True):
    path = Path(norm_path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = None
    if backup and path.exists():
        backup_path = path.with_suffix(path.suffix + f".{now_stamp()}.bak")
        backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return backup_path


def get_by_path(data, path, default=None):
    node = data
    for part in path:
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def set_by_path(data, path, value):
    if not path:
        return
    node = data
    for part in path[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[path[-1]] = value


def display_value(value):
    if value is None:
        return ""
    if value is True:
        return "True"
    if value is False:
        return "False"
    if isinstance(value, (dict, list)):
        return yaml.safe_dump(value, allow_unicode=True, sort_keys=False).strip()
    return str(value)


def _value_type(value):
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    if value is None:
        return "none"
    return "str"


def _editor_kind(path, value):
    key = path[-1]
    if key == "mode" and isinstance(value, str) and value in MODES:
        return "mode"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (list, dict)):
        return "yaml"
    return "text"


def _browse_kind(path, value):
    if isinstance(value, (dict, list)):
        return None
    key = path[-1].lower()
    if key in DIR_FIELD_NAMES or key.endswith("_dir"):
        return "dir"
    if key in FILE_FIELD_NAMES or key.endswith("_path") or "file" in key:
        return "file"
    raw = str(value or "").strip()
    if raw and not raw.startswith(("http://", "https://")) and (
        "/" in raw or "\\" in raw or Path(raw).suffix
    ):
        return "file"
    return None


def build_field_specs(data):
    specs = []

    def walk(path, value):
        if isinstance(value, dict) and value:
            for key, child in value.items():
                walk(path + (str(key),), child)
            return
        section = path[0] if len(path) > 1 else "Root"
        label = path[0] if len(path) == 1 else ".".join(path[1:])
        specs.append(
            FieldSpec(
                path=path,
                section=section,
                label=label,
                editor_kind=_editor_kind(path, value),
                value_type=_value_type(value),
                browse_kind=_browse_kind(path, value),
            )
        )

    for key, value in data.items():
        walk((str(key),), value)
    return specs


def parse_editor_value(raw, spec):
    text = str(raw).strip()
    if spec.editor_kind == "bool":
        return text.lower() in ("true", "1", "yes", "y", "on", "是")
    if spec.editor_kind == "mode":
        return text
    if spec.editor_kind == "yaml":
        if not text:
            return [] if spec.value_type == "list" else {}
        value = yaml.safe_load(text)
        expected = list if spec.value_type == "list" else dict
        if not isinstance(value, expected):
            raise ValueError(f"需要 {expected.__name__}，实际是 {type(value).__name__}")
        return value
    if spec.value_type == "int":
        return int(text)
    if spec.value_type == "float":
        return float(text)
    if spec.value_type == "none":
        return None if text.lower() in ("", "none", "null", "~") else text
    return text


def is_weight_field(path):
    return bool(path) and path[-1].lower() in WEIGHT_FIELD_NAMES


def is_image_field(path):
    return bool(path) and path[-1].lower() in IMAGE_FIELD_NAMES


def dataset_updates_for(model_type, out_dir):
    out = Path(norm_path(out_dir))
    updates = {
        ("Global", "dataset_dir"): str(out),
        ("Train", "dataset", "data_dir"): str(out),
        ("Train", "dataset", "label_file_list"): [str(out / "train.txt")],
        ("Eval", "dataset", "data_dir"): str(out),
        ("Eval", "dataset", "label_file_list"): [str(out / "val.txt")],
    }
    dict_path = out / "dict.txt"
    if model_type == "REC" and dict_path.exists():
        updates[("Global", "character_dict_path")] = str(dict_path)
    return updates
