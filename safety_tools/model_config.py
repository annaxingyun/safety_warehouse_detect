"""Shared model defaults without importing the inference runtime during server startup."""

DEFAULT_MODEL_NAME = "all_best.pt"
INFERENCE_IMAGE_SIZE = 640
NMS_IOU = 0.7
CLASS_NAMES = dict(enumerate((
    "person", "helmet", "head_no_helmet", "forklift", "fence", "boom", "large cable",
    "4D distribution box", "6D distribution box", "200kV Transformer", "400kV Transformer",
    "400kV distribution box", "500kV distribution box", "630kV Transformer", "630kV Transformer NX2",
    "safety officer", "crane",
)))
CLASS_IDS = {name: class_id for class_id, name in CLASS_NAMES.items()}
MATERIAL_CLASS_NAMES = tuple(CLASS_NAMES[index] for index in range(6, 15))


def normalize_class_name(name: str) -> str:
    """Accept checkpoint names with either spaces or underscores."""
    return str(name).strip().casefold().replace(" ", "_")
