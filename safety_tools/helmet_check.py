import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from common import (
    NMS_IOU,
    box_center,
    filter_detections,
    draw_box,
    draw_raw_detection_boxes,
    draw_unsafe_banner,
    int_box,
    load_model,
    overlap_ratio,
    parse_detections,
    person_head_region,
    point_in_box,
    print_json,
    resolve_device,
    run_detection,
    save_hazard_image,
    save_json,
    add_finished_at,
)


def best_helmet_in_region(helmets: List[Dict[str, Any]], region: List[float]) -> Optional[Dict[str, Any]]:
    candidates = [det for det in helmets if point_in_box(box_center(det["box"]), region)]
    if not candidates:
        return None
    return max(candidates, key=lambda det: det["confidence"])


def best_no_helmet_overlap(no_helmets: List[Dict[str, Any]], region: List[float], min_overlap: float) -> Optional[Dict[str, Any]]:
    candidates = [det for det in no_helmets if overlap_ratio(det["box"], region) >= min_overlap]
    if not candidates:
        return None
    return max(candidates, key=lambda det: (overlap_ratio(det["box"], region), det["confidence"]))


def evaluate_persons(
    detections: List[Dict[str, Any]],
    image_shape,
    no_helmet_overlap: float = 0.2,
) -> List[Dict[str, Any]]:
    persons = filter_detections(detections, "person")
    helmets = filter_detections(detections, "helmet")
    no_helmets = filter_detections(detections, "head_no_helmet")

    results = []
    associated_head_indices: set[int] = set()
    for person_index, person in enumerate(persons, start=1):
        person_id = person.get("track_id") if person.get("track_id") is not None else person_index
        head_region = person_head_region(person["box"], image_shape)
        helmet = best_helmet_in_region(helmets, head_region)
        no_helmet = best_no_helmet_overlap(no_helmets, head_region, no_helmet_overlap)
        associated_head_indices.update(
            index for index, head in enumerate(no_helmets)
            if overlap_ratio(head["box"], head_region) >= no_helmet_overlap
        )

        if no_helmet is not None:
            status = "no_helmet"
            confidence = no_helmet["confidence"]
        elif helmet is not None:
            status = "helmet"
            confidence = helmet["confidence"]
        else:
            # The configured safety rule treats a person without helmet evidence as unsafe.
            status = "no_helmet"
            confidence = person["confidence"]

        results.append(
            {
                "person_id": person_id,
                "track_id": person.get("track_id"),
                "box": int_box(person["box"]),
                "status": status,
                "confidence": round(float(confidence), 4),
                "detection_source": "person",
                "associated_helmet_box": int_box(helmet["box"]) if helmet else None,
                "associated_no_helmet_box": int_box(no_helmet["box"]) if no_helmet else None,
                "_head_region": head_region,
            }
        )

    # A head without an associated body is still evidence of an unhelmeted person.
    # Heads already associated above must not create a second record for the same person.
    next_person_id = max((int(item["person_id"]) for item in results), default=0) + 1
    for head_index, head in enumerate(no_helmets):
        if head_index in associated_head_indices:
            continue
        person_id = head.get("track_id") if head.get("track_id") is not None else next_person_id
        next_person_id = max(next_person_id, int(person_id)) + 1
        results.append({
            "person_id": person_id,
            "track_id": head.get("track_id"),
            "box": int_box(head["box"]),
            "status": "no_helmet",
            "confidence": round(float(head["confidence"]), 4),
            "detection_source": "head_no_helmet",
            "associated_helmet_box": None,
            "associated_no_helmet_box": int_box(head["box"]),
            "_head_region": head["box"],
        })
    return results


def dump_detections(detections: List[Dict[str, Any]]) -> None:
    print("[detections] class confidence xyxy", file=sys.stderr)
    for index, det in enumerate(detections, start=1):
        box = [round(float(v), 2) for v in det["box"]]
        print(
            f"[detections] #{index} {det['class_name']} conf={det['confidence']:.4f} box={box}",
            file=sys.stderr,
        )


def build_output(image_path: str, person_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    public_persons = []
    for person in person_results:
        item = dict(person)
        item.pop("_head_region", None)
        public_persons.append(item)

    no_helmet_count = sum(1 for p in public_persons if p["status"] == "no_helmet")
    unknown_count = sum(1 for p in public_persons if p["status"] == "unknown")
    return {
        "image": str(image_path),
        "total_persons": len(public_persons),
        "helmet_worn_count": sum(1 for p in public_persons if p["status"] == "helmet"),
        "no_helmet_count": no_helmet_count,
        "unknown_count": unknown_count,
        "persons": public_persons,
        "result": (
            "unsafe"
            if no_helmet_count
            else ("unknown" if not public_persons or unknown_count else "safe")
        ),
        "has_hazard": bool(no_helmet_count),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safety helmet detection for one image.")
    parser.add_argument("--weights", required=True, help="YOLO weights path, e.g. safety_best.pt")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output annotated image path")
    parser.add_argument("--json", help="Output JSON path. If omitted, JSON is printed to terminal")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=NMS_IOU, help="Framework NMS IoU threshold")
    parser.add_argument("--device", default=None, help="Inference device, e.g. 0 for GPU or cpu for CPU")
    parser.add_argument("--no-helmet-overlap", type=float, default=0.2, help="Minimum head_no_helmet overlap with head region")
    parser.add_argument("--dump-detections", action="store_true", help="Print model output boxes with class, confidence, and xyxy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {args.image}")

    model = load_model(args.weights)
    result = run_detection(model, image, conf=args.conf, iou=args.iou, device=args.device)
    # Model callbacks have applied class-aware NMS before these safety rules.
    detections = parse_detections(result)
    if args.dump_detections:
        dump_detections(detections)
    person_results = evaluate_persons(detections, image.shape, args.no_helmet_overlap)
    annotated_image = draw_raw_detection_boxes(image, detections)

    result_json = build_output(Path(args.image).name, person_results)
    result_json.update(frame_id=1, source_type="single_image")
    if result_json["has_hazard"]:
        unsafe_persons = [person for person in result_json["persons"] if person.get("status") == "no_helmet"]
        for person in unsafe_persons:
            draw_box(annotated_image, person.get("box"), "UNSAFE: NO HELMET", (0, 0, 255), 5)
        draw_unsafe_banner(annotated_image, ["人员未佩戴安全帽"])
        evidence = save_hazard_image(annotated_image, args.output, "helmet", 1)
        event = {
            "algorithm": "helmet",
            "source_type": "single_image",
            "frame_id": 1,
            "is_first_hazard": True,
            "hazard_timestamp": evidence["hazard_timestamp"],
            "hazard_image_path": evidence["hazard_image_path"],
            "hazard_count": len(unsafe_persons),
            "hazards": [{
                "code": "NO_HELMET",
                "check": "person_helmet",
                "message": "人员未佩戴安全帽",
                "count": len(unsafe_persons),
                "details": unsafe_persons,
            }],
        }
        result_json.update(
            is_first_hazard=True,
            hazard_timestamp=event["hazard_timestamp"],
            hazard_image_path=event["hazard_image_path"],
            hazard_count=event["hazard_count"],
            hazards=event["hazards"],
            hazard_event=event,
            hazard_events=[event],
            hazard_event_count=1,
        )
    else:
        result_json.update(hazard_events=[], hazard_event_count=0)
    result_json = add_finished_at(result_json)
    if args.json:
        save_json(result_json, args.json)
    else:
        print_json(result_json)


if __name__ == "__main__":
    main()
