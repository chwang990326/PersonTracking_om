from __future__ import annotations

import argparse
import base64
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib import error, request


DEFAULT_API_URL = "http://127.0.0.1:8000/third/face/verify"
DEFAULT_DATASET_DIR = "faceImage_test"
DEFAULT_GALLERY_DIR = "faceImage"
DEFAULT_SAMPLE_COUNT = 500
DEFAULT_SEED = 20260511
DEFAULT_TIMEOUT = 10.0
DEFAULT_REPORT_PATH = "results/face_verify_report.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test /third/face/verify face verification accuracy."
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="API endpoint URL")
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help="Image dataset directory. Images should be grouped by label folder.",
    )
    parser.add_argument(
        "--gallery-dir",
        default=DEFAULT_GALLERY_DIR,
        help="Known face gallery directory.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help="Number of images to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--report-path",
        default=DEFAULT_REPORT_PATH,
        help="JSON report output path.",
    )
    return parser.parse_args()


def load_known_person_ids(gallery_dir: Path) -> List[str]:
    if not gallery_dir.exists():
        raise FileNotFoundError(f"Known face gallery does not exist: {gallery_dir}")
    if not gallery_dir.is_dir():
        raise NotADirectoryError(f"Known face gallery is not a directory: {gallery_dir}")

    return sorted(path.name for path in gallery_dir.iterdir() if path.is_dir())


def collect_image_paths(dataset_dir: Path) -> List[Path]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_dir}")

    image_paths = [
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    image_paths.sort(key=lambda path: str(path).lower())
    return image_paths


def choose_samples(image_paths: List[Path], sample_count: int, seed: int) -> List[Path]:
    if sample_count <= 0:
        raise ValueError("sample-count must be greater than 0")
    if len(image_paths) < sample_count:
        raise ValueError(
            f"Not enough images: found {len(image_paths)}, requested {sample_count}"
        )

    rng = random.Random(seed)
    sampled = rng.sample(image_paths, sample_count)
    sampled.sort(key=lambda path: str(path).lower())
    return sampled


def post_json(api_url: str, payload: Dict[str, str], timeout: float) -> Dict[str, object]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        api_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {raw}") from exc
    except error.URLError as exc:
        raise ConnectionError(f"API is not reachable: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Response is not valid JSON: {raw}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Response JSON root is not an object: {data!r}")

    return data


def preflight_service(api_url: str, timeout: float) -> None:
    try:
        response = post_json(api_url, {"picBase64": ""}, timeout)
    except Exception as exc:
        raise RuntimeError(f"Service preflight failed: {exc}") from exc

    if "code" not in response or "msg" not in response:
        raise RuntimeError(f"Service response is missing code/msg: {response!r}")


def encode_image_to_base64(image_path: Path) -> str:
    with image_path.open("rb") as file_obj:
        return base64.b64encode(file_obj.read()).decode("utf-8")


def build_expected_result(label: str, known_person_ids: set) -> Tuple[str, str, str]:
    if label in known_person_ids:
        return "0", "success", label
    return "1202", "person not found", ""


def evaluate_response(
    expected_code: str,
    expected_msg: str,
    expected_person_id: str,
    response: Optional[Dict[str, object]],
    response_error: Optional[str],
) -> Dict[str, object]:
    actual_code = None
    actual_msg = None
    actual_person_id = None
    passed = False
    failure_reason = ""

    if response_error is not None:
        failure_reason = response_error
        return {
            "actual_code": actual_code,
            "actual_msg": actual_msg,
            "actual_person_id": actual_person_id,
            "passed": passed,
            "failure_reason": failure_reason,
        }

    assert response is not None

    actual_code_value = response.get("code")
    actual_msg_value = response.get("msg")
    data_value = response.get("data")

    actual_code = None if actual_code_value is None else str(actual_code_value)
    actual_msg = None if actual_msg_value is None else str(actual_msg_value)

    if isinstance(data_value, dict):
        person_id_value = data_value.get("personId")
        if person_id_value is not None:
            actual_person_id = str(person_id_value)

    if actual_code != expected_code:
        failure_reason = f"code_mismatch(expected={expected_code}, actual={actual_code})"
    elif actual_msg != expected_msg:
        failure_reason = f"msg_mismatch(expected={expected_msg}, actual={actual_msg})"
    elif expected_code == "0" and actual_person_id != expected_person_id:
        failure_reason = (
            f"person_id_mismatch(expected={expected_person_id}, actual={actual_person_id})"
        )
    else:
        passed = True

    return {
        "actual_code": actual_code,
        "actual_msg": actual_msg,
        "actual_person_id": actual_person_id,
        "passed": passed,
        "failure_reason": failure_reason,
    }


def safe_accuracy(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def format_accuracy(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def describe_failure_reason(reason: str) -> str:
    code_match = re.fullmatch(
        r"code_mismatch\(expected=(?P<expected>[^,]+), actual=(?P<actual>[^)]+)\)",
        reason,
    )
    if code_match:
        expected_code = code_match.group("expected")
        actual_code = code_match.group("actual")
        return f"Expected code {expected_code}, got {actual_code}"

    if reason.startswith("msg_mismatch("):
        return "The code matched but msg did not match."

    if reason.startswith("person_id_mismatch("):
        return "The API returned success but personId did not match the label."

    if reason.startswith("HTTP "):
        return "The API returned an HTTP error response."

    if "API is not reachable" in reason:
        return "The API is not reachable or the service is not running."

    if "Response is not valid JSON" in reason:
        return "The API response is not valid JSON."

    if "Response JSON root is not an object" in reason:
        return "The API response JSON root is not an object."

    return "Request or response handling failed."


def save_report(report_path: Path, report: Dict[str, object]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    gallery_dir = Path(args.gallery_dir)
    report_path = Path(args.report_path)

    try:
        known_person_ids_list = load_known_person_ids(gallery_dir)
        known_person_ids = set(known_person_ids_list)
        image_paths = collect_image_paths(dataset_dir)
        sampled_paths = choose_samples(image_paths, args.sample_count, args.seed)
        preflight_service(args.api_url, args.timeout)
    except Exception as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        return 1

    print(f"Known gallery IDs: {len(known_person_ids_list)}")
    print(f"Dataset image count: {len(image_paths)}")
    print(f"Sample count: {len(sampled_paths)}")
    print(f"Random seed: {args.seed}")
    print(f"API URL: {args.api_url}")

    sample_results: List[Dict[str, object]] = []
    failure_counter: Counter[str] = Counter()
    known_total = 0
    unknown_total = 0
    known_passed = 0
    unknown_passed = 0

    for index, image_path in enumerate(sampled_paths, start=1):
        label = image_path.parent.name
        expected_code, expected_msg, expected_person_id = build_expected_result(
            label,
            known_person_ids,
        )
        is_known_sample = expected_code == "0"
        if is_known_sample:
            known_total += 1
        else:
            unknown_total += 1

        response = None
        response_error = None
        try:
            pic_base64 = encode_image_to_base64(image_path)
            response = post_json(
                args.api_url,
                {"picBase64": pic_base64},
                args.timeout,
            )
        except Exception as exc:
            response_error = str(exc)

        evaluation = evaluate_response(
            expected_code=expected_code,
            expected_msg=expected_msg,
            expected_person_id=expected_person_id,
            response=response,
            response_error=response_error,
        )

        if evaluation["passed"]:
            if is_known_sample:
                known_passed += 1
            else:
                unknown_passed += 1
        else:
            failure_counter[str(evaluation["failure_reason"])] += 1

        sample_results.append(
            {
                "image_path": str(image_path),
                "label": label,
                "expected_code": expected_code,
                "expected_msg": expected_msg,
                "expected_person_id": expected_person_id,
                "actual_code": evaluation["actual_code"],
                "actual_msg": evaluation["actual_msg"],
                "actual_person_id": evaluation["actual_person_id"],
                "passed": evaluation["passed"],
                "failure_reason": evaluation["failure_reason"],
            }
        )

        if index % 50 == 0 or index == len(sampled_paths):
            print(f"Completed {index}/{len(sampled_paths)}")

    total_samples = len(sample_results)
    total_passed = known_passed + unknown_passed
    total_failed = total_samples - total_passed

    overall_accuracy = safe_accuracy(total_passed, total_samples)
    known_accuracy = safe_accuracy(known_passed, known_total)
    unknown_accuracy = safe_accuracy(unknown_passed, unknown_total)

    report = {
        "generated_at": datetime.now().isoformat(),
        "api_url": args.api_url,
        "dataset_dir": str(dataset_dir),
        "gallery_dir": str(gallery_dir),
        "report_path": str(report_path),
        "sample_count": args.sample_count,
        "seed": args.seed,
        "timeout": args.timeout,
        "dataset_image_count": len(image_paths),
        "sampled_image_count": total_samples,
        "known_person_ids": known_person_ids_list,
        "metrics": {
            "total_samples": total_samples,
            "known_samples": known_total,
            "unknown_samples": unknown_total,
            "passed_samples": total_passed,
            "failed_samples": total_failed,
            "overall_accuracy": overall_accuracy,
            "known_accuracy": known_accuracy,
            "unknown_accuracy": unknown_accuracy,
        },
        "failure_reason_counts": dict(sorted(failure_counter.items())),
        "samples": sample_results,
    }

    try:
        save_report(report_path, report)
    except Exception as exc:
        print(f"[Error] Failed to save report: {exc}", file=sys.stderr)
        return 1

    print("")
    print("===== Test Summary =====")
    print(f"Total samples: {total_samples}")
    print(f"Known samples: {known_total}")
    print(f"Unknown samples: {unknown_total}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Overall accuracy: {format_accuracy(overall_accuracy)}")
    print(f"Known accuracy: {format_accuracy(known_accuracy)}")
    print(f"Unknown accuracy: {format_accuracy(unknown_accuracy)}")
    print(f"Report path: {report_path}")

    if failure_counter:
        print("")
        print("Failure reasons:")
        for reason, count in sorted(failure_counter.items()):
            print(f"- {reason}: {count}")
            print(f"  Description: {describe_failure_reason(reason)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
