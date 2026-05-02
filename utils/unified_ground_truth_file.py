import os
import json
import cv2
from typing import Dict, List, Optional


# ═════════════════════════════════════════════════════════════════════
# CONFIGURATION — UPDATE THESE PATHS
# ═════════════════════════════════════════════════════════════════════

ANOMALY_TEST_TXT = r"C:\Users\Kameshwaran\Downloads\pes\Capstone\dataset\UCF_Crimes\UCF_Crimes\Anomaly_Detection_splits\Anomaly_Test.txt"

UCA_JSON_FILES = [
    r"C:\Users\Kameshwaran\Downloads\pes\Capstone\dataset\UCFCrime_Train.json",
    r"C:\Users\Kameshwaran\Downloads\pes\Capstone\dataset\UCFCrime_Test.json",
    r"C:\Users\Kameshwaran\Downloads\pes\Capstone\dataset\UCFCrime_Val.json",
]

VIDEO_ROOT = r"C:\Users\Kameshwaran\Downloads\pes\Capstone\dataset\UCF_Crimes\UCF_Crimes\Videos"

OUTPUT_PATH = r"C:\Users\Kameshwaran\Downloads\pes\Capstone\dataset\unified_ground_truth.json"

CATEGORY_OUTPUT_DIR = os.path.join(
    os.path.dirname(OUTPUT_PATH),
    "categorywise"
)


# ═════════════════════════════════════════════════════════════════════
# STRICT CATEGORY MAPPING (14 CLASSES)
# ═════════════════════════════════════════════════════════════════════

FOLDER_TO_CATEGORY = {
    "Abuse": "Abuse",
    "Arrest": "Arrest",
    "Arson": "Arson",
    "Assault": "Assault",
    "Burglary": "Burglary",
    "Explosion": "Explosion",
    "Fighting": "Fighting",
    "Normal": "Normal Activity",
    "Normal_Videos_event": "Normal Activity",
    "RoadAccidents": "Road Accident",
    "Robbery": "Robbery",
    "Shooting": "Shooting",
    "Shoplifting": "Shoplifting",
    "Stealing": "Stealing",
    "Vandalism": "Vandalism",
}


def infer_category_from_name(video_name: str) -> str:
    for folder, cat in FOLDER_TO_CATEGORY.items():
        if video_name.startswith(folder):
            return cat
    return "Normal Activity"


def normalize_name(name: str) -> str:
    name = os.path.splitext(name)[0]
    name = name.replace(".mp4", "").replace(".avi", "")
    return name


def parse_anomaly_test_txt(txt_path: str) -> Dict[str, Dict]:
    videos = {}

    with open(txt_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            path_part = parts[0].replace("\\", "/")
            filename = os.path.basename(path_part)
            video_name = normalize_name(filename)

            frame_nums = []
            for p in parts[1:]:
                try:
                    frame_nums.append(int(float(p)))
                except ValueError:
                    continue

            intervals = []
            i = 0
            while i + 1 < len(frame_nums):
                s, e = frame_nums[i], frame_nums[i + 1]
                if s >= 0 and e >= 0 and e > s:
                    intervals.append((s, e))
                i += 2

            is_anomaly = len(intervals) > 0
            if not is_anomaly:
                if "Normal" in video_name:
                    is_anomaly = False
                elif any(frame_nums) and all(f == -1 for f in frame_nums):
                    is_anomaly = False
                else:
                    is_anomaly = not video_name.startswith("Normal")

            videos[video_name] = {
                "raw_line": line,
                "filename": filename,
                "folder_path": path_part,
                "frame_intervals": intervals,
                "is_anomaly": is_anomaly,
                "binary": 1 if is_anomaly else 0,
                "category": infer_category_from_name(video_name),
            }

    return videos


def load_all_uca_captions(json_paths: List[str]) -> Dict[str, Dict]:
    merged = {}
    for path in json_paths:
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found")
            continue
        with open(path) as f:
            data = json.load(f)
        print(f"  Loaded {len(data)} entries from {os.path.basename(path)}")
        merged.update(data)
    return merged


def find_uca_match(video_name: str, uca_data: Dict) -> Optional[str]:
    if video_name in uca_data:
        return video_name

    no_x264 = video_name.replace("_x264", "")
    if no_x264 in uca_data:
        return no_x264

    base = "_".join(video_name.split("_")[:-1])
    if base and base in uca_data:
        return base

    for key in uca_data:
        if normalize_name(key) == normalize_name(video_name):
            return key

    return None


def find_video_file(video_name: str, folder_path: str, video_root: str) -> Optional[str]:
    candidates = []

    if folder_path:
        candidates.append(os.path.join(video_root, folder_path))

    for ext in [".mp4", ".avi", ".mkv"]:
        candidates.append(os.path.join(video_root, video_name + ext))

    category_folder = video_name.split("_")[0]
    if video_name.startswith("Normal"):
        category_folder = "Normal"

    for ext in [".mp4", ".avi", ".mkv"]:
        candidates.append(
            os.path.join(video_root, category_folder, video_name + ext)
        )

        if not video_name.endswith("_x264"):
            candidates.append(
                os.path.join(video_root, category_folder, video_name + "_x264" + ext)
            )

    for path in candidates:
        if os.path.exists(path):
            return path

    return None

def populate_frame_intervals(gt_data):
    """
    Convert uca_timestamps (seconds) → frame_intervals (frame numbers).
    This is the missing step causing AUC = 0%.
    """
    for vname, vinfo in gt_data.items():
        fps = vinfo.get("fps", 30.0)
        total_frames = vinfo.get("total_frames", 0)
        uca_ts = vinfo.get("uca_timestamps", [])
        binary = vinfo.get("binary_label", 0)

        if binary == 0:
            # Normal video: no anomaly frames
            vinfo["frame_intervals"] = []
            vinfo["has_frame_gt"] = True
            continue

        if not uca_ts:
            # Anomaly video with no timestamps — fallback
            vinfo["frame_intervals"] = [[0, total_frames - 1]]
            vinfo["has_frame_gt"] = False
            continue

        # Convert seconds → frame numbers
        intervals = []
        for start_sec, end_sec in uca_ts:
            start_frame = int(start_sec * fps)
            end_frame = min(int(end_sec * fps), total_frames - 1)
            if end_frame > start_frame:
                intervals.append([start_frame, end_frame])

        # Merge overlapping intervals
        if intervals:
            intervals.sort(key=lambda x: x[0])
            merged = [intervals[0]]
            for s, e in intervals[1:]:
                if s <= merged[-1][1] + 1:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            intervals = merged

        vinfo["frame_intervals"] = intervals
        vinfo["has_frame_gt"] = len(intervals) > 0

    return gt_data

def get_video_metadata(video_path: str) -> Dict:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / max(fps, 1)
    cap.release()

    return {
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_sec": round(duration, 2),
    }


def main():
    print("=" * 70)
    print("AgentVigil — Ground Truth Builder")
    print("=" * 70)

    print("\n[1] Parsing anomaly test file...")
    test_videos = parse_anomaly_test_txt(ANOMALY_TEST_TXT)

    anomalous = sum(1 for v in test_videos.values() if v["is_anomaly"])
    normal = len(test_videos) - anomalous

    print(f"Videos found: {len(test_videos)}")
    print(f"Anomalous: {anomalous}")
    print(f"Normal: {normal}")

    print("\n[2] Loading UCA captions...")
    uca_data = load_all_uca_captions(UCA_JSON_FILES)

    unified = {}
    missing_videos = []

    print("\n[3] Merging datasets...")

    for video_name, info in test_videos.items():
        entry = {
            "video_name": video_name,
            "filename": info["filename"],
            "binary_label": info["binary"],
            "is_anomaly": info["is_anomaly"],
            "category": info["category"],
            "frame_intervals": info["frame_intervals"],
            "has_frame_gt": len(info["frame_intervals"]) > 0,
            "uca_key": None,
            "uca_timestamps": [],
            "uca_sentences": [],
            "has_captions": False,
            "video_path": None,
            "fps": None,
            "total_frames": None,
            "duration_sec": None,
        }

        uca_key = find_uca_match(video_name, uca_data)
        if uca_key:
            uca_info = uca_data[uca_key]
            entry["uca_key"] = uca_key
            entry["uca_timestamps"] = uca_info.get("timestamps", [])
            entry["uca_sentences"] = uca_info.get("sentences", [])
            entry["has_captions"] = True

        video_path = find_video_file(
            video_name,
            info.get("folder_path", ""),
            VIDEO_ROOT
        )

        if video_path:
            entry["video_path"] = video_path
            entry.update(get_video_metadata(video_path))
        else:
            missing_videos.append(video_name)

        unified[video_name] = entry

    # print("\n[4] Converting timestamps → frame intervals...")
    unified = populate_frame_intervals(unified)
    print("\n[4] Saving unified ground truth...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(unified, f, indent=2)

    print("Unified JSON saved")

    print("\n[5] Creating category-wise files...")
    os.makedirs(CATEGORY_OUTPUT_DIR, exist_ok=True)

    category_data = {}
    for vid, data in unified.items():
        cat = data["category"]
        category_data.setdefault(cat, {})[vid] = data

    for cat, data in category_data.items():
        safe_cat = cat.replace(" ", "_")
        path = os.path.join(CATEGORY_OUTPUT_DIR, f"{safe_cat}.json")

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"{safe_cat}.json → {len(data)} videos")

    print("\nDone.")
    print(f"Unified file: {OUTPUT_PATH}")
    print(f"Category files: {CATEGORY_OUTPUT_DIR}")

    if missing_videos:
        print("\nMissing videos:")
        for v in missing_videos[:10]:
            print(v)


if __name__ == "__main__":
    main()