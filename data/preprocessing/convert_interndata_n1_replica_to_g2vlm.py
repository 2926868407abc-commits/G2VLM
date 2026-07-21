#!/usr/bin/env python3
"""Convert InternData-N1 replica_d435i trajectories to G2VLM joint-train parquet."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


INSTRUCTION_KEYS = (
    "revised_sub_instruction",
    "sub_instruction",
    "sum_instruction",
    "instruction",
    "language_instruction",
    "task",
)


def read_jsonl(path):
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def matrix3_to_4x4(values):
    mat = np.asarray(values, dtype=np.float32).reshape(3, 3)
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = mat
    return out.reshape(-1).tolist()


def matrix4(values):
    return np.asarray(values, dtype=np.float32).reshape(4, 4).reshape(-1).tolist()


def matrix4_np(values):
    return np.asarray(values, dtype=np.float32).reshape(4, 4)


def camera_center_candidates(pose_values):
    pose = matrix4_np(pose_values)
    candidates = [pose[:3, 3]]
    try:
        rot = pose[:3, :3]
        trans = pose[:3, 3]
        candidates.append(-(rot.T @ trans))
    except Exception:
        pass
    return candidates


def project_world_point_to_image(point_world, pose_values, intrinsic_values, width, height):
    pose = matrix4_np(pose_values)
    point_h = np.ones(4, dtype=np.float32)
    point_h[:3] = np.asarray(point_world, dtype=np.float32)
    intrinsic = np.asarray(intrinsic_values, dtype=np.float32).reshape(3, 3)

    camera_points = []
    try:
        camera_points.append(np.linalg.inv(pose) @ point_h)
    except np.linalg.LinAlgError:
        pass
    camera_points.append(pose @ point_h)

    best = None
    for camera_point in camera_points:
        x, y, z = camera_point[:3]
        if not np.isfinite(camera_point[:3]).all() or abs(float(z)) < 1e-6:
            continue
        if z < 0:
            continue
        u = intrinsic[0, 0] * x / z + intrinsic[0, 2]
        v = intrinsic[1, 1] * y / z + intrinsic[1, 2]
        if not np.isfinite([u, v]).all():
            continue
        inside = 0 <= u < width and 0 <= v < height
        score = 1 if inside else 0
        candidate = (score, float(u), float(v))
        if best is None or candidate[0] > best[0]:
            best = candidate

    if best is None:
        return None

    _, u, v = best
    u = max(0.0, min(float(width - 1), u))
    v = max(0.0, min(float(height - 1), v))
    return [round(u / max(width, 1) * 1000), round(v / max(height, 1) * 1000)]


def infer_goal_pixel(frame_table, frame_ids, image_list):
    final_idx = frame_ids[-1]
    target_points = []
    target_points.extend(camera_center_candidates(frame_table["observation.camera_extrinsic"][final_idx]))
    if "action" in frame_table:
        target_points.extend(camera_center_candidates(frame_table["action"][final_idx]))

    for local_idx, frame_idx in enumerate(frame_ids):
        if local_idx == len(frame_ids) - 1:
            continue
        with Image.open(image_list[local_idx]) as image:
            width, height = image.size
        pose = frame_table["observation.camera_extrinsic"][frame_idx]
        intrinsic = frame_table["observation.camera_intrinsic"][frame_idx]
        for point_world in target_points:
            goal_pixel = project_world_point_to_image(point_world, pose, intrinsic, width, height)
            if goal_pixel is not None:
                return goal_pixel, local_idx, "projected_final_pose"

    return [500, 500], len(image_list) - 1, "final_frame_center"


def clamp_range(indexes, num_frames):
    if not indexes or len(indexes) < 2:
        return 0, num_frames - 1
    start = max(0, min(int(indexes[0]), num_frames - 1))
    end = max(0, min(int(indexes[1]), num_frames - 1))
    if end < start:
        start, end = end, start
    return start, end


def sample_frame_ids(start, end, frames_per_sample):
    if end <= start:
        return [start]
    count = min(frames_per_sample, end - start + 1)
    return sorted(set(np.linspace(start, end, count, dtype=np.int32).tolist()))


def path_for_frame(scene_dir, episode_index, frame_index, kind):
    if kind == "rgb":
        base_root = scene_dir / "videos" / "chunk-000" / "observation.images.rgb"
        glob_pattern = "chunk-*/observation.images.rgb"
        suffixes = (".jpg", ".png")
    else:
        base_root = scene_dir / "videos" / "chunk-000" / "observation.images.depth"
        glob_pattern = "chunk-*/observation.images.depth"
        suffixes = (".png", ".jpg")

    roots = [base_root]
    videos_root = scene_dir / "videos"
    if videos_root.exists():
        roots.extend(path for path in sorted(videos_root.glob(glob_pattern)) if path != base_root)

    stems = (
        f"episode_{episode_index:06d}_{frame_index:03d}",
        f"episode_{episode_index:06d}_{frame_index:06d}",
        f"episode_{episode_index:06d}_{frame_index}",
    )
    for root in roots:
        for stem in stems:
            for suffix in suffixes:
                path = root / f"{stem}{suffix}"
                if path.exists():
                    return str(path.resolve())
    raise FileNotFoundError(f"Missing {kind} frame for episode={episode_index}, frame={frame_index}")


def collect_task_records(scene_dir):
    tasks = read_jsonl(scene_dir / "meta" / "tasks.jsonl")
    by_index = {}
    by_episode = {}
    for task in tasks:
        task_id = task.get("task_index", task.get("index", task.get("id")))
        if task_id is not None:
            by_index[task_id] = task
            by_index[str(task_id)] = task
            try:
                by_index[int(task_id)] = task
            except (TypeError, ValueError):
                pass
        episode_index = task.get("episode_index")
        if episode_index is not None:
            by_episode.setdefault(int(episode_index), []).append(task)
    return by_index, by_episode, tasks


def resolve_episode_tasks(episode, tasks_by_index, tasks_by_episode):
    episode_index = int(episode.get("episode_index", -1))
    records = [episode]

    for key in ("tasks", "task_indexes", "task_indices"):
        value = episode.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    records.append(item)
                elif item in tasks_by_index:
                    records.append(tasks_by_index[item])

    for key in ("task_index", "task_id"):
        value = episode.get(key)
        if value in tasks_by_index:
            records.append(tasks_by_index[value])
        elif str(value) in tasks_by_index:
            records.append(tasks_by_index[str(value)])

    records.extend(tasks_by_episode.get(episode_index, []))
    return records


def instruction_candidates_from_records(records, num_frames):
    seen = set()
    candidates = []
    for record in records:
        for key in INSTRUCTION_KEYS:
            text = record.get(key)
            if not text:
                continue
            if key.startswith("sub"):
                range_keys = ("sub_indexes", "indexes", "frame_indexes")
            elif key.startswith("sum"):
                range_keys = ("sum_indexes", "indexes", "frame_indexes")
            else:
                range_keys = ("indexes", "frame_indexes", "sub_indexes", "sum_indexes")

            indexes = None
            for range_key in range_keys:
                if record.get(range_key):
                    indexes = record[range_key]
                    break
            start, end = clamp_range(indexes, num_frames)
            dedup_key = (text, start, end)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            candidates.append((text, start, end, key))
    return candidates


def instruction_candidates(episode, tasks_by_index, tasks_by_episode, all_tasks, num_frames):
    candidates = instruction_candidates_from_records(
        resolve_episode_tasks(episode, tasks_by_index, tasks_by_episode),
        num_frames,
    )
    if candidates:
        return candidates

    episode_index = int(episode.get("episode_index", -1))
    if 0 <= episode_index < len(all_tasks):
        return instruction_candidates_from_records([all_tasks[episode_index]], num_frames)

    return []


def discover_scene_dirs(input_root, requested_scenes):
    if requested_scenes:
        scene_dirs = []
        for scene in requested_scenes:
            direct = input_root / scene
            if (direct / "meta" / "episodes.jsonl").exists():
                scene_dirs.append(direct)
                continue

            matches = sorted(input_root.rglob(f"{scene}/meta/episodes.jsonl"))
            scene_dirs.extend(path.parent.parent for path in matches)
        return sorted(set(scene_dirs))

    return sorted({path.parent.parent for path in input_root.rglob("meta/episodes.jsonl")})


def build_answer(instruction, final_pose, final_action, goal_pixel, goal_pixel_img_idx):
    pose = np.asarray(final_pose, dtype=np.float32).reshape(4, 4)
    action = np.asarray(final_action, dtype=np.float32).reshape(4, 4)
    pose_xyz = pose[:3, 3].round(4).tolist()
    action_xyz = action[:3, 3].round(4).tolist()
    return (
        f"The trajectory follows this instruction: {instruction}\n"
        f"Pixel goal: image_index={goal_pixel_img_idx}, x={goal_pixel[0]}, y={goal_pixel[1]} "
        "(normalized to 0-1000).\n"
        f"Final camera translation: {pose_xyz}.\n"
        f"Final action translation: {action_xyz}."
    )


def convert_scene(scene_dir, frames_per_sample, scene_name, dataset_name):
    episodes = read_jsonl(scene_dir / "meta" / "episodes.jsonl")
    episodes_by_index = {
        int(item["episode_index"]): item
        for item in episodes
        if "episode_index" in item
    }
    tasks_by_index, tasks_by_episode, all_tasks = collect_task_records(scene_dir)

    rows = []
    data_files = sorted((scene_dir / "data").rglob("episode_*.parquet"))
    print(
        f"[convert] scene={scene_dir} episodes={len(episodes)} "
        f"tasks={len(all_tasks)} parquet_files={len(data_files)}"
    )
    for parquet_path in data_files:
        match = re.search(r"episode_(\d+)\.parquet$", parquet_path.name)
        if not match:
            continue
        episode_index = int(match.group(1))
        episode = episodes_by_index.get(episode_index, {"episode_index": episode_index})
        table = pq.read_table(parquet_path)
        frame_table = table.to_pydict()
        num_frames = table.num_rows
        if num_frames == 0:
            continue

        candidates = instruction_candidates(episode, tasks_by_index, tasks_by_episode, all_tasks, num_frames)
        if not candidates and episode_index == 0:
            print(
                f"Warning: no instruction candidates found in {scene_dir}. "
                "Check meta/episodes.jsonl and meta/tasks.jsonl schema."
            )
        for task_idx, (instruction, start, end, source_key) in enumerate(candidates):
            frame_ids = sample_frame_ids(start, end, frames_per_sample)
            try:
                image_list = [path_for_frame(scene_dir, episode_index, idx, "rgb") for idx in frame_ids]
                depth_list = [path_for_frame(scene_dir, episode_index, idx, "depth") for idx in frame_ids]
            except FileNotFoundError as exc:
                print(f"Warning: skip episode {episode_index} task {task_idx}: {exc}")
                continue
            poses = [matrix4(frame_table["observation.camera_extrinsic"][idx]) for idx in frame_ids]
            intrinsic = matrix3_to_4x4(frame_table["observation.camera_intrinsic"][frame_ids[0]])
            final_idx = frame_ids[-1]
            final_pose = frame_table["observation.camera_extrinsic"][final_idx]
            final_action = frame_table["action"][final_idx]
            goal_pixel, goal_pixel_img_idx, goal_pixel_source = infer_goal_pixel(frame_table, frame_ids, image_list)

            question = (
                "Given the RGB-D trajectory, camera poses, and the navigation instruction, "
                "infer the final navigation goal as a pixel coordinate. "
                "Return the target image index and the normalized x,y pixel goal. "
                f"Instruction: {instruction}"
            )
            metadata = {
                "type": "nav",
                "id": f"{scene_dir.name}_episode_{episode_index:06d}_{task_idx:03d}",
                "scene": scene_dir.name,
                "episode_index": episode_index,
                "frame_ids": frame_ids,
                "instruction_source": source_key,
                "goal_pixel": [goal_pixel],
                "goal_pixel_img_idx": [goal_pixel_img_idx],
                "goal_pixel_source": goal_pixel_source,
            }
            rows.append(
                {
                    "question": question,
                    "answer": build_answer(instruction, final_pose, final_action, goal_pixel, goal_pixel_img_idx),
                    "scene_name": scene_name,
                    "dataset_name": dataset_name,
                    "image_list": image_list,
                    "depth_list": depth_list,
                    "poses": poses,
                    "intrinsic": intrinsic,
                    "depth_intrinsic": intrinsic,
                    "metadata": repr(metadata),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        default="/mnt/data/wangqq/G2VLM/data/InternData-N1-extracted/vln_n1/traj_data/replica_d435i",
        help="Extracted replica_d435i root containing scene folders such as apartment_2.",
    )
    parser.add_argument(
        "--output-root",
        default="/mnt/data/wangqq/G2VLM/data/g2vlm_interndata_n1/replica_d435i",
        help="Output root for converted G2VLM parquet files.",
    )
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--frames-per-sample", type=int, default=8)
    parser.add_argument("--row-group-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--scene-name", default="replica")
    parser.add_argument("--dataset-name", default="spar_interndata_n1_replica_d435i")
    parser.add_argument("--output-name", default="interndata_n1_replica_d435i")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    parquet_dir = output_root / "parquets"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    scene_dirs = discover_scene_dirs(input_root, args.scenes)
    print(f"[convert] discovered scene dirs: {len(scene_dirs)}")
    for scene_dir in scene_dirs:
        print(f"[convert] will process: {scene_dir}")

    rows = []
    for scene_dir in scene_dirs:
        if not (scene_dir / "meta" / "episodes.jsonl").exists():
            continue
        before = len(rows)
        rows.extend(convert_scene(scene_dir, args.frames_per_sample, args.scene_name, args.dataset_name))
        print(f"[convert] rows from {scene_dir.name}: {len(rows) - before}")
        if args.max_samples and len(rows) >= args.max_samples:
            rows = rows[: args.max_samples]
            break

    if not rows:
        raise RuntimeError(f"No rows converted from {input_root}")

    parquet_path = (parquet_dir / f"{args.output_name}.parquet").resolve()
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, parquet_path, row_group_size=args.row_group_size)

    num_row_groups = pq.ParquetFile(parquet_path).num_row_groups
    parquet_info = {str(parquet_path): {"num_row_groups": num_row_groups}}
    parquet_info_path = output_root / "parquet_info.json"
    with parquet_info_path.open("w", encoding="utf-8") as f:
        json.dump(parquet_info, f, indent=2)

    dataset_info = {
        "data_dir": str(parquet_dir.resolve()),
        "num_files": 1,
        "num_total_samples": len(rows),
        "parquet_info_path": str(parquet_info_path.resolve()),
    }
    with (output_root / "dataset_info_snippet.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"Converted rows: {len(rows)}")
    print(f"Parquet: {parquet_path}")
    print(f"Parquet info: {parquet_info_path.resolve()}")
    print("DATASET_INFO snippet:")
    print(json.dumps(dataset_info, indent=2))


if __name__ == "__main__":
    main()
