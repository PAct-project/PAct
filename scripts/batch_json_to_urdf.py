import argparse
import glob
import os
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch convert exported articulation objects to URDF files."
    )
    parser.add_argument(
        "--exported_art_objs_dir",
        required=True,
        help="Directory produced by inference containing exported_arti_objects/*/object.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exported_art_objs_dir = args.exported_art_objs_dir

    pattern = os.path.join(exported_art_objs_dir, "*/", "object.json")
    data_meta_list = sorted(glob.glob(pattern))

    if not data_meta_list:
        print(f"No object.json files found under {exported_art_objs_dir}")
        return 1

    exit_code = 0

    # Iterate over the metadata files and call the merge script for each
    for meta_path in data_meta_list:
        print(f"Running merge for: {meta_path}")
        cmd = [
            "python3",
            "scripts/json_to_urdf.py",
            "--input",
            meta_path,
            "--output",
            meta_path.replace(".json", "_fromJson2urdf.urdf"),
            "--glb",
        ]
        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
            print(f"Return code: {proc.returncode}")
            if proc.stdout:
                print("STDOUT:\n", proc.stdout)
            if proc.stderr:
                print("STDERR:\n", proc.stderr)
            if proc.returncode != 0:
                exit_code = proc.returncode
        except Exception as e:
            print(f"Failed to run command for {meta_path}: {e}")
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
