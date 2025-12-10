import os
import shutil
import re
import argparse

def get_target_nrrd_files(root_dir):
    """
    Find all .nrrd files in subdirectories matching pattern: 
    starts with a letter, ends with a number, e.g. L27.nrrd
    """
    pattern_dir = re.compile(r'^[A-Za-z]_\d{1,3}-\d{1,3}$')
    pattern_file = re.compile(r'^([A-Za-z])(\d{1,3})\.nrrd$')
    files_to_move = []
    
    for entry in os.listdir(root_dir):
        full_path = os.path.join(root_dir, entry)
        if os.path.isdir(full_path) and pattern_dir.match(entry):
            for fname in os.listdir(full_path):
                m = pattern_file.match(fname)
                if m:
                    prefix, num = m.groups()
                    files_to_move.append({
                        "src": os.path.join(full_path, fname),
                        "prefix": prefix,
                        "num": int(num)
                    })
    return files_to_move

def move_and_rename_nrrd(files_to_move, dest_dir):
    for file_info in files_to_move:
        new_fname = f"{file_info['prefix']}{file_info['num']:03d}.nrrd"
        dest_path = os.path.join(dest_dir, new_fname)
        print(f"Moving {file_info['src']} -> {dest_path}")
        shutil.move(file_info['src'], dest_path)

def main(root_dir):
    in_dir = os.path.join(root_dir, "in")
    os.makedirs(in_dir, exist_ok=True)
    files_to_move = get_target_nrrd_files(root_dir)
    move_and_rename_nrrd(files_to_move, in_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move and rename .nrrd files to 'in' dir.")
    parser.add_argument("--root_dir", type=str, default=".", help="Top directory to search (default: current dir)")
    args = parser.parse_args()
    main(args.root_dir)
