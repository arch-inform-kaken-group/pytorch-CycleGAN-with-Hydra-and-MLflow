import os
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--from_dir", required=True, type=str, help="experiment dir to gather from")
    parser.add_argument("--to_dir", required=True, type=str, help="dir to store gathered dirs")
    parser.add_argument("--result_dir", type=str, default="results", help="name of the result dir")
    args = parser.parse_args()

    dir_entries = os.listdir(args.from_dir)
    dir_entries = [e for e in dir_entries if os.path.isdir(os.path.join(args.from_dir, e))]
    for d in dir_entries:
        r_dir = os.path.join(args.from_dir, d, args.result_dir)
        if os.path.isdir(r_dir):
            shutil.copytree(r_dir, os.path.join(args.to_dir, d))

if __name__ == "__main__":
    main()