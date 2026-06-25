# scanner.py
# scans a folder and returns all video file paths

from pathlib import Path

SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']

def get_video_files(folder_path):
    folder = Path(folder_path)

    if not folder.exists():
        print(f"  ERROR: Folder not found -> {folder_path}")
        return []

    if not folder.is_dir():
        print(f"  ERROR: Path is not a folder -> {folder_path}")
        return []

    videos = []
    for ext in SUPPORTED_FORMATS:
        # rglob searches subfolders too
        # use glob if you only want the top level folder
        videos.extend(folder.rglob(f'*{ext}'))

    videos = sorted(videos)

    print(f"  Found {len(videos)} video(s) in {folder_path}")
    return videos
