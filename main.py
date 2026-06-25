# main.py
# entry point — run this file to start the app

from pathlib import Path
from scanner import get_video_files
from extractor import extract_frames
import searcher


def ask_folder():
    print("\nEnter the folder path that contains your videos.")
    print("Example: D:/videos/zoo\n")

    while True:
        folder = input("Folder path: ").strip().strip('"').strip("'")

        if not folder:
            print("  Please enter a path.\n")
            continue
        if not Path(folder).exists():
            print(f"  Folder not found: {folder}\n")
            continue
        if not Path(folder).is_dir():
            print(f"  That is not a folder: {folder}\n")
            continue

        return folder


def ask_mode():
    print("\nChoose search mode:")
    print("  1 — Fast   (fewer frames, quicker)")
    print("  2 — Better (more frames, accurate)\n")

    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            return "fast"
        elif choice == "2":
            return "better"
        else:
            print("  Please enter 1 or 2.\n")


def ask_query():
    while True:
        query = input("\nSearch keyword (or 'quit' to exit): ").strip()
        if query.lower() == "quit":
            return None
        if query:
            return query
        print("  Please enter a keyword.\n")


def main():
    # load CLIP model once at startup
    searcher.init()

    print("=" * 50)
    print("  SmartSearch — Video Content Search")
    print("=" * 50)

    # step 1 — ask folder
    FOLDER = ask_folder()

    # step 2 — scan folder for videos
    print("\nScanning folder...")
    videos = get_video_files(FOLDER)
    if not videos:
        print("No videos found in that folder.")
        return
    print(f"Found {len(videos)} video(s).\n")

    # step 3 — ask mode
    MODE = ask_mode()

    # step 4 — search loop
    # user can search multiple keywords without re-scanning
    while True:
        query = ask_query()
        if query is None:
            print("\nBye!")
            break

        print(f"\nSearching for '{query}'...\n")
        found_count = 0

        # process one video at a time
        # extract frames and search immediately — show result right away
        for i, video_path in enumerate(videos, 1):
            print(f"  [{i}/{len(videos)}] Checking {video_path.name}...")

            try:
                # extract frames from this video
                frames = extract_frames(video_path, mode=MODE)

                # search this video's frames immediately
                result = searcher.search_single_video(str(video_path), frames, query)

                if result:
                    found_count += 1
                    timestamps = ", ".join(result["timestamps"])
                    print(f"  ✓ FOUND — at {timestamps} (confidence {result['best_score']:.0%})\n")
                else:
                    print(f"  ✗ not found\n")

            except KeyboardInterrupt:
                print("\n\nSearch interrupted by user.")
                break
            except Exception as e:
                print(f"  ✗ ERROR processing {video_path.name}: {e}\n")
                continue

        if found_count == 0:
            print(f"'{query}' was not found in any video.\n")
        else:
            print(f"Done. '{query}' found in {found_count} video(s).\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBye!")