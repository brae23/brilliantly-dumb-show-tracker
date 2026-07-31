import argparse
from pathlib import Path

from ingestion.downloader import download_video
from ingestion.transcriber import transcribe_video

def run_pipeline():
    print("Starting WhisperX ingestion pipeline...")
    parser = argparse.ArgumentParser(
        description="Process podcast episodes with WhisperX."
    )
    parser.add_argument(
        "--url", type=str, required=True, help="Podcast episode YouTube URL"
    )
    args = parser.parse_args()

    print(f"Starting ingestion pipeline for: {args.url}")
    test_url = args.url
    output_dir = Path("downloads")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading video from URL: {test_url} to directory: {output_dir}")
    downloaded_file = download_video(test_url, output_dir)
    print(f"Downloaded file to: {downloaded_file}")

    print(f"Beginning transcription, alignment, and diarization of video file: {downloaded_file}")
    transcription = transcribe_video(downloaded_file)
    print(f"Transcription processing completed. Output: {transcription}")


if __name__ == "__main__":
    run_pipeline()