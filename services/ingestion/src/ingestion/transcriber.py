from pathlib import Path

import whisperx

def transcribe_video(video_file: Path) -> Path:
    """
    Transcribe the given video file using WhisperX.

    Args:
        video_file (Path): Path to the video file to transcribe.

    Returns:
        Path: Path to the transcription text file.
    """
    # Load WhisperX model (compute_type int8 for quick local testing without GPU)
    device = "cpu"
    model = whisperx.load_model("small", device, compute_type="int8")
    audio_path = video_file
    output_txt_path = video_file.with_suffix(".txt")

    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16)
    
    # Write plain text segments to destination file
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            f.write(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text'].strip()}\n")
    return Path(output_txt_path)
