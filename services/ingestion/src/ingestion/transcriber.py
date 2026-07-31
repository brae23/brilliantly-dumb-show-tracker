import json
import os
from pathlib import Path

import whisperx
from whisperx.diarize import DiarizationPipeline

def transcribe_video(video_file: Path) -> Path:
    """Transcribe the given video file using WhisperX with Pyannote speaker diarization.
    Outputs in a JSON file with the same name as the video file, but with a `.json` extension.

    Args:
        video_file (Path): Path to the video file to transcribe.
        None: HuggingFace user token for Pyannote model is read from the HF_TOKEN environment variable.

    Returns:
        Path: Path to the transcription json file.
    """
    device = "cpu"
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError(
            "Hugging Face token is required for Pyannote speaker diarization. "
            "Pass it as an arg or set HF_TOKEN environment variable."
        )

    audio_path = str(video_file)
    output_json_path = video_file.with_suffix(".json")

    print(f"Transcribing video file: {audio_path}")
    audio = whisperx.load_audio(audio_path)
    model = whisperx.load_model("small", device, compute_type="int8")
    result = model.transcribe(audio, batch_size=16)
    print(f"Transcription completed for video file: {audio_path}")

    print(f"Aligning transcription for video file: {audio_path}")
    language_code = result.get("language", "en")
    align_model, metadata = whisperx.load_align_model(
        language_code=language_code, device=device
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, device
    )
    print(f"Alignment completed for video file: {audio_path}")

    print(f"Running speaker diarization for video file: {audio_path}")
    diarize_model = DiarizationPipeline(device=device)
    diarize_segments = diarize_model(audio, min_speakers=2)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    print(f"Speaker diarization completed for video file: {audio_path}")

    segment_json_payload = {
        "language": result.get("language", "en"),
        "segments": [
            {
                "speaker": segment.get("speaker", "UNKNOWN"),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": segment.get("text", "").strip(),
            }
            for segment in result.get("segments", [])
        ],
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(segment_json_payload, f, ensure_ascii=False, indent=2)

    return output_json_path