from pathlib import Path

import yt_dlp

def download_video(url: str, output_dir: Path) -> Path:
    """
    Download the video from the given URL and save it to the specified output directory.

    Args:
        url (str): The URL of the video to download.
        output_dir (Path): The directory where the downloaded video will be saved.

    Returns:
        Path: The path to the downloaded video file.
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(output_dir / '%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
    return output_dir / f"{info_dict['id']}.mp3"