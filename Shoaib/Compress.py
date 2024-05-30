import ffmpeg
import subprocess

def compress_video(input_path, output_path, bitrate="1M"):
    try:
        # Build the FFmpeg command
        (
            ffmpeg
            .input(input_path)
            .output(output_path, video_bitrate=bitrate, vcodec='libx264', acodec='aac')
            .run(overwrite_output=True)
        )
        print(f"Video compressed and saved as {output_path}")
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.output.decode()}")

# Usage example
input_video_path = 'connector.avi'
output_video_path = 'compressed_connector.mp4'
compress_video(input_video_path, output_video_path, bitrate="1M")
