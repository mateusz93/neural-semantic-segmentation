"""A simple script to turn an MP4 file into a GIF file."""
import sys
from moviepy.editor import VideoFileClip


clip = VideoFileClip(sys.argv[1]).subclip((0, 50), (1, 10)).set_fps(5).resize(0.5)
clip.write_gif(sys.argv[1].replace('.mp4', '.gif'))
