import cv2
import numpy as np
import supervision as sv
import matplotlib.pyplot as plt

# Constants and paths
SOURCE_VIDEO_PATH = "/content/drive/MyDrive/Real-Time Vehicle Speed Estimation System/Final1.mp4"  # change according to your source video path

# Source polygon definition
SOURCE = np.array([
    [550, 135],
    [1100, 135],
    [2100, 1200],
    [0, 1200]
])

TARGET_WIDTH = 15
TARGET_HEIGHT = 60

TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1],
])

# Function to get frames from video
frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)
frame_iterator = iter(frame_generator)
frame = next(frame_iterator)

# Annotate the frame
annotated_frame = frame.copy()
annotated_frame = sv.draw_polygon(scene=annotated_frame, polygon=SOURCE, color=sv.Color.RED, thickness=4)

# Convert BGR to RGB for matplotlib
annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

# Display the annotated frame using matplotlib in Colab
plt.figure(figsize=(10, 8))  # Adjust figure size as needed
plt.imshow(annotated_frame_rgb)
plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')
plt.title('Annotated Frame')
plt.axis('on')  # Turn off axis if not needed
plt.show()
