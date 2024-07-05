import argparse
from collections import defaultdict, deque
import cv2
import numpy as np
from ultralytics import YOLO  # Assuming this is for object detection
import supervision as sv  # Custom module for video processing and annotations
import math
import csv
import os
from datetime import datetime

# Define source and target points for perspective transformation
SOURCE = np.array([
    [550, 135],
    [1100, 135],
    [2100, 1200],
    [0, 1200]
])

TARGET_WIDTH = 15
TARGET_HEIGHT = 60

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

# Function to estimate speed between two points
def estimatespeed(Location1, Location2):
    d_pixel = math.sqrt(math.pow(Location2[0] - Location1[0], 2) + math.pow(Location2[1] - Location1[1], 2))
    ppm = 16  # Assuming 16 pixels per meter
    d_meters = d_pixel / ppm
    time_constant = 15 * 3.6  # To convert from meters per second to kilometers per hour
    speed = d_meters * time_constant
    return speed

# Class for perspective transformation of points
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

# Function to parse command-line arguments
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Estimation using Ultralytics and Supervision"
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        required=True,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )
    parser.add_argument(
        "--csv_output_dir",
        required=True,
        help="Directory to store the output CSV file",
        type=str,
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Ensure the CSV output path is a directory
    if not os.path.exists(args.csv_output_dir):
        os.makedirs(args.csv_output_dir)
    
    # Create a new CSV file with a timestamp to ensure uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_output_path = os.path.join(args.csv_output_dir, f"vehicle_speeds_{timestamp}.csv")

    # Initialize video information and YOLO model
    video_info = sv.VideoInfo.from_video_path(video_path=args.source_video_path)
    model = YOLO("/content/drive/MyDrive/yolov8x.pt")

    # Initialize byte tracking, annotators, and transformers
    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps, track_activation_threshold=args.confidence_threshold
    )
    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER,
    )

    # Generate frames from the source video
    frame_generator = sv.get_video_frames_generator(source_path=args.source_video_path)

    # Define polygon zone and view transformer for perspective adjustment
    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    # Initialize containers for tracking coordinates and speeds
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    speeds = defaultdict(list)

    # Open video sink for output and CSV file for writing
    with sv.VideoSink(args.target_video_path, video_info) as sink, open(csv_output_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Vehicle ID", "Average Speed (km/h)"])

        # Iterate through each frame in the video
        for frame in frame_generator:
            # Perform object detection using YOLO model
            result = model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)

            # Filter detections for vehicles only
            vehicle_classes = [2, 3, 5, 7]  # Assuming these are the class IDs for vehicles
            detections = detections[np.isin(detections.class_id, vehicle_classes)]
            detections = detections[detections.confidence > args.confidence_threshold]
            detections = detections[polygon_zone.trigger(detections)]
            detections = detections.with_nms(threshold=args.iou_threshold)
            detections = byte_track.update_with_detections(detections=detections)

            # Transform detection points based on perspective
            points = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )
            points = view_transformer.transform_points(points=points).astype(int)

            # Update tracker coordinates and calculate speeds
            for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)

            labels = []
            for tracker_id in detections.tracker_id:
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    labels.append(f"#{tracker_id}")
                else:
                    coordinate_start = coordinates[tracker_id][-1]
                    coordinate_end = coordinates[tracker_id][0]
                    location1 = [0, coordinate_start]  # Assuming a fixed reference point
                    location2 = [0, coordinate_end]
                    speed = estimatespeed(location1, location2)
                    speeds[tracker_id].append(speed)
                    labels.append(f"#{tracker_id} {int(speed)} km/h")

            # Annotate frame with traces, bounding boxes, and labels
            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = bounding_box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

            # Write annotated frame to output video
            sink.write_frame(annotated_frame)

        # Write average speeds to CSV file
        for tracker_id, speed_list in speeds.items():
            if speed_list:
                average_speed = round(sum(speed_list) / len(speed_list), 2)
                csv_writer.writerow([tracker_id, average_speed])
