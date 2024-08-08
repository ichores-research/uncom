import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np
import numpy.typing as npt
import torch
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from PIL import Image, ImageDraw, ImageFont

from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

from uncom.geometry import points_straight_distance, straight_from_points


def extract_frame(video_path, time):
    """
    Extracts a frame from a video at a given time. Uses ffmpeg.
    """
    dir = Path(video_path).parent
    image_path = dir / f"{time}.png"
    # Extract a frame from a video
    command = [
        "ffmpeg",
        "-ss",
        str(time),
        "-i",
        video_path,
        "-frames:v",
        "1",
        "-q:v",
        "1",
        image_path,
    ]
    subprocess.run(
        command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    return str(image_path)


def show_image(image_path: str):
    """
    Helper function to show an image.
    """
    image = Image.open(image_path)
    image.show()


def load_image(image_str: str) -> Image.Image:
    """
    Load an image from a file.
    """
    image = Image.open(image_str).convert("RGB")

    return image


class PointingDetector:
    """
    A class to detect pointing vectors from an image.
    """

    def __init__(self) -> None:
        # Setup Mediapipe HandLandmarker
        base_options = BaseOptions(model_asset_path="models/hand_landmarker.task")
        options = HandLandmarkerOptions(
            base_options=base_options, num_hands=2, min_hand_detection_confidence=0.3
        )
        self.detector = HandLandmarker.create_from_options(options)

    def detect(self, image_path):
        # Detect hand landmarks from the input image
        # mp_hands = mp.solutions.hands

        # hands = mp_hands.Hands(
        #     static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
        # )
        # hand_img = cv2.imread(image_path)

        # # Resizing the image for faster processing.
        # hand_img = cv2.resize(hand_img, None, fx=0.1, fy=0.1)

        # # Convert the BGR image to RGB before processing.
        # rgb_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

        # # Process.
        # results = hands.process(rgb_img)

        # print("Resuuuuuuuuuuu", results)

        image = mp.Image.create_from_file(image_path)
        landmarks = self.detector.detect(image)
        # Convert landmarks to numpy
        landmarks = hand_landmarks_to_numpy(landmarks.hand_landmarks)

        # Find the hand with the lowest z coordinate, use it as the pointing hand
        print(f"Detected {len(landmarks)} hands")
        print([l[8] for l in landmarks])
        pointing_zs = [l[8][2] for l in landmarks]
        idx = np.argmin(pointing_zs)
        hand = landmarks[idx]

        # Get the pointing vector
        palm_vec = finger_vector(hand)

        # Convert to pixel coordinates
        palm_vec *= np.array([image.width, image.height])

        return palm_vec.astype(int)


def finger_vector(hand_landmarks: npt.NDArray) -> npt.NDArray:
    """
    Get the pointing vector from the hand landmarks.
    """
    # From 0 to 9
    return np.stack([hand_landmarks[5, 0:2], hand_landmarks[8, 0:2]])


def hand_landmarks_to_numpy(
    hand_landmarks: list[list[NormalizedLandmark]],
) -> npt.NDArray:
    """
    Convert a list of hand landmarks to a numpy array.
    """
    hands = []

    for hand in hand_landmarks:
        # 21 landmarks
        landmarks = np.array(
            [[landmark.x, landmark.y, landmark.z] for landmark in hand]
        )
        hands.append(landmarks)

    return hands


@dataclass
class BoundingBox:
    """
    A class to represent a bounding box.
    """

    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    @property
    def center(self) -> Tuple[int, int]:
        return (self.xmin + self.xmax) // 2, (self.ymin + self.ymax) // 2


@dataclass
class DetectionResult:
    """
    A class to represent a detection result. Used for object detection and segmentation.
    """

    score: float
    label: str
    box: BoundingBox
    mask: Optional[npt.NDArray] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> "DetectionResult":
        return cls(
            score=detection_dict["score"],
            label=detection_dict["label"],
            box=BoundingBox(
                xmin=detection_dict["box"]["xmin"],
                ymin=detection_dict["box"]["ymin"],
                xmax=detection_dict["box"]["xmax"],
                ymax=detection_dict["box"]["ymax"],
            ),
        )


class ObjectDetector:
    """
    A class to detect objects in an image based on a description.
    """

    def __init__(
        self, device="cuda", torch_dtype="auto", detection_threshold=0.3
    ) -> None:
        # Grounding DINO model for zero-shot object detection
        model_id = "IDEA-Research/grounding-dino-tiny"
        self.pipeline = pipeline(
            model=model_id,
            task="zero-shot-object-detection",
            device=device,
            torch_dtype=torch_dtype,
        )
        self.detection_threshold = detection_threshold

    def detect(
        self, image: Union[str, Image.Image], description: str
    ) -> List[DetectionResult]:
        # The model works better with a period at the end of the description
        if not description.endswith("."):
            description += "."

        results = self.pipeline(
            image, candidate_labels=[description], threshold=self.detection_threshold
        )

        # Convert results to DetectionResult objects
        results = [DetectionResult.from_dict(result) for result in results]

        return results


def annotate_image(
    image: Union[Image.Image, npt.NDArray],
    detection_results: List[DetectionResult],
    pointing_vec: npt.NDArray,
    emph_idx=None,
) -> npt.NDArray:
    """
    Annotate an image with bounding boxes and masks.

    Args:
    - image (Image.Image or np.ndarray): Input image.
    - detection_results (list): List of DetectionResult objects.
    - pointing_vec (np.ndarray): Pointing vector.
    - emph_idx (int): Index of the detection result to emphasize (i.e. selected object).
    """
    # Convert PIL Image to OpenCV format
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    # Iterate over detections and add bounding boxes and masks
    for i, detection in enumerate(detection_results):
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        # Sample a random color for each detection
        if emph_idx is not None and i != emph_idx:
            # No color
            color = np.array([255, 255, 255])
        else:
            color = np.array([0, 165, 255])

        # Draw bounding box
        cv2.rectangle(
            image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2
        )
        cv2.putText(
            image_cv2,
            f"{label}: {score:.2f}",
            (box.xmin, box.ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color.tolist(),
            4,
        )

        # If mask is available, apply it
        if mask is not None:
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    # Draw pointing vector
    cv2.arrowedLine(
        image_cv2,
        (pointing_vec[0, 0], pointing_vec[0, 1]),
        (pointing_vec[1, 0], pointing_vec[1, 1]),
        (0, 255, 0),
        2,
    )

    image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

    return Image.fromarray(image_rgb)


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon


def polygon_to_mask(
    polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask


def refine_masks(
    masks: torch.BoolTensor, polygon_refinement: bool = False
) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks


class Segmenter:
    """
    A class to segment objects in an image.
    """

    def __init__(self, device="cuda", torch_dtype="auto") -> None:
        # SegmentAnything model for object segmentation
        model_id = "facebook/sam-vit-base"
        self.model = AutoModelForMaskGeneration.from_pretrained(
            model_id, device_map=device, torch_dtype=torch_dtype
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def segment(
        self,
        image: Image.Image,
        detection_results: List[DetectionResult],
        polygon_refinement: bool = True,
    ) -> List[DetectionResult]:
        """
        Segment objects in an image based on detection results.

        Args:
        - image (Image.Image): Input image.
        - detection_results (list): List of DetectionResult objects.
        - polygon_refinement (bool): Whether to refine the segmentation masks using polygons.

        Returns:
        - list: List of DetectionResult objects with segmentation masks.
        """
        boxes = [[r.box.xyxy for r in detection_results]]
        inputs = self.processor(
            images=image, input_boxes=boxes, return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model(**inputs)

        masks = self.processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes,
        )[0]

        masks = refine_masks(masks, polygon_refinement)

        # Assign masks to detection results
        for detection_result, mask in zip(detection_results, masks):
            detection_result.mask = mask

        return detection_results


def pointed_result_index(
    detection_results: List[DetectionResult], pointing_vec: npt.NDArray
) -> int:
    """
    Find the index of the detection result that is pointed at by the pointing vector.
    """

    # Solved by rotating the reference frame so that the pointing vector is along the x-axis
    straight = straight_from_points(pointing_vec[0], pointing_vec[1])
    points = np.array([r.box.center for r in detection_results])
    distances = points_straight_distance(points, straight)
    idx = np.argmin(distances)

    return idx


def annotate_action(
    object_image: Image.Image, target_image: Image.Image, caption: str
) -> Image.Image:
    # Put images side by side and add caption near the top
    image1 = object_image
    image2 = target_image

    # Get dimensions of the images
    width1, height1 = image1.size
    width2, height2 = image2.size

    # Determine the size of the combined image
    total_width = width1 + width2
    max_height = max(height1, height2)

    # Create a new image with a white background
    combined_image = Image.new("RGB", (total_width, max_height), (255, 255, 255))

    # Paste the images into the combined image
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (width1, 0))

    # Create a draw object
    draw = ImageDraw.Draw(combined_image)

    # Choose a font and size
    try:
        font = ImageFont.truetype("arial.ttf", 120)
    except IOError:
        font = ImageFont.load_default(size=120)

    # Calculate the bounding box of the caption text
    text_bbox = draw.textbbox((0, 0), caption, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Calculate X position to center the caption
    caption_x = (total_width - text_width) // 2
    caption_y = 10  # Adjust this value to move the caption down

    # Draw the caption at the top center of the image
    draw.text((caption_x, caption_y), caption, fill="yellow", font=font)

    return combined_image
