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
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from random import seed
from random import randint
import math
import itertools
from functools import partial
import torch.nn.functional as F
from dinov2.eval.depth.models import build_depther
import urllib
import matplotlib
from torchvision import transforms
import urllib
import mmcv
from mmcv.runner import load_checkpoint
from random import choice


from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

from uncom.geometry import points_straight_distance, straight_from_points


def voronoi_segmenting(x_max, y_max, seed_num, x_min=0, y_min=0): #, img):
    seed(1)
    seeds= []
    for _ in range(seed_num):
        x, y = randint(x_min, x_max), randint(y_min, y_max)
        seeds.append([x, y])
    seeds = np.array(seeds)
    # uncomment if you want the voronoi regions plotted
   # voronoi = Voronoi(seeds)
    # img = plt.imread(img)
#    fig, ax = plt.subplots()
    # ax.imshow(img, extent=[0, 1920, 0, 1080])    
 #   voronoi_plot_2d(voronoi, ax=ax)
    #ax.set_xlim((0, 1920))
    #ax.set_ylim((0, 1080))
    # ax.axis('off')
  #  plt.show()
    # uncomment if you want to save the resulting voronoi regions over an image
#    plt.savefig('output.png', bbox_inches='tight', pad_inches=0, dpi=96, transparent=True)
    return  Voronoi(seeds)


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
        if len(pointing_zs)>0:
            idx = np.argmin(pointing_zs)
            hand = landmarks[idx]

            # Get the pointing vector
            palm_vec = finger_vector(hand)

            # Convert to pixel coordinates
            palm_vec *= np.array([image.width, image.height])

            return palm_vec.astype(int)
        else:
            return []

    def detect3D(self, image_path):
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
        if len(pointing_zs)>0:
            idx = np.argmin(pointing_zs)
            hand = landmarks[idx]

            # Get the pointing vector
            return np.stack([hand[5, 0:3], hand[8, 0:3]]) #  for 3D finger vector
        else:
            return []

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


#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# # Depth Estimation <a target="_blank" href="https://colab.research.google.com/github/facebookresearch/dinov2/blob/main/notebooks/depth_estimation.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>



class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


class DepthEstimator:

    def __init__(self):
        BACKBONE_SIZE = "small" # in ("small", "base", "large" or "giant")
        backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        backbone_arch = backbone_archs[BACKBONE_SIZE]
        backbone_name = f"dinov2_{backbone_arch}"
        backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
        backbone_model.eval()
        backbone_model.cuda()
        HEAD_DATASET = "nyu" # in ("nyu", "kitti")
        HEAD_TYPE = "dpt" # in ("linear", "linear4", "dpt")

        DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
        head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
        head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"

        cfg_str = self.load_config_from_url(head_config_url)
        cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

        self.model = self.create_depther(
            cfg,
            backbone_model=backbone_model,
            backbone_size=BACKBONE_SIZE,
            head_type=HEAD_TYPE,
        )

        load_checkpoint(self.model, head_checkpoint_url, map_location="cpu")
        self.model.eval()
        self.model.cuda()


    def estimate_depth(self, img):
        image = Image.open(img)
        transform = self.make_depth_transform()
        scale_factor = 1
        rescaled_image = image.resize((scale_factor * image.width, scale_factor * image.height))
        transformed_image = transform(rescaled_image)
        batch = transformed_image.unsqueeze(0).cuda() # Make a batch of one image
        result = image
        with torch.inference_mode():
            result = self.model.whole_inference(batch, img_meta=None, rescale=True)
        
        return result.squeeze().cuda()
        
    def make_depth_transform(self, ) -> transforms.Compose:
        return transforms.Compose([
            transforms.ToTensor(),
            lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
            transforms.Normalize(
                mean=(123.675, 116.28, 103.53),
                std=(58.395, 57.12, 57.375),
            ),
        ])

    def render_depth(self, values, colormap_name="magma_r") -> Image:
        values = values.cpu()
        min_value, max_value = values.min(), values.max()
        normalized_values = (values - min_value) / (max_value - min_value)
        print(values)
        colormap = matplotlib.colormaps[colormap_name]
        colors = colormap(normalized_values, bytes=True) # ((1)xhxwx4)
        colors = colors[:, :, :3] # Discard alpha component
        dpth_img = Image.fromarray(colors)
        dpth_img.save("/home/robot/Code/uncom-non-concrete-handling/output_dir/depth.png")
        return dpth_img

    def create_depther(self, cfg, backbone_model, backbone_size, head_type):
        train_cfg = cfg.get("train_cfg")
        test_cfg = cfg.get("test_cfg")
        depther = build_depther(cfg.model, train_cfg=train_cfg, test_cfg=test_cfg)

        depther.backbone.forward = partial(
            backbone_model.get_intermediate_layers,
            n=cfg.model.backbone.out_indices,
            reshape=True,
            return_class_token=cfg.model.backbone.output_cls_token,
            norm=cfg.model.backbone.final_norm,
        )

        if hasattr(backbone_model, "patch_size"):
            depther.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))

        return depther

    def load_config_from_url(self, url: str) -> str:
        with urllib.request.urlopen(url) as f:
            return f.read().decode()


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


def closest_to_fingertip(fingertip_pose,voronoi_centers):
    distances = [np.linalg.norm(np.array(fingertip_pose)-np.array(c)) for c in voronoi_centers]
    return distances.index(min(distances))


def minimum_distante_to_vector_line(fingertip_pose, pointing_vector, voronoi_centers):
    line_from_finger = [np.array(fingertip_pose)+t/100*np.array(pointing_vector)/np.linalg.norm(np.array(pointing_vector)) for t in range(1000)]
    distances = [min([np.linalg.norm(p-np.array(c)) for p in line_from_finger]) for c in voronoi_centers]
    return distances.index(min(distances))

def line_plane_intersection(A, B, grid):
    # Convert points to numpy arrays for easier manipulation
    my_centers,my_regions = zip(*grid)
    plane_points = []
    while len(plane_points)<3:
        point = choice(my_centers)
        if point not in plane_points:
            plane_points.append(point)

    A = np.array([A[1], A[0], A[2]])
    B = np.array([B[1], B[0], B[2]])
    P1, P2, P3 = [np.array(P) for P in plane_points]
    # Direction vector of the line
    AB = B - A
    # Two vectors on the plane
    v1 = P2 - P1
    v2 = P3 - P1
    # Normal vector of the plane (cross product of v1 and v2)
    normal = np.cross(v1, v2)    
    # Calculate dot product of the normal and line direction
    dot_product = np.dot(normal, AB)

    # Check if the line and plane are parallel
    if np.isclose(dot_product, 0):
        raise ValueError("The line is parallel to the plane, no unique intersection exists.")    
    # Calculate parameter t for the intersection point
    t = np.dot(normal, P1 - A) / dot_product
    # Calculate the intersection point
    intersection_point = A + t * AB    
    ip = intersection_point
    print("interection Point: ", ip)
    dists = [np.sqrt((ip[0]-p[0])**2+(ip[1]-p[1])**2+(ip[2]-p[2])**2) for p in my_centers]
    print("Distancias: ", my_centers)
    my_centers = list(zip(dists,my_regions))
    my_centers.sort(key=lambda x:x[0])
    return my_centers[0][1]


# def pointed_area(pointing_vector, line_point, area_centers):
    # # Convert inputs to numpy arrays for vectorized operations
    # pointing_vector = np.array(pointing_vector)
    # line_point = np.array(line_point)
    # points = np.array(area_centers)    
    # # Calculate vector from line_point to each point in area_centers
    # vectors_to_points = points - line_point
    # # Calculate cross product of each vector_to_point with the pointing_vector
    # cross_prod = np.cross(vectors_to_points, pointing_vector)
    # # Compute the norm of each cross product vector
    # cross_prod_norms = np.linalg.norm(cross_prod, axis=1)
    # # Compute the norm of the pointing_vector (direction vector of the line)
    # pointing_vector_norm = np.linalg.norm(pointing_vector)
    # # Compute the distances from each point to the line
    # distances = cross_prod_norms / pointing_vector_norm
    # return distances

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
