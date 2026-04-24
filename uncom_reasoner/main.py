#!/usr/bin/env python3

###############################################################################
#               ██    ██ ███    ██  ██████   ██████     ███    ███            #
#               ██    ██ ████   ██ ██      ██  ██  ██   ████  ████            #
#               ██    ██ ██ ██  ██ ██    ██   ████   ██ ██ ████ ██            #
#               ██    ██ ██  ██ ██ ██      ██  ██  ██   ██  ██  ██            #
#                ██████  ██   ████  ██████   ██████     ██      ██            #
###############################################################################
#                      UNCOM - Understanding Commands                         #
###############################################################################
# Authors: Antonio Galiza Cerdeira Gonzalez, Pawel Gajewski and Bipin         #
# Indurkhya                                                                   #
###############################################################################
# Release version: 0.2v                                                       #
###############################################################################
# For inquiries, please contact: angacego (at) gmail.com                      #
###############################################################################

import copy
import json
import logging
import random
import time as time_module

from pathlib import Path

import cv2
import faiss
import numpy as np
import paho.mqtt.client as mqtt
import torch
from shapely.geometry import Polygon

from uncom_utils.audio import AudioTranscriber
from uncom_utils.image import (
    BoundingBox,
    DepthEstimator,
    DetectionResult,
    ObjectDetector,
    PointingDetector,
    Segmenter,
    SimilarityCalculator,
    annotate_action,
    annotate_image,
    compute_color_histogram_similarity,
    extract_frame,
    get_color_profile,
    load_image,
    minimum_distance_to_vector_line,
    pointed_result_index,
    voronoi_segmenting,
)
from uncom_utils.text import CommandExtractor, check_agreement

logger = logging.getLogger("uncom_reasoner")
logging.basicConfig(level=logging.INFO)

t0 = time_module.time()

# Pre-load ObjectDetector once at startup to avoid slow reload every call
_device = "cuda" if torch.cuda.is_available() else "cpu"
_torch_dtype = torch.float32 if _device == "cpu" else "auto"
_shared_object_detector = ObjectDetector(device=_device, torch_dtype=_torch_dtype, detection_threshold=0.4)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_timestamp(word):
    """Return the end timestamp if available, otherwise the start."""
    return word.timestamp[1] if word.timestamp[1] else word.timestamp[0]


def _detect_pointing_with_retry(hand_detector, frame_path, video_path, timestamp):
    """Detect a pointing gesture, retrying on nearby frames if needed.

    Alternates between frames slightly after and before the given timestamp
    (+0.01s, -0.01s, +0.02s, -0.02s, +0.03s, -0.03s).

    Returns:
        (pointing_vec, frame_path) -- pointing_vec may be an empty list.
    """
    pointing_vec = hand_detector.detect(frame_path)
    if len(pointing_vec) > 0:
        return pointing_vec, frame_path

    frame_count = 1
    while frame_count < 4:
        try:
            retry_path = extract_frame(video_path, timestamp + frame_count * 0.01)
            pointing_vec = hand_detector.detect(retry_path)
            if len(pointing_vec) > 0:
                return pointing_vec, retry_path
        except Exception as e:
            logger.error("Failed to extract frame at offset %.3f: %s",
                         frame_count * 0.01, e)
            break
        frame_count = -frame_count + 1 if frame_count < 0 else -frame_count

    return [], frame_path


def _detect_pickable_objects(object_detector, image):
    """Detect all pickable objects visible in the scene.

    Returns a list of DetectionResult for every pickable object found.
    """
    results = object_detector.detect(image, "pickable objects")
    logger.info("Pickable objects on table: %d found", len(results))
    for r in results:
        logger.info("  - %s (score=%.2f) at %s", r.label, r.score, r.box.xyxy)
    return results


def _find_object_by_relative_position(candidates, reference_result, position):
    """Filter candidates by spatial relation to a reference object.

    Positions from user perspective (user facing camera):
        "left"   → camera right → higher x
        "right"  → camera left  → lower x
        "front"  → further from camera → lower y
        "behind" → closer to camera   → higher y

    Args:
        candidates:       List of (original_index, DetectionResult).
        reference_result: DetectionResult of the anchor object.
        position:         Spatial keyword.

    Returns:
        List of (original_index, DetectionResult) sorted closest-first.
    """
    ref_cx, ref_cy = reference_result.box.center
    ref_box = reference_result.box

    pool = []
    for i, obj in candidates:
        cx, cy = obj.box.center
        if (ref_box.xmin <= cx <= ref_box.xmax
                and ref_box.ymin <= cy <= ref_box.ymax):
            continue
        pool.append((i, obj, cx, cy))

    if not pool:
        logger.warning("No candidates outside reference bounding box")
        return []

    if position in ["left"]:
        filtered = [(i, o, cx, cy) for i, o, cx, cy in pool if cx > ref_cx]
    elif position in ["right"]:
        filtered = [(i, o, cx, cy) for i, o, cx, cy in pool if cx < ref_cx]
    elif position in ["front", "in front", "in front of"]:
        filtered = [(i, o, cx, cy) for i, o, cx, cy in pool if cy < ref_cy]
    elif position in ["behind"]:
        filtered = [(i, o, cx, cy) for i, o, cx, cy in pool if cy > ref_cy]
    else:
        filtered = pool  # next/near/beside/unknown: keep all, sort by distance

    if not filtered:
        logger.warning("No objects found %s the reference", position)
        return []

    filtered.sort(key=lambda t: np.sqrt((t[2]-ref_cx)**2 + (t[3]-ref_cy)**2))
    result = [(i, o) for i, o, _, _ in filtered]
    logger.info("_find_object_by_relative_position '%s': %d candidate(s)", position, len(result))
    return result


def _find_object_by_color(image, candidates, reference_result):
    """Rank candidates by colour histogram similarity to reference_result.

    Args:
        image:            PIL Image of the scene.
        candidates:       List of (original_index, DetectionResult).
        reference_result: DetectionResult of the reference object.

    Returns:
        List of (original_index, DetectionResult) sorted best-first.
    """
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    ref_box = reference_result.box
    ref_crop = image_bgr[ref_box.ymin:ref_box.ymax, ref_box.xmin:ref_box.xmax]
    ref_hist = get_color_profile(ref_crop)

    scored = []
    for i, obj in candidates:
        box = obj.box
        if (box.xmin == ref_box.xmin and box.ymin == ref_box.ymin
                and box.xmax == ref_box.xmax and box.ymax == ref_box.ymax):
            continue
        crop = image_bgr[box.ymin:box.ymax, box.xmin:box.xmax]
        if crop.size == 0:
            continue
        score = compute_color_histogram_similarity(ref_hist, get_color_profile(crop))
        logger.info("  colour similarity -> %s: %.3f", obj.label, score)
        scored.append((score, i, obj))

    scored.sort(reverse=True, key=lambda t: t[0])
    result = [(i, obj) for _, i, obj in scored]
    logger.info("_find_object_by_color: %d candidate(s) ranked", len(result))
    return result


def _find_object_by_shape(image, candidates, reference_result):
    """Rank candidates by DINOv2 embedding similarity to reference_result.

    Args:
        image:            PIL Image of the full scene.
        candidates:       List of (original_index, DetectionResult).
        reference_result: DetectionResult of the reference object.

    Returns:
        List of (original_index, DetectionResult) sorted best-first.
    """
    sim_calc = SimilarityCalculator()
    ref_box = reference_result.box
    ref_crop = image.crop((ref_box.xmin, ref_box.ymin, ref_box.xmax, ref_box.ymax))

    with torch.no_grad():
        ref_inputs = sim_calc.processor(images=ref_crop, return_tensors="pt").to(sim_calc.device)
        ref_vec = sim_calc.model(**ref_inputs).last_hidden_state.mean(dim=1)
        ref_vec = ref_vec.cpu().numpy().astype(np.float32)
        faiss.normalize_L2(ref_vec)

    scored = []
    for i, obj in candidates:
        box = obj.box
        if (box.xmin == ref_box.xmin and box.ymin == ref_box.ymin
                and box.xmax == ref_box.xmax and box.ymax == ref_box.ymax):
            continue
        crop = image.crop((box.xmin, box.ymin, box.xmax, box.ymax))
        if crop.size[0] == 0 or crop.size[1] == 0:
            continue
        with torch.no_grad():
            inputs = sim_calc.processor(images=crop, return_tensors="pt").to(sim_calc.device)
            vec = sim_calc.model(**inputs).last_hidden_state.mean(dim=1)
            vec = vec.cpu().numpy().astype(np.float32)
            faiss.normalize_L2(vec)
        score = float(np.dot(ref_vec, vec.T))
        logger.info("  shape similarity -> %s: %.3f", obj.label, score)
        scored.append((score, i, obj))

    del sim_calc
    torch.cuda.empty_cache()

    scored.sort(reverse=True, key=lambda t: t[0])
    result = [(i, obj) for _, i, obj in scored]
    logger.info("_find_object_by_shape: %d candidate(s) ranked", len(result))
    return result


def _find_object_by_size(candidates, reference_result, prefer=None):
    """Rank candidates by bounding-box area relative to reference_result.

    Args:
        candidates:       List of (original_index, DetectionResult).
        reference_result: DetectionResult of the reference object.
        prefer:           None    – all candidates, sorted by closest area
                          "larger"  / "bigger"  – only candidates with area
                                                   greater than reference,
                                                   sorted largest-first
                          "smaller" / "tinier"  – only candidates with area
                                                   less than reference,
                                                   sorted smallest-first

    Returns:
        List of (original_index, DetectionResult).
    """
    ref_box = reference_result.box
    ref_area = (ref_box.xmax - ref_box.xmin) * (ref_box.ymax - ref_box.ymin)

    scored = []
    for i, obj in candidates:
        box = obj.box
        if (box.xmin == ref_box.xmin and box.ymin == ref_box.ymin
                and box.xmax == ref_box.xmax and box.ymax == ref_box.ymax):
            continue
        area = (box.xmax - box.xmin) * (box.ymax - box.ymin)
        scored.append((area, i, obj))

    if prefer in ("larger", "bigger"):
        scored = [(a, i, o) for a, i, o in scored if a > ref_area]
        scored.sort(reverse=True, key=lambda t: t[0])
    elif prefer in ("smaller", "tinier"):
        scored = [(a, i, o) for a, i, o in scored if a < ref_area]
        scored.sort(key=lambda t: t[0])
    else:
        # Closest area first
        scored.sort(key=lambda t: abs(t[0] - ref_area))

    result = [(i, obj) for _, i, obj in scored]
    logger.info("_find_object_by_size (prefer=%s): %d candidate(s)", prefer, len(result))
    return result


_SIZE_LARGE_WORDS = {"large", "big", "huge", "tall", "wide", "biggest",
                     "largest", "tallest", "widest", "heavy", "heaviest"}
_SIZE_SMALL_WORDS = {"small", "little", "tiny", "short", "narrow", "smallest",
                     "littlest", "tiniest", "shortest", "light", "lightest"}


def _find_object_by_size_absolute(candidates, prefer):
    """Sort candidates by absolute size.

    Args:
        candidates: List of (original_index, DetectionResult).
        prefer:     Size descriptor string e.g. "small", "large", "big".

    Returns:
        List of (original_index, DetectionResult) sorted accordingly.
    """
    def area(obj):
        return (obj.box.xmax - obj.box.xmin) * (obj.box.ymax - obj.box.ymin)

    prefer_lower = prefer.lower().strip()
    reverse = any(w in prefer_lower for w in _SIZE_LARGE_WORDS)
    result = sorted(candidates, key=lambda t: area(t[1]), reverse=reverse)
    logger.info("_find_object_by_size_absolute prefer=%s: %d candidate(s)", prefer, len(result))
    return result


def _find_objects_between(pickable_objects, ref_a, ref_b):
    """Find pickable objects whose centre falls strictly between ref_a and ref_b.

    Objects are sorted by x-centre.  If ref_a and ref_b have nearly the same
    x (i.e. they are vertically aligned), we fall back to sorting by y-centre.

    Args:
        pickable_objects: List of DetectionResult (all candidates).
        ref_a: DetectionResult of the first anchor object.
        ref_b: DetectionResult of the second anchor object.

    Returns:
        List of (original_index, DetectionResult) for every candidate that
        lies between the two anchors.  Empty list if none found.
    """
    ref_a_cx, ref_a_cy = ref_a.box.center
    ref_b_cx, ref_b_cy = ref_b.box.center

    # Choose axis: use x unless the anchors are nearly vertically aligned
    use_x = abs(ref_a_cx - ref_b_cx) >= abs(ref_a_cy - ref_b_cy)

    if use_x:
        lo = min(ref_a_cx, ref_b_cx)
        hi = max(ref_a_cx, ref_b_cx)
        between = [
            (i, obj) for i, obj in enumerate(pickable_objects)
            if lo < obj.box.center[0] < hi
        ]
    else:
        lo = min(ref_a_cy, ref_b_cy)
        hi = max(ref_a_cy, ref_b_cy)
        between = [
            (i, obj) for i, obj in enumerate(pickable_objects)
            if lo < obj.box.center[1] < hi
        ]

    logger.info(
        "_find_objects_between %s and %s: %d candidate(s) found",
        ref_a.label, ref_b.label, len(between),
    )
    return between


def _resolve_reference(detector, image, reference_str, pickable_objects,
                       hand_detector, frame_path, patience=3):
    """Resolve a reference string to a DetectionResult using spatial decomposition.

    Algorithm
    ---------
    1. Try to detect *reference_str* directly with the object detector.
       If found and unambiguous → return it.
    2. If not found (or patience > 0), check whether *reference_str* contains
       a spatial keyword:
         a. "between … and …"  → split into two sub-references, resolve both,
            then find pickable objects between them.
         b. Any other keyword  → split into (sub_obj, relation, sub_ref),
            resolve sub_ref first, then filter pickable objects by relation.
    3. If multiple candidates remain → try gesture disambiguation.
    4. If gesture also fails, or patience is exhausted → return None.

    Args:
        detector:         ObjectDetector instance.
        image:            PIL Image of the scene.
        reference_str:    The noun phrase to resolve, e.g. "cup near banana",
                          "bottle between the apple and the mug".
        pickable_objects: Pre-detected list of all pickable DetectionResults.
        hand_detector:    PointingDetector instance (used as fallback).
        frame_path:       Path to the frame used for gesture detection.
        patience:         Maximum number of recursive decomposition steps.

    Returns:
        DetectionResult if resolved, None otherwise.
    """
    if patience < 0:
        logger.warning("_resolve_reference: patience exhausted for '%s'", reference_str)
        return None

    # --- Step 1: Try direct detection ------------------------------------
    results = detector.detect(image, reference_str)

    if len(results) == 1:
        logger.info("_resolve_reference: direct hit for '%s'", reference_str)
        return results[0]

    if len(results) > 1:
        logger.info(
            "_resolve_reference: %d detections for '%s', trying gesture",
            len(results), reference_str,
        )
        if hand_detector is not None:
            pointing_vec = hand_detector.detect(frame_path)
            if len(pointing_vec) > 0:
                idx = pointed_result_index(results, pointing_vec)
                if idx is not None:
                    return results[idx]
        logger.warning(
            "_resolve_reference: ambiguous detection for '%s', taking first", reference_str,
        )
        return results[0]

    # --- Step 2: No direct detection — try spatial decomposition ---------
    from uncom_utils.text import check_relative_position

    # Special case: "between X and Y"
    if "between" in reference_str.lower():
        # Split on "between" then on " and "
        parts = reference_str.lower().split("between", 1)
        sub_obj_str = parts[0].strip()
        rest = parts[1].strip() if len(parts) > 1 else ""

        if " and " in rest:
            ref_a_str, ref_b_str = rest.split(" and ", 1)
            ref_a_str = ref_a_str.strip()
            ref_b_str = ref_b_str.strip()

            logger.info(
                "_resolve_reference: between decomposition: obj='%s' a='%s' b='%s'",
                sub_obj_str, ref_a_str, ref_b_str,
            )

            ref_a = _resolve_reference(
                detector, image, ref_a_str, pickable_objects,
                hand_detector, frame_path, patience - 1,
            )
            ref_b = _resolve_reference(
                detector, image, ref_b_str, pickable_objects,
                hand_detector, frame_path, patience - 1,
            )

            if ref_a is None or ref_b is None:
                logger.warning(
                    "_resolve_reference: could not resolve both anchors for between"
                )
                return None

            # Find sub_obj candidates between the two anchors
            between_candidates = _find_objects_between(pickable_objects, ref_a, ref_b)

            # Filter to only those matching sub_obj_str label
            sub_obj_results = detector.detect(image, sub_obj_str) if sub_obj_str else []
            sub_obj_boxes = {
                (r.box.xmin, r.box.ymin, r.box.xmax, r.box.ymax)
                for r in sub_obj_results
            }
            if sub_obj_boxes:
                between_candidates = [
                    (i, obj) for i, obj in between_candidates
                    if (obj.box.xmin, obj.box.ymin, obj.box.xmax, obj.box.ymax)
                    in sub_obj_boxes
                ]

            if len(between_candidates) == 1:
                return between_candidates[0][1]

            if len(between_candidates) > 1:
                logger.info(
                    "_resolve_reference: %d between candidates, trying gesture",
                    len(between_candidates),
                )
                if hand_detector is not None:
                    pointing_vec = hand_detector.detect(frame_path)
                    if len(pointing_vec) > 0:
                        candidate_results = [obj for _, obj in between_candidates]
                        idx = pointed_result_index(candidate_results, pointing_vec)
                        if idx is not None:
                            return candidate_results[idx]
                logger.warning("_resolve_reference: between ambiguous, taking closest")
                return between_candidates[0][1]

            logger.warning("_resolve_reference: no objects found between anchors")
            return None

    # General case: "cup near banana", "bottle to the left of the mug", etc.
    kw = check_relative_position(reference_str)
    if not kw:
        # Leaf node — no spatial keyword, nothing left to decompose
        logger.warning(
            "_resolve_reference: leaf node, no detection for '%s'", reference_str,
        )
        return None

    # Split on the keyword: left part = sub_obj, right part = sub_ref
    split_idx = reference_str.lower().find(kw)
    sub_obj_str = reference_str[:split_idx].strip()
    sub_ref_str = reference_str[split_idx + len(kw):].strip()

    # Strip leading articles ("the", "a", "an")
    for article in ("the ", "a ", "an "):
        if sub_ref_str.lower().startswith(article):
            sub_ref_str = sub_ref_str[len(article):]
            break

    logger.info(
        "_resolve_reference: decompose '%s' → obj='%s' rel='%s' ref='%s'",
        reference_str, sub_obj_str, kw, sub_ref_str,
    )

    # Resolve the anchor first
    anchor = _resolve_reference(
        detector, image, sub_ref_str, pickable_objects,
        hand_detector, frame_path, patience - 1,
    )
    if anchor is None:
        return None

    # Filter pickable objects by spatial relation to anchor.
    # _find_object_by_relative_position expects List[(original_idx, DetectionResult)]
    if sub_obj_str:
        sub_obj_results = detector.detect(image, sub_obj_str)
        pool = list(enumerate(sub_obj_results)) if sub_obj_results else list(enumerate(pickable_objects))
    else:
        pool = list(enumerate(pickable_objects))

    spatial_candidates = _find_object_by_relative_position(pool, anchor, kw)
    if not spatial_candidates:
        return None

    if len(spatial_candidates) == 1:
        return spatial_candidates[0][1]

    # If multiple remain after spatial filter, try gesture
    if hand_detector is not None:
        pointing_vec = hand_detector.detect(frame_path)
        if len(pointing_vec) > 0:
            candidate_results = [obj for _, obj in spatial_candidates]
            g_idx = pointed_result_index(candidate_results, pointing_vec)
            if g_idx is not None:
                return candidate_results[g_idx]

    # Return closest (first in sorted list)
    logger.warning(
        "_resolve_reference: ambiguous after spatial filter for '%s', taking closest",
        reference_str,
    )
    return spatial_candidates[0][1]


def _dispatch_property_reference(entity, detector, image, pickable_objects,
                                  hand_detector, frame_path, patience=3):
    """Resolve property_reference into a ranked candidate list.

    Returns:
        List of (original_index, DetectionResult) ranked best-first.
        Empty list if unresolvable.
    """
    prop = entity.property_reference
    tgt  = entity.property_reference_target or ""

    if prop is None:
        return []

    logger.info("_dispatch_property_reference: prop='%s' target='%s'", prop, tgt)

    candidates = list(enumerate(pickable_objects))  # (idx, DetectionResult)

    # ----------------------------------------------------------------
    # BETWEEN
    # ----------------------------------------------------------------
    if prop == "between":
        if " and " not in tgt:
            logger.warning("between: expected 'X and Y', got '%s'", tgt)
            return []
        ref_a_str, ref_b_str = tgt.split(" and ", 1)
        ref_a = _resolve_reference(detector, image, ref_a_str.strip(),
                                   pickable_objects, hand_detector, frame_path, patience)
        ref_b = _resolve_reference(detector, image, ref_b_str.strip(),
                                   pickable_objects, hand_detector, frame_path, patience)
        if ref_a is None or ref_b is None:
            logger.warning("between: could not resolve both anchors")
            return []
        return _find_objects_between(pickable_objects, ref_a, ref_b)

    # ----------------------------------------------------------------
    # SPATIAL
    # ----------------------------------------------------------------
    SPATIAL = {"left","right","front","behind","next","near","close",
               "beside","above","below","under","inside","beneath"}
    if prop in SPATIAL:
        anchor = _resolve_reference(detector, image, tgt,
                                    pickable_objects, hand_detector, frame_path, patience)
        if anchor is None:
            logger.warning("spatial ref: could not resolve anchor '%s'", tgt)
            return []
        return _find_object_by_relative_position(candidates, anchor, prop)

    # ----------------------------------------------------------------
    # VISUAL — color / shape / size
    # ----------------------------------------------------------------
    if prop in ("color", "shape", "size", "larger", "bigger", "smaller", "tinier"):
        if tgt in ("this", "that", "it", "that one", "this one"):
            pointing_vec = hand_detector.detect(frame_path)
            if not len(pointing_vec):
                logger.warning("visual ref: deictic but no gesture")
                return []
            ref_idx = pointed_result_index(pickable_objects, pointing_vec)
            if ref_idx is None:
                return []
            reference = pickable_objects[ref_idx]
        else:
            reference = _resolve_reference(detector, image, tgt,
                                           pickable_objects, hand_detector, frame_path, patience)
            if reference is None:
                logger.warning("visual ref: could not resolve '%s'", tgt)
                return []

        if prop == "color":
            return _find_object_by_color(image, candidates, reference)
        elif prop == "shape":
            return _find_object_by_shape(image, candidates, reference)
        elif prop in ("size", "larger", "bigger", "smaller", "tinier"):
            return _find_object_by_size(candidates, reference, prefer=prop)

    logger.warning("_dispatch_property_reference: unknown prop '%s'", prop)
    return []


def _build_free_cell_grid(object_detector, target_image, object_image, seed_count,
                          int_centers=False):
    """Build a voronoi grid over the detected table and return unoccupied cells.

    Args:
        int_centers: If True, cast cell centres to uint16 (needed for depth indexing).

    Returns:
        List of (0, centre, region) tuples for unoccupied cells.
    """
    table_detections = object_detector.detect(target_image, "table")
    if not table_detections:
        logger.warning("_build_free_cell_grid: no table detected, using full image bounds")
        w, h = target_image.size
        from uncom_utils.image import BoundingBox as _BB
        class _FakeDet:
            box = _BB(xmin=0, ymin=0, xmax=w, ymax=h)
        table_detections = [_FakeDet()]
    table_bb = table_detections[0].box
    table_cells = voronoi_segmenting(
        table_bb.xmax, table_bb.ymax, seed_count, table_bb.xmin, table_bb.ymin,
    )

    table_cells_regions = [
        [table_cells.vertices[p] for p in r] for r in table_cells.regions
    ]
    table_cells_regions = [r for r in table_cells_regions if len(r) > 0]

    if int_centers:
        table_cell_centers = [
            np.array(r).mean(axis=0).astype(np.uint16).tolist()
            for r in table_cells_regions
        ]
    else:
        table_cell_centers = [
            np.array(r).mean(axis=0).tolist()
            for r in table_cells_regions
        ]

    other_objects = object_detector.detect(object_image, "objects")
    other_objects_bb = [
        [[o.box.xmin, o.box.ymin], [o.box.xmax, o.box.ymax],
         [o.box.xmin, o.box.ymax], [o.box.xmax, o.box.ymin]]
        for o in other_objects
    ]

    occupancy = [0] * len(table_cells_regions)
    for i, tc in enumerate(table_cells_regions):
        for bb in other_objects_bb:
            if Polygon(bb).intersects(Polygon(tc)):
                occupancy[i] = 1

    grid = list(zip(occupancy, table_cell_centers, table_cells_regions))
    return [g for g in grid if not g[0]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _understand_single(command, video_path, output_dir, object_detector,
                       device="auto", torch_dtype="auto"):
    """Execute a single resolved Command against a video frame.

    Args:
        command:         Resolved Command (state already applied).
        video_path:      Path to the video file.
        output_dir:      Directory for output files.
        object_detector: Pre-loaded ObjectDetector instance (shared across tasks).
        device:          Torch device string.
        torch_dtype:     Torch dtype string.

    Returns:
        List result tuple: ["OK", obj, action, target, obj_center, tgt_center]
        or ["ambiguous", reason].
    """
    logger.info("_understand_single: obj='%s' action='%s' target='%s'",
                command.object.text, command.action.text, command.target.text)

    # Validate — allow missing target (pick-and-hold) or missing object
    # (place-held), but not both missing
    if not command.object.text and not command.target.text:
        logger.warning("FAILURE 1: both object and target missing")
        return ["ambiguous", "failure 1: both object and target missing"]

    # --- Step 3: Extract frames and load images --------------------------
    obj_ts = _get_timestamp(command.object)
    tgt_ts = _get_timestamp(command.target)

    object_frame_path = extract_frame(video_path, obj_ts)
    target_frame_path = extract_frame(video_path, tgt_ts)

    logger.info("Extracted %.3fs frame from %s", obj_ts, object_frame_path)
    logger.info("Extracted %.3fs frame from %s", tgt_ts, target_frame_path)

    object_image = load_image(object_frame_path)
    target_image = load_image(target_frame_path)
    image_width, image_height = object_image.size

    # Save the raw frame used by Grounded-DINO for comparison with YOLO frame
    import shutil
    shutil.copy(str(object_frame_path), str(output_dir / "dino_img.png"))
    logger.info("Saved dino frame to %s", output_dir / "dino_img.png")

    object_concrete = command.object.concrete
    target_concrete = command.target.concrete

    has_object = bool(command.object.text and command.object.text.strip())
    has_target = bool(command.target.text and command.target.text.strip())

    # --- Step 4: Detect objects ------------------------------------------
    # NOTE: object_detector is shared/passed in from the orchestrator.
    all_pickable = _detect_pickable_objects(object_detector, object_image)

    # If _expand_quantity pre-resolved a specific detection, use it directly
    # to avoid re-detecting and potentially picking an already-moved object.
    if hasattr(command, '_resolved_object_result') and command._resolved_object_result is not None:
        object_results = [command._resolved_object_result]
        logger.info("Using pre-resolved object result from quantity expansion: %s",
                    object_results[0].label)
    else:
        object_results = object_detector.detect(
            object_image, command.object.detection_query() or command.object.text,
        )

    target_results = []
    target_location = None
    relative_position_kw = False

    if has_target:
        # Spatial relation comes directly from the structured schema.
        # target.location.relation = "next to", "to the left of", "between", etc.
        # target.location.reference = the full noun phrase to resolve.
        #
        # We first try to detect the reference string directly.  If the detector
        # fails, _resolve_reference decomposes it spatially (with patience) and
        # uses _find_objects_between / _find_object_by_relative_position + gesture
        # as fallback at each level.
        target_location = command.target.location
        relative_position_kw = (
            check_relative_position(target_location.relation) if target_location else False
        )

        if target_location and relative_position_kw:
            # Try to resolve the reference object — may recurse if complex
            resolved_ref = _resolve_reference(
                object_detector, target_image,
                target_location.reference,
                all_pickable,
                hand_detector=None,   # hand_detector not yet initialised; set below
                frame_path=target_frame_path,
                patience=3,
            )
            target_results = [resolved_ref] if resolved_ref is not None else []
        else:
            target_results = object_detector.detect(
                target_image, command.target.detection_query() or command.target.text,
            )

        if target_results:
            logger.info("Detected %d target instances of '%s'",
                         len(target_results), command.target.text)
        else:
            logger.warning("'%s' could not be detected.", command.target.text)
    else:
        logger.info("No target specified — pick-only mode.")

    logger.info("Detected %d object instances of '%s'",
                len(object_results), command.object.text)

    # --- Step 5: Detect pointing gestures (lazy — only when needed) ------
    # Gesture is needed when object or target is non-concrete (deictic),
    # or when multiple candidates exist. We defer until we know.
    hand_detector = PointingDetector()

    needs_object_gesture = (not object_concrete) or (len(object_results) > 1)
    needs_target_gesture = (has_target
                            and ((not target_concrete) or (len(target_results) > 1)))

    # Skip gesture only when a concrete noun got zero DINO detections —
    # gesture can't disambiguate what DINO couldn't find. For deictic words
    # (concrete=False) gesture is the primary resolution method, so never skip.
    if len(object_results) == 0 and object_concrete:
        needs_object_gesture = False
    if has_target and len(target_results) == 0 and target_concrete:
        needs_target_gesture = False

    if needs_object_gesture:
        object_pointing_vec, object_frame_path = _detect_pointing_with_retry(
            hand_detector, object_frame_path, video_path, obj_ts,
        )
    else:
        object_pointing_vec = []
    object_pointing_detected = len(object_pointing_vec) > 0

    if needs_target_gesture:
        target_pointing_vec, target_frame_path = _detect_pointing_with_retry(
            hand_detector, target_frame_path, video_path, tgt_ts,
        )
    else:
        target_pointing_vec = []
    target_pointing_detected = len(target_pointing_vec) > 0

    # If step 4 deferred reference resolution (hand_detector was None),
    # retry now that we have it — only needed when reference was not found.
    if target_location and relative_position_kw and not target_results:
        logger.info("Retrying reference resolution with gesture support")
        resolved_ref = _resolve_reference(
            object_detector, target_image,
            target_location.reference,
            all_pickable,
            hand_detector=hand_detector,
            frame_path=target_frame_path,
            patience=3,
        )
        target_results = [resolved_ref] if resolved_ref is not None else []

    # --- Step 6: Check task feasibility ----------------------------------
    # A deictic object/target is unambiguous when only one pickable thing
    # exists — "this" / "it" can only refer to that single object, no
    # gesture required.
    single_pickable = len(all_pickable) == 1

    impossible_task = (
        # Object present but unresolvable (deictic, no gesture, no other hint)
        (has_object and not object_concrete and not object_pointing_detected
         and command.object.property_reference is None
         and not single_pickable)
        # Object present, concrete, but nothing detected
        or (has_object and object_concrete and len(object_results) == 0
            and command.object.property_reference is None)
        # Target present but unresolvable
        or (has_target and not target_concrete and not target_pointing_detected
            and command.target.property_reference is None
            and not single_pickable)
        # Target present, concrete, but nothing detected
        or (has_target and target_concrete and len(target_results) == 0)
    )

    if impossible_task:
        logger.warning("FAILURE 2: unclear object/target and no pointing detected")
        return ["ambiguous", "failure 2: unclear object/target and no pointing detected"]

    # --- Step 7: Resolve object -----------------------------------------
    #
    # Cascade pipeline:
    #   1. property_reference → spatial/visual filter → candidate list
    #   2. descriptors        → further filter within candidates
    #      a. size_absolute   → "small"/"large" without a reference object
    #      b. color/shape/size → with a reference object (already in prop_ref)
    #   3. If one candidate   → done
    #   4. If multiple        → gesture
    #   5. If gesture fails   → ambiguous
    #
    # NOTE: object_pointing_vec is already set by step 5 — do NOT reset it here.

    pointed_object_idx = None

    # Build initial candidate list
    if command.object.property_reference is not None:
        # Explicit instruction on HOW to find — dispatch first
        logger.info("Object has property_reference='%s'", command.object.property_reference)
        obj_candidates = _dispatch_property_reference(
            command.object, object_detector, object_image, all_pickable,
            hand_detector, object_frame_path, patience=3,
        )
        if not obj_candidates:
            logger.warning("FAILURE 3d: property_reference unresolvable")
            return ["ambiguous", "failure 3d: property_reference resolution failed"]

    elif object_concrete:
        obj_candidates = list(enumerate(object_results))
    else:
        # Non-concrete: start with all pickable objects
        obj_candidates = list(enumerate(all_pickable))

    # Cascade through descriptors to further narrow candidates
    size_words = command.object.size_descriptors()
    if size_words and command.object.property_reference not in ("size",):
        # Absolute size filter — no reference object needed
        obj_candidates = _find_object_by_size_absolute(obj_candidates, size_words[0])

    # If still multiple, try gesture disambiguation
    if len(obj_candidates) > 1:
        logger.info("Object disambiguation: %d candidates remain, trying gesture",
                    len(obj_candidates))
        object_pointing_vec = hand_detector.detect(object_frame_path)
        if len(object_pointing_vec) > 0:
            candidate_results = [obj for _, obj in obj_candidates]
            g_idx = pointed_result_index(candidate_results, object_pointing_vec)
            if g_idx is not None:
                obj_candidates = [obj_candidates[g_idx]]

    if not obj_candidates:
        logger.warning("FAILURE 3: no object candidates after filtering")
        return ["ambiguous", "failure 3: could not resolve object"]

    if len(obj_candidates) > 1:
        logger.warning("FAILURE 3e: ambiguous after gesture, %d candidates remain",
                       len(obj_candidates))
        return ["ambiguous", "failure 3e: object still ambiguous after gesture"]

    pointed_object_idx = obj_candidates[0][0]
    resolved_object_result = obj_candidates[0][1]
    logger.info("Resolved object: %s", resolved_object_result.label)


    # --- Step 8: Resolve target ------------------------------------------
    # Six cases:
    #   1) target is a concrete object
    #   2) target is described relative to another object
    #   3) target is a deictic reference ("this", "here", "there")
    #   4) target is an empty area on the table
    #   5) no target at all — pick-only
    #   6) target has property_reference (similar in color/shape/size, or spatial)

    area_target = has_target and ("here" in command.target.text or "there" in command.target.text)
    chosen_area = []
    pointed_target_idx = None

    # Pre-resolve target property_reference (e.g. "object similar to THIS one",
    # "THIS small object") before concrete/non-concrete branching.
    if has_target and command.target.property_reference is not None:
        logger.info("Target has property_reference='%s'",
                    command.target.property_reference)
        tgt_candidates = _dispatch_property_reference(
            command.target, object_detector, target_image, all_pickable,
            hand_detector, target_frame_path, patience=3,
        )
        if tgt_candidates:
            # Apply size descriptor filtering if applicable
            size_words = command.target.size_descriptors()
            if size_words and command.target.property_reference not in ("size",):
                tgt_candidates = _find_object_by_size_absolute(
                    tgt_candidates, size_words[0])

            # Narrow target_results to the resolved candidate(s)
            target_results = [obj for _, obj in tgt_candidates]
            target_concrete = True  # treat as concrete from here on
            logger.info("Target property_reference resolved to %d candidate(s)",
                        len(target_results))

    # Fallback: target has size descriptors but no property_reference — apply
    # absolute size filtering to narrow multiple candidates.
    elif (has_target and not target_concrete
          and command.target.size_descriptors()
          and len(target_results) > 1):
        size_words = command.target.size_descriptors()
        tgt_indexed = list(enumerate(target_results))
        tgt_indexed = _find_object_by_size_absolute(tgt_indexed, size_words[0])
        target_results = [obj for _, obj in tgt_indexed]
        logger.info("Target size-filtered to %d candidate(s)", len(target_results))

    if not has_target:
        logger.info("No target — skipping target resolution (pick-only).")

    elif target_concrete:
        pointed_target_idx = 0

        if len(target_results) > 1 and len(target_pointing_vec) > 0:
            logger.info("Detected target pointing %s", target_pointing_vec)
            pointed_target_idx = pointed_result_index(
                target_results, target_pointing_vec,
            )
        elif len(target_results) == 1:
            pointed_target_idx = 0
        else:
            logger.warning("FAILURE 4: multiple concrete targets, no pointing")
            return ["ambiguous",
                    "failure 4: multiple concrete targets, no pointing detected"]

        if relative_position_kw:
            # --- Case 2: target relative to a reference object --------
            free_cells = _build_free_cell_grid(
                object_detector, target_image, object_image, 300,
            )

            tgt_box = target_results[pointed_target_idx].box
            reference_center = [
                (tgt_box.xmax + tgt_box.xmin) / 2,
                (tgt_box.ymax + tgt_box.ymin) / 2,
            ]
            obj_height = (tgt_box.ymax - tgt_box.ymin) / 2
            obj_width = (tgt_box.xmax - tgt_box.xmin) / 2

            # Directions from user's perspective (inverted from camera):
            #   user "left"  → higher x,  user "right"  → lower x
            #   user "front" → lower  y,  user "behind" → higher y
            if relative_position_kw in ["left"]:
                grid = [g for g in free_cells
                        if g[1][0] > reference_center[0] + obj_width]

            elif relative_position_kw in ["right"]:
                grid = [g for g in free_cells
                        if g[1][0] < reference_center[0] - obj_width]

            elif relative_position_kw in ["front", "in front", "in front of",
                                           "up", "above", "over", "higher"]:
                grid = [g for g in free_cells
                        if g[1][1] < reference_center[1] - obj_height]

            elif relative_position_kw in ["behind", "down", "under", "lower"]:
                grid = [g for g in free_cells
                        if g[1][1] > reference_center[1] + obj_height]

            else:
                grid = ([g for g in free_cells
                         if g[1][0] > reference_center[0] + obj_width]
                        + [g for g in free_cells
                           if g[1][0] < reference_center[0] - obj_width])

            if len(grid) > 0:
                area_target = True
                _, centers, regions = zip(*grid)
                try:
                    p1, p2 = target_pointing_vec
                except ValueError:
                    logger.warning("FAILURE 5: area target, no pointing")
                    return ["ambiguous",
                            "failure 5: area target, but no pointing detected."]

                distances = [
                    np.sqrt((c[0] - p2[0]) ** 2 + (c[1] - p2[1]) ** 2)
                    for c in centers
                ]
                decision = sorted(zip(distances, regions), key=lambda x: x[0])
                chosen_area = decision[0][1]
            else:
                logger.warning("FAILURE 6: no valid cells for relative position")
                return ["ambiguous"]

    else:
        # Non-concrete target
        if not area_target:
            target_results = object_detector.detect(object_image, "container")
            logger.info("Container objects: %s", target_results)

            if len(target_results) >= 1:
                _pt_idx = pointed_result_index(
                    target_results, target_pointing_vec,
                )
                pointed_target_idx = _pt_idx if _pt_idx is not None else 0
                logger.info("Inferred target object: %s",
                            target_results[pointed_target_idx])
            else:
                area_target = True

        if area_target:
            # --- Case 4: target is an empty area on the table ---------
            if not target_pointing_detected:
                logger.warning("FAILURE 5b: area target but no pointing vector")
                return ["ambiguous", "failure 5b: area target but no pointing detected"]

            depth_estimator = DepthEstimator()
            depths = depth_estimator.estimate_depth(target_frame_path).cpu()
            depth_estimator.render_depth(depths, str(video_path.parent))
            del depth_estimator

            p1, p2 = target_pointing_vec
            p1_depth = depths[p1[1]][p1[0]]
            p2_depth = depths[p2[1]][p2[0]]
            p1_3d = p1.tolist() + [p1_depth]
            p2_3d = p2.tolist() + [p2_depth]

            logger.info("Fingers: %s %s", p1_3d, p2_3d)
            target_pointing_vec_3D = np.array(p2_3d) - np.array(p1_3d)

            free_cells = _build_free_cell_grid(
                object_detector, target_image, object_image, 400,
                int_centers=True,
            )

            # Discard outlier cells and enrich centres with depth
            valid_cells = []
            for _, center, region in free_cells:
                if center[0] <= 1079 and center[1] <= 1919:
                    x = int(center[1])
                    y = int(center[0])
                    if 0 <= x < image_width and 0 <= y < image_height:
                        depth = depths[y, x].numpy().tolist()
                    else:
                        depth = float("inf")
                    valid_cells.append((center + [depth], region))

            if not valid_cells:
                logger.warning("FAILURE: no valid voronoi cells after filtering")
                return ["ambiguous", "failure: no valid voronoi cells"]

            cell_centers_3d, cell_regions = zip(*valid_cells)
            cell_centers_3d = list(cell_centers_3d)
            cell_regions = list(cell_regions)

            chosen_area = cell_regions[
                minimum_distance_to_vector_line(
                    p2_3d, target_pointing_vec_3D, cell_centers_3d,
                )
            ]
            logger.info("Chosen area: %s", chosen_area)

    # object_detector is owned by the orchestrator — do not delete here
    del hand_detector
    torch.cuda.empty_cache()

    # --- Step 9: Final validation ----------------------------------------
    if pointed_object_idx is None:
        logger.warning("FAILURE 7: failed to identify pointed object")
        return ["ambiguous", "failure 7: failed to identify pointed object"]

    if has_target and pointed_target_idx is None and not chosen_area:
        logger.warning("FAILURE 7: failed to identify pointed target/area")
        return ["ambiguous", "failure 7: failed to identify pointed target/area"]

    # --- Step 10: Segment objects ----------------------------------------
    segmenter = Segmenter(device=device, torch_dtype=torch_dtype)

    # Use the already-resolved DetectionResult directly to avoid index mismatch
    # between object_results and all_pickable.
    [resolved_object_result] = segmenter.segment(
        object_image, [resolved_object_result]
    )

    if has_target:
        if area_target:
            x, y = zip(*chosen_area)
            target_results = [DetectionResult(
                score=1.0,
                label="target.",
                box=BoundingBox(
                    xmin=int(min(x)), ymin=int(min(y)),
                    xmax=int(max(x)), ymax=int(max(y)),
                ),
                mask=np.array(chosen_area).astype(np.uint8),
            )]
            pointed_target_idx = 0
        else:
            [target_results[pointed_target_idx]] = segmenter.segment(
                target_image, [target_results[pointed_target_idx]]
            )
        logger.info("Segmented target '%s'", command.target.text)

    logger.info("Segmented object '%s'", command.object.text)

    del segmenter
    torch.cuda.empty_cache()

    # --- Step 11: Annotate and save --------------------------------------
    annotated_object_image = annotate_image(
        object_image, [resolved_object_result], object_pointing_vec,
        emph_idx=0,
    )
    annotated_object_image_path = output_dir / "annotated_object.png"
    annotated_object_image.save(annotated_object_image_path)
    logger.info("Saved annotated object image to %s", annotated_object_image_path)

    if has_target:
        annotated_target_image = annotate_image(
            target_image, target_results, target_pointing_vec,
            emph_idx=pointed_target_idx,
        )
        annotated_target_image_path = output_dir / "annotated_target.png"
        annotated_target_image.save(annotated_target_image_path)
        logger.info("Saved annotated target image to %s", annotated_target_image_path)

        caption = (f"{command.object.text} - {command.action.text}"
                   f" - {command.target.text}")
        annotated_action = annotate_action(
            annotated_object_image, annotated_target_image, caption,
        )
        name = video_path.stem
        annotated_action_path = output_dir / f"{name}_annotated_action.png"
        annotated_action.save(annotated_action_path)
        logger.info("Saved annotated action image to %s", annotated_action_path)

    # --- Step 12: Return result ------------------------------------------
    try:
        tgt_center = (list(target_results[pointed_target_idx or 0].box.center)
                      if has_target else None)
        return [
            "OK",
            command.object.text,
            command.action.text,
            command.target.text,
            list(resolved_object_result.box.center),
            tgt_center,
        ]
    except Exception as e:
        logger.error("Failed to extract bounding box: %s", e)
        return ["ambiguous", "failure 8: failed to extract bounding box"]



# ---------------------------------------------------------------------------
# Task state — threads context between sequential commands
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass

@_dataclass
class TaskState:
    """Carries forward context between sequential commands.

    Allows resolving pronouns and deictic references across tasks:
      "grab the apple" → "place it here"
      "put it next to the mug" → "do the same with the banana"
    """
    held_object: object = None    # EntityWord currently held by robot
    last_object: object = None    # last object mentioned
    last_target: object = None    # last destination mentioned
    last_action: object = None    # last action performed


_DEICTIC = {"this","that","it","here","there","over there",
            "that one","this one","that thing","this thing","them","those"}


def _resolve_from_state(command, state: TaskState):
    """Fill missing or deictic fields in command from TaskState.

    Patching priority:
      object  → held_object > last_object
      target  → last_target
      action  → last_action

    Timestamps are always preserved from the current command so that frame
    extraction targets the correct moment in the new video, not the old one.
    """
    obj  = command.object
    tgt  = command.target
    act  = command.action

    # Object missing or deictic pronoun
    if not obj.text or obj.text.lower().strip() in _DEICTIC:
        fill = state.held_object or state.last_object
        if fill is not None:
            logger.info("State: filling object '%s' → '%s'", obj.text, fill.text)
            filled = copy.copy(fill)
            filled.timestamp = obj.timestamp   # keep current frame timestamp
            command.object = filled

    # Target missing or deictic
    if not tgt.text or tgt.text.lower().strip() in _DEICTIC:
        if state.last_target is not None:
            logger.info("State: filling target '%s' → '%s'", tgt.text, state.last_target.text)
            filled = copy.copy(state.last_target)
            filled.timestamp = tgt.timestamp   # keep current frame timestamp
            command.target = filled

    # Action missing
    if not act.text and state.last_action is not None:
        logger.info("State: filling action '' → '%s'", state.last_action.text)
        filled = copy.copy(state.last_action)
        filled.timestamp = act.timestamp
        command.action = filled

    return command


def _expand_group(command, object_detector, image, video_path=None):
    """Resolve a group command (quantity="these") via DBSCAN clustering.

    1. Detect all instances of the descriptor (or ask detector for anything
       if descriptor is empty).
    2. Cluster bounding box centers with DBSCAN.
    3. If only one cluster — use all detections in it.
    4. If multiple clusters — use the pointing vector at the object timestamp
       to pick the cluster whose centroid is closest to the ray.
    5. Return one Command per detection in the chosen cluster.
    """
    try:
        from sklearn.cluster import DBSCAN
        import numpy as np
    except ImportError:
        logger.warning("_expand_group: sklearn not available, falling back to all detections")
        DBSCAN = None

    query = command.object.text or "object"
    results = object_detector.detect(image, query)

    if not results:
        logger.warning("_expand_group: no detections for '%s'", query)
        return [command]

    if len(results) == 1:
        c = copy.deepcopy(command)
        c._resolved_object_result = results[0]
        return [c]

    # --- DBSCAN on bounding box centers ----------------------------------
    centers = np.array([
        [(r.bbox.xmin + r.bbox.xmax) / 2, (r.bbox.ymin + r.bbox.ymax) / 2]
        for r in results
    ], dtype=float)

    chosen_indices = list(range(len(results)))  # default: all

    if DBSCAN is not None:
        # eps in pixels — tune if needed
        labels = DBSCAN(eps=45, min_samples=1).fit_predict(centers)
        unique_labels = [l for l in set(labels) if l >= 0]

        if len(unique_labels) == 1:
            # One cluster — take everything
            chosen_indices = [i for i, l in enumerate(labels) if l == unique_labels[0]]
        else:
            # Multiple clusters — use pointing vector to pick closest centroid
            cluster_centroids = {
                l: centers[[i for i, lb in enumerate(labels) if lb == l]].mean(axis=0)
                for l in unique_labels
            }
            pointing_vec = []
            try:
                obj_ts = _get_timestamp(command.object)
                frame_path = extract_frame(video_path, obj_ts) if video_path else None
                from uncom_reasoner.hand_detection import HandDetector
                hd = HandDetector()
                pointing_vec = hd.detect(frame_path) if frame_path else []
            except Exception as e:
                logger.warning("_expand_group: gesture detection failed: %s", e)

            if pointing_vec and len(pointing_vec) > 0:
                # Find cluster centroid closest to pointing ray origin
                pv = np.array(pointing_vec[0][:2])
                best_label = min(
                    unique_labels,
                    key=lambda l: np.linalg.norm(cluster_centroids[l] - pv)
                )
            else:
                # No gesture — pick largest cluster
                best_label = max(unique_labels,
                                 key=lambda l: sum(1 for lb in labels if lb == l))

            chosen_indices = [i for i, l in enumerate(labels) if l == best_label]

    logger.info("_expand_group: '%s' → %d detections in chosen cluster",
                query, len(chosen_indices))

    expanded = []
    for i in chosen_indices:
        c = copy.deepcopy(command)
        c._resolved_object_result = results[i]
        expanded.append(c)
    return expanded


def _expand_quantity(command, object_detector, image, video_path=None):
    """Expand a command with quantity != 1 into a list of atomic commands.

    - quantity=None  → detect all instances, one command per detection
    - quantity=N     → detect all, shuffle, take N
    - quantity="these" → handled downstream via property_reference clustering
    - quantity=1     → return [command] unchanged

    Args:
        command:         Command with quantity set on object.
        object_detector: ObjectDetector instance.
        image:           PIL Image of the object frame.

    Returns:
        List[Command] — one per pick, all sharing the same target.
    """

    qty = command.object.quantity

    if qty == 1 or qty is None and command.object.text == "":
        return [command]

    if qty == "these":
        return _expand_group(command, object_detector, image, video_path=video_path)

    # Detect all instances
    query = command.object.detection_query() or command.object.text
    results = object_detector.detect(image, query)

    if not results:
        logger.warning("_expand_quantity: no detections for '%s'", query)
        return [command]

    # quantity=None means all; quantity=N means take N (random)
    if qty is not None and isinstance(qty, int) and qty < len(results):
        random.shuffle(results)
        results = results[:qty]

    logger.info("_expand_quantity: expanding '%s' x%s → %d tasks",
                command.object.text, qty, len(results))

    # Build one command per detection, reusing the same target
    expanded = []
    for r in results:
        c = copy.deepcopy(command)
        # Stamp the specific detection center as a hint for _understand_single
        c._resolved_object_result = r
        expanded.append(c)

    return expanded


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_agree(audio_path, device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float32 if device == "cpu" else "auto"
    transcriber = AudioTranscriber(device=device, torch_dtype=torch_dtype)
    transcription = transcriber.transcribe(str(audio_path))
    return check_agreement(transcription["text"])


def understand(audio_path, video_path, device="auto", model_id=None, load_in_4bit=False,
               torch_dtype=None):
    """Orchestrate transcription, command extraction, state resolution,
    quantity expansion, and per-task understanding.

    Returns:
        List of result tuples — one per atomic task.
        Each is ["OK", obj, action, target, obj_center, tgt_center]
        or ["ambiguous", reason].
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch_dtype is None:
        torch_dtype = torch.float32 if device == "cpu" else "auto"
    torch.cuda.empty_cache()
    logger.info("Device: %s", device)

    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_dir = video_path.parent

    # --- Step 1: Transcribe -----------------------------------------------
    transcriber = AudioTranscriber(device=device, torch_dtype=torch_dtype)
    transcription = transcriber.transcribe(str(audio_path))
    del transcriber
    torch.cuda.empty_cache()
    logger.info("HEARD: %s", transcription["text"])

    # --- Step 2: Extract all commands ------------------------------------
    extractor_kwargs = dict(device=device, torch_dtype=torch_dtype, load_in_4bit=load_in_4bit)
    if model_id is not None:
        extractor_kwargs["model_id"] = model_id
    command_extractor = CommandExtractor(**extractor_kwargs)
    commands = command_extractor.extract_all(transcription)
    del command_extractor
    torch.cuda.empty_cache()

    logger.info("Extracted %d command(s)", len(commands))
    for idx, cmd in enumerate(commands):
        cmd_path = output_dir / f"command_{idx}.json"
        cmd.save(cmd_path)
        logger.info("  [%d] obj='%s' action='%s' target='%s'",
                    idx, cmd.object.text, cmd.action.text, cmd.target.text)

    # --- Step 3: Reuse pre-loaded detector --------------------------------
    object_detector = _shared_object_detector

    # --- Step 4: Resolve state and expand quantity -----------------------
    state = TaskState()
    all_results = []

    for cmd in commands:
        # Apply state to fill pronouns/missing fields
        cmd = _resolve_from_state(cmd, state)

        # Expand quantity into atomic tasks (detect once, split)
        if cmd.object.quantity != 1:
            # Need a frame to detect objects for expansion
            obj_ts = _get_timestamp(cmd.object)
            frame_path = extract_frame(video_path, obj_ts)
            image = load_image(frame_path)
            atomic_commands = _expand_quantity(cmd, object_detector, image, video_path=video_path)
        else:
            atomic_commands = [cmd]

        # --- Step 5: Execute each atomic command -------------------------
        for atomic in atomic_commands:
            result = _understand_single(
                atomic, video_path, output_dir, object_detector,
                device=device, torch_dtype=torch_dtype,
            )
            all_results.append(result)

            # Update state from result
            if result[0] == "OK":
                state.held_object = atomic.object
                state.last_object = atomic.object
                if atomic.target.text and atomic.target.text.strip():
                    state.last_target = atomic.target
                    state.held_object = None  # placed, no longer holding
                state.last_action = atomic.action
            else:
                # Failed task — clear held object to avoid cascading errors
                state.held_object = None

    # Keep object_detector alive (shared global)
    torch.cuda.empty_cache()

    return all_results

# ---------------------------------------------------------------------------
# MQTT listener
# ---------------------------------------------------------------------------

def on_message(client, userdata, message):
    logger.info("Message received: %s", message)
    file_paths = json.loads(message.payload.decode("utf-8"))
    audio_path = file_paths[0]
    video_path = file_paths[1]
    inference = file_paths[2]

    try:
        result = None
        if inference == "understand":
            results = understand(audio_path, video_path)
            logger.info("Result: %s", results)
            # Publish full list of task results — ROS side iterates sequentially.
            # Format: [["OK", obj, action, tgt, obj_center, tgt_center], ...]
            result = results if results else [["ambiguous", "no results"]]
        elif inference == "check_agree":
            result = [check_agree(audio_path)]
            logger.info("Responding with understanding: %s", result)
        client.publish("inference/response", json.dumps(str(result)))
    except Exception as e:
        logger.error("Understanding commands failed: %s", e)
        client.publish("inference/response", json.dumps(str([["ambiguous", str(e)]])))  # same wrapping as success path


client = mqtt.Client()
client.on_message = on_message
client.connect("localhost", 1883, 60)
client.subscribe("inference/request")

client.loop_forever()