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

import time
t0=time.time()
from pathlib import Path
import torch
from uncom_utils.audio import AudioTranscriber#, separate_audio
from uncom_utils.image import (
    PointingDetector,
    ObjectDetector,
    Segmenter,
    DetectionResult,
    BoundingBox,
    DepthEstimator,
    SimilarityCalculator,
    annotate_action,
    annotate_image,
    extract_frame,
    load_image,
    pointed_result_index,
    voronoi_segmenting,
    #line_plane_intersection,
    minimum_distante_to_vector_line
)
from uncom_utils.text import CommandExtractor, check_relative_position, check_agreement
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from scipy.spatial import voronoi_plot_2d
import numpy as np 
import paho.mqtt.client as mqtt
import json

def check_agree(audio_path, device='auto'):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float32 if device == "cpu" else "auto"
    transcriber = AudioTranscriber(device=device, torch_dtype=torch_dtype)
    transcription = transcriber.transcribe(str(audio_path))
    return check_agreement(transcription['text'])

def understand(audio_path, video_path, device="auto"):

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Torch on CPU does not support half precision
    torch_dtype = torch.float32 if device == "cpu" else "auto"

    torch.cuda.empty_cache()

    print("Device:", device)

    transcriber = AudioTranscriber(device=device, torch_dtype=torch_dtype)

    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_dir = video_path.parent

    same_property = False # TODO: pick objects with similar properties to the pointed/refered one
    similar_object = True # TODO: implement selecting objects that look similar to the pointed object
    how_many_instances = 1 # TODO: implement setting number of similar objects. 
    multiple_objects = False # TODO: allow picking multiple objects

    # Transcribe the audio
    transcription = transcriber.transcribe(str(audio_path))
    # transcription = {'text': 'Take the Pringles can and put it on top of the coca cola.', 
    #                      'chunks': [{'text': ' Take', 'timestamp': (1.74, 3.2)},
    #                                 {'text': ' the', 'timestamp': (3.2, 3.5)},
    #                                 {'text': ' Pringles', 'timestamp': (3.5, 3.88)},
    #                                 {'text': ' can', 'timestamp': (3.88, 4.52)},
    #                                 {'text': ' and', 'timestamp': (4.52, 4.94)},
    #                                 {'text': ' put', 'timestamp': (4.94, 5.4)},
    #                                 {'text': ' it', 'timestamp': (5.4, 5.84)},
    #                                 {'text': ' on', 'timestamp': (5.84, 6.06)},
    #                                 {'text': ' top', 'timestamp': (6.06, 6.5)},
    #                                 {'text': ' of the', 'timestamp': (6.5, 6.8)},
    #                                 {'text': ' coca cola.', 'timestamp': (6.8, 7.18)}
    #                                 ]
    #                     }
    del transcriber
    torch.cuda.empty_cache()

    command_extractor = CommandExtractor(device=device, torch_dtype=torch_dtype)
    
    print("HEARD: ", transcription["text"])
    
    command = command_extractor.extract(transcription)
    
    if isinstance(command.object, list):
        if len(command.object)>1:
            multiple_objects = True
        else:
            command.object = command.object[0]

    print("Command:", command)
    command_path = output_dir / "command.json"
    command.save(command_path)
    print(f"Saved command to {command_path}")
    
    if command.object == '' or command.target == '' or len(command.object.timestamp)<1 or len(command.target.timestamp)<1:
        print("FAILURE 1: OBJECT OR TARGET NOT UNDERSTOOD")
        return ['ambiguous', "Failure 1: object or target missing"]

    # unload command_extractor model 
    del command_extractor
    torch.cuda.empty_cache()

    # Extract relevant frames from the video
    time = command.object.timestamp[1] if command.object.timestamp[1] else command.object.timestamp[0]
    object_frame_path = extract_frame(video_path, time)
    
    time = command.target.timestamp[1] if command.target.timestamp[1] else command.target.timestamp[0]
    target_frame_path = extract_frame(video_path, time)

    print(f"Extracted {command.object.timestamp[1]}s frame from {object_frame_path}")
    print(f"Extracted {command.target.timestamp[1]}s frame from {target_frame_path}")

    print("uncom_reasoner thinking stage 1")

    # Load images of the extracted frames
    object_image = load_image(object_frame_path)
    target_image = load_image(target_frame_path)
    print("uncom_reasoner thinking stage 2")
    image_width, image_height = object_image.size

    object_concrete = command.object.concrete
    target_concrete = command.target.concrete
    print("uncom_reasoner thinking stage 3")
    # Load oject detector model
    object_detector = ObjectDetector(device=device, torch_dtype=torch_dtype, detection_threshold=0.4)
    # Detect objects in the corresponding frames
    print("uncom_reasoner thinking stage 3.5")
    object_results = object_detector.detect(object_image, command.object.text)
    target_results = []
    print("uncom_reasoner thinking stage 3.75")
    relative_position = check_relative_position(command.action.text+command.target.text)
    print("uncom_reasoner thinking stage 4")
 
    if relative_position:
        print("uncom_reasoner thinking stage 5")
        reference_object, position = command.target.text, relative_position
        target_results = object_detector.detect(target_image, reference_object)

    else:
        print("uncom_reasoner thinking stage 6")
        target_results = object_detector.detect(target_image, command.target.text)

    print(f"Detected {len(object_results)} object instances of '{command.object.text}'")
    if len(target_results)>0:
        print(f"Detected {len(target_results)} target instances of '{command.target.text}'")
        print("uncom_reasoner thinking stage 7")
    else:
        print("uncom_reasoner thinking stage 8")
        print(f"\n\n\n'{command.target.text}' could not be detected.\n\n\n")
    #TODO: Here is the moment to handle concrete vs non-concrete targets
    print("uncom_reasoner thinking stage 9")
    # load hand detector model 
    hand_detector = PointingDetector()

    # It is necessary to check if it is impossible to understand what the object and/or what the target is, 
    # that is, object/target is not concrete and  no pointing vector could be detected. Robot should ask for
    # new set of instructions
    object_pointing_vec = hand_detector.detect(object_frame_path)
    object_pointing_detected = len(object_pointing_vec)>0
    
    target_pointing_vec = hand_detector.detect(target_frame_path)
    target_pointing_detected = len(target_pointing_vec)>0
 
    frame_count = 1
    print("uncom_reasoner thinking stage 10")
    while ((not object_pointing_detected) and frame_count<4): #If no hand was detected, try the next frame
        print("uncom_reasoner thinking stage 11")
        try:
            object_frame_path = extract_frame(video_path, command.object.timestamp[1]+frame_count*0.01) if command.object.timestamp[1] else extract_frame(video_path, command.object.timestamp[0]+frame_count*0.01)
            object_pointing_vec = hand_detector.detect(object_frame_path)
            object_pointing_detected = len(object_pointing_vec)>0

            if frame_count < 0:
                frame_count = -frame_count+1
                print("uncom_reasoner thinking stage 12")
            else:
                frame_count *= -1
                print("uncom_reasoner thinking stage 13")

        except Exception as e:
            print(f"ERROR {e}; failed to extract next frame.")
            print("uncom_reasoner thinking stage 14")
            break

    frame_count = 1
    print("uncom_reasoner thinking stage 15")
    while ((not target_pointing_detected) and frame_count<4): #If no hand was detected, try the next frame
        print("uncom_reasoner thinking stage 16")
        try:
            target_frame_path = extract_frame(video_path, command.target.timestamp[1]+frame_count*0.01) if command.target.timestamp[1] else extract_frame(video_path, command.target.timestamp[0]+frame_count*0.01)
            target_pointing_vec = hand_detector.detect(target_frame_path)
            target_pointing_detected = len(target_pointing_vec)>0

            if frame_count < 0:
                frame_count = -frame_count+1
                print("uncom_reasoner thinking stage 17")
            else:
                frame_count *= -1
                print("uncom_reasoner thinking stage 18")

        except Exception as e:
            print(f"ERROR {e}; failed to extract next frame.")
            print("uncom_reasoner thinking stage 19")
            break

    
    impossible_task = (not (object_concrete or object_pointing_detected) or  # checks if the object is not concrete and if no hands were detected
                       not (target_concrete or target_pointing_detected) or  # checks if the target is not concrete and if no hands were detected
                       (object_concrete and len(object_results)==0) or  # checks if the object is concrete but could not be identified
                       (target_concrete and len(target_results)==0))  # checks if the targect is concrete but could not be identified
    print("uncom_reasoner thinking stage 20")
    if impossible_task:
        print("uncom_reasoner thinking stage 21")
        print("FAILURE 2")
        return ["ambiguous", "failure 2: unclear object/target and no pointing detected"]
    print("uncom_reasoner thinking stage 22")
    pointed_object_idx = None
    object_pointing_vec = None
    pointed_target_idx = None
    target_pointing_vec = None
    
    if object_concrete:
        print("uncom_reasoner thinking stage 23")
    # If there are multiple objects detected, detect the pointing direction and choose the most likely one
        if len(object_results) > 1:
            print("uncom_reasoner thinking stage 24")
            object_pointing_vec = hand_detector.detect(object_frame_path)
            print(f"Detected object pointing {object_pointing_vec}")
            pointed_object_idx = pointed_result_index(object_results, object_pointing_vec)
        else:
            print("uncom_reasoner thinking stage 25")
            try:
                object_pointing_vec = hand_detector.detect(object_frame_path)
            except Exception as e:
                print(e)
                object_pointing_vec = np.array([float("inf"),float("inf"),float("inf"),])
            pointed_object_idx = 0
    else:  # non-concrete object cases
        print("uncom_reasoner thinking stage 26")
        try:
            object_pointing_vec = hand_detector.detect(object_frame_path)
            object_results = object_detector.detect(object_image, "pickable objects") # TODO: we can further speed it up by croping the image to the pointed region
            print("Pickable objects: ", object_results)
            pointed_object_idx = pointed_result_index(object_results, object_pointing_vec)
            print("Inferred object to be picked: ", object_results[pointed_object_idx])
            print("uncom_reasoner thinking stage 27")
        except Exception as e:
            print("FAILURE 3")
            print("uncom_reasoner thinking stage 28")
            return ["ambiguous", "failure 3: multiple objects detected, no pointing dected"]
    print("uncom_reasoner thinking stage 29")
    target_pointing_vec = hand_detector.detect(target_frame_path)
    
    # Target handling cases. There are 4 cases:
        # 1) target is a concrete object;
        # 2) target is described relatively to another object;
        # 3) target is an object described as "this" or "there";
        # 4) target is an empty space.
    print("uncom_reasoner thinking stage 30")
    area_target = False
    if "here" in command.target.text or "there" in command.target.text:
        area_target = True
        print("uncom_reasoner thinking stage 31")

    chosen_area = []
    if target_concrete:
        print("uncom_reasoner thinking stage 32")   

        pointed_target_idx = 0 
        if len(target_results) > 1 and len(target_pointing_vec)>0:
            print("uncom_reasoner thinking stage 33")
            print(f"Detected target pointing {target_pointing_vec}")
            pointed_target_idx = pointed_result_index(target_results, target_pointing_vec)
        elif len(target_results) == 1:
            print("uncom_reasoner thinking stage 34")
            pointed_target_idx = 0
        else:
            print("uncom_reasoner thinking stage 35")
            print("FAILURE 4")
            return ["ambiguous", "failure 4: multiple concrete targets, no pointing detected"]

        if relative_position: # Case 2), relative to an object
            print("uncom_reasoner thinking stage 36")
            table_bb = object_detector.detect(target_image, "table")[0].box
            table_cells = voronoi_segmenting(table_bb.xmax, table_bb.ymax, 300, table_bb.xmin, table_bb.ymin)
            table_cells_regions = [[table_cells.vertices[p] for p in r] for r in table_cells.regions]
            table_cells_regions = [r for r in table_cells_regions if len(r)>0]
            table_cell_centers =  [np.array(r).mean(axis=0).tolist() for r in table_cells_regions]
            reference_center = [(target_results[pointed_target_idx].box.xmax+target_results[pointed_target_idx].box.xmin)/2,
                             (target_results[pointed_target_idx].box.ymax+target_results[pointed_target_idx].box.ymin)/2]
            
            other_objects = object_detector.detect(object_image, "objects")

            other_objects_bb = []
            for o in other_objects:
                print("uncom_reasoner thinking stage 37")
                other_objects_bb.append([[o.box.xmin, o.box.ymin],
                                         [o.box.xmax, o.box.ymax],
                                         [o.box.xmin, o.box.ymax],
                                         [o.box.xmax, o.box.ymin]])

            occupancy_grid = [0]*len(table_cells_regions)
            print("uncom_reasoner thinking stage 38")
            for i, tc in enumerate(table_cells_regions):
                for object in other_objects_bb:
                    if Polygon(object).intersects(Polygon(tc)):
                        occupancy_grid[i]=1
            print("uncom_reasoner thinking stage 39")
            grid = list(zip(occupancy_grid, table_cell_centers, table_cells_regions))
            grid = [g for g in grid if not g[0]]
            obj_height = (target_results[pointed_target_idx].box.ymax-target_results[pointed_target_idx].box.ymin)/2
            obj_width = (target_results[pointed_target_idx].box.xmax-target_results[pointed_target_idx].box.xmin)/2
            print("uncom_reasoner thinking stage 40")
            if position in ["left"]:
                print("uncom_reasoner thinking stage 41")
                grid = [g for g in grid if g[1][0]<reference_center[0]-obj_width]

            elif position in ["right"]:
                print("uncom_reasoner thinking stage 42")
                grid = [g for g in grid if g[1][0]>reference_center[0]+obj_width]
            
            elif position in ["in front", "in front of", "up", "above", "over", "higher"]:
                print("uncom_reasoner thinking stage 43")
                grid = [g for g in grid if  g[1][1]>reference_center[1]+obj_height]
            
            elif position in ["behind", "down", "under", "above", "lower"]:
                print("uncom_reasoner thinking stage 44")
                grid = [g for g in grid if g[1][1]<reference_center[0]-obj_height]
            
            else: 
                print("uncom_reasoner thinking stage 45")
                grid = grid = [g for g in grid if g[1][0]>reference_center[0]+obj_width]+  [g for g in grid if g[1][0]<reference_center[0]-obj_width]
            #print("Reference `object and relative position: ", reference_object, relative_position)
            if len(grid)>0:
                print("uncom_reasoner thinking stage 46")
                area_target = True
                _, center , region = zip(*grid)
                try:
                    p1, p2 = target_pointing_vec
                    print("uncom_reasoner thinking stage 47")
                except ValueError:
                    print("uncom_reasoner thinking stage 48")
                    print("FAILURE 5")
                    return ["ambiguous", "failure 5: area target, but no pointing detected."]
                print("uncom_reasoner thinking stage 49")
                distances = [np.sqrt( (c[0]-p2[0])**2+(c[1]-p2[1])**2 ) for c in center]
                decision = list(zip(distances, region))
                decision.sort(key=lambda x:x[0])                
                chosen_area = decision[0][1]
                print("uncom_reasoner thinking stage 50")
            else:
                print("uncom_reasoner thinking stage 51")
                print("FAILURE 6")
                return ["ambiguous"]
        else:
            print("uncom_reasoner thinking stage 52")
            pass

    else:
        print("uncom_reasoner thinking stage 53")
        if not area_target:
            print("uncom_reasoner thinking stage 54") 
            target_results = object_detector.detect(object_image, "container") # TODO: we can further speed it up by croping the image to the pointed region
            print("Container objects: ", object_results)
            # target_results = [] #  TEST PURPOSES ONLY, comment/remove for final code.
            if len(target_results)>=1:  # case 1) or 3), we need to check if the user is pointing at an object.
                print("uncom_reasoner thinking stage 55")
                pointed_target_idx = pointed_result_index(target_results, target_pointing_vec)
                print("Inferred target object: ", target_results[pointed_target_idx])
            else:
                print("uncom_reasoner thinking stage 56")
                area_target = True

        if area_target: # if no objects are detected, target is an empty space; case 4)
            print("uncom_reasoner thinking stage 57")
            # load depth_estimator
            depth_estimator = DepthEstimator()
            depths = depth_estimator.estimate_depth(target_frame_path).cpu()
            depth_estimator.render_depth(depths, str(video_path.parent))
            # unload depth estimator
            del depth_estimator
            print("uncom_reasoner thinking stage 58")
            p1, p2 = target_pointing_vec
            p1_depth = depths[p1[1]][p1[0]] #  TODO: Verify if it is 0 and 1 or 1 and 0.
            p2_depth = depths[p2[1]][p2[0]] #  TODO: Verify if it is 0 and 1 or 1 and 0.
            p1 = p1.tolist()+[p1_depth]
            p2 = p2.tolist()+[p2_depth]

            print("Fingers: ", p1, p2)
            target_pointing_vec_3D = np.array(p2)-np.array(p1)
            print("uncom_reasoner thinking stage 59")
######################################################################################################

            table_bb = object_detector.detect(target_image, "table")[0].box
            table_cells = voronoi_segmenting(table_bb.xmax, table_bb.ymax, 400, table_bb.xmin, table_bb.ymin)
            table_cells_regions = [[table_cells.vertices[p] for p in r] for r in table_cells.regions]
            table_cells_regions = [r for r in table_cells_regions if len(r)>0]
            table_cell_centers =  [np.array(r).mean(axis=0).astype(np.uint16).tolist() for r in table_cells_regions]
            discard_outliers = zip(table_cells_regions, table_cell_centers)
            saved_voronois = []
            print("uncom_reasoner thinking stage 60")
            for c in discard_outliers:
                # print(123, c[1][0],c[1][1])
                if c[1][0]<=1079 and c[1][1]<=1919:
                    # print(45100, c[1][0],c[1][1])
                    saved_voronois.append(c)
            print("uncom_reasoner thinking stage 61")
            table_cells_regions, table_cell_centers = zip(*saved_voronois)
            table_cells_regions, table_cell_centers = list(table_cells_regions), list(table_cell_centers) 
            table_cell_centers_depth = []
            print("uncom_reasoner thinking stage 62")

            for i,c in enumerate(table_cell_centers):
                x = int(c[1])
                y = int(c[0])
                if x<0 or x> image_width-1 or y<0 or y>image_height-1:
                    print("uncom_reasoner thinking stage 63")
                    table_cell_centers_depth.append(float("inf"))
                else:
                    print("uncom_reasoner thinking stage 64")
                    table_cell_centers_depth.append(depths[y,x].numpy().tolist())
            print("uncom_reasoner thinking stage 65")
            for i, c in enumerate(table_cell_centers):
                table_cell_centers[i] = c+[table_cell_centers_depth[i]]
                print("uncom_reasoner thinking stage 66")

            other_objects = object_detector.detect(object_image, "objects")
            other_objects_bb = []
            for o in other_objects:
                print("uncom_reasoner thinking stage 67")
                other_objects_bb.append([[o.box.xmin, o.box.ymin],
                                         [o.box.xmax, o.box.ymax],
                                         [o.box.xmin, o.box.ymax],
                                         [o.box.xmax, o.box.ymin]])
            print("uncom_reasoner thinking stage 68")
            occupancy_grid = [0]*len(table_cells_regions)
            for i, tc in enumerate(table_cells_regions):
                for object in other_objects_bb:
                    if Polygon(object).intersects(Polygon(tc)):
                        print("uncom_reasoner thinking stage 69")
                        occupancy_grid[i]=1
            print("uncom_reasoner thinking stage 70")
            grid = list(zip(occupancy_grid, table_cell_centers, table_cells_regions))
            grid = [g for g in grid if not g[0]]

##########################################################################################################

            _, table_cell_centers, table_cells_regions = zip(*grid)
            chosen_area = table_cells_regions[minimum_distante_to_vector_line(p2, target_pointing_vec_3D, table_cell_centers)]
            print(chosen_area)
            print("uncom_reasoner thinking stage 71")

    del object_detector
    del hand_detector
    torch.cuda.empty_cache()
    
    if pointed_object_idx is None or (pointed_target_idx is None and chosen_area is []):
        print("uncom_reasoner thinking stage 72")
        print("FAILURE 7")
        return ['ambiguous', "failure 7: failed to identify pointed target/area"]
    print("uncom_reasoner thinking stage 73")
    # load segmenter model 
    segmenter = Segmenter(device=device, torch_dtype=torch_dtype)

    # Segment only the relevant (pointed at) objects in the corresponding frames
    # Yes... quite a strange destruction expression
    [object_results[pointed_object_idx]] = segmenter.segment(
        object_image, [object_results[pointed_object_idx]]
    )
    print("uncom_reasoner thinking stage 74")
    if area_target:
        print("uncom_reasoner thinking stage 75")
        x, y = zip(*chosen_area)
        target_results = [DetectionResult(score=1.0, label='target.', box=BoundingBox(xmin=int(min(x)), ymin=int(min(y)), xmax=int(max(x)), ymax=int(max(y))), mask=np.array(chosen_area).astype(np.uint8))]
        pointed_target_idx = 0       
    else:
        print("uncom_reasoner thinking stage 76")
        [target_results[pointed_target_idx]] = segmenter.segment(target_image, [target_results[pointed_target_idx]])
    print("uncom_reasoner thinking stage 77")
    print(f"Segmented object '{command.object.text}'")
    print(f"Segmented target '{command.target.text}'")
    print("uncom_reasoner thinking stage 78")
    # unload segmenter_model 
    del segmenter
    torch.cuda.empty_cache()
    print("uncom_reasoner thinking stage 79")
    # Annotate object image
    annotated_object_image = annotate_image(
        object_image, object_results, object_pointing_vec, emph_idx = pointed_object_idx
    )
    annotated_object_image_path = output_dir / "annotated_object.png"
    annotated_object_image.save(annotated_object_image_path)
    print(f"Saved annotated object image to {annotated_object_image_path}")

    # Annotate target image
    annotated_target_image = annotate_image(
        target_image, target_results, target_pointing_vec, emph_idx=pointed_target_idx
    )
    annotated_target_image_path = output_dir / "annotated_target.png"
    annotated_target_image.save(annotated_target_image_path)
    print(f"Saved annotated target image to {annotated_target_image_path}")

    # Produce a complete annotated action image
    caption = f"{command.object.text} - {command.action.text} - {command.target.text}"
    annotated_action = annotate_action(
        annotated_object_image, annotated_target_image, caption
    )
    name = str(video_path).replace('.mp4','')
    annotated_action_path = output_dir / f"{name}_annotated_action.png"
    annotated_action.save(annotated_action_path)
    print(f"Saved annotated action image to {annotated_action_path}")
    print("uncom_reasoner thinking stage 80")
    try:
        print("uncom_reasoner thinking stage 81")
        return ["OK", command.object.text, command.action.text, command.target.text, list(object_results[pointed_object_idx or 0].box.center), list(target_results[pointed_target_idx or 0].box.center)]
    
    except Exception as e:
        print("uncom_reasoner thinking stage 82")
        print (f"Failed to extract bouding box due to error: {e}")
        print("FAILURE 8")
        return ["ambiguous", "failure 8: failed to extract bounding box"]

def on_message(client, userdata, message):
    #file_paths = literal_eval(json.loads(message.payload.decode("utf-8")))
    print(f"Message: {message} received!")
    file_paths = json.loads(message.payload.decode("utf-8"))
    audio_path = file_paths[0]
    video_path = file_paths[1]
    inference = file_paths[2]
    try:
        result = None
        if inference == 'understand':
            result = understand(audio_path, video_path)
            print(result)
        elif inference == 'check_agree':
            result = [check_agree(audio_path)]
            print(f"\n#-------------------------------------------------------------#\nResponding with undestanding: {result}!\n#-------------------------------------------------------------#\n")
        client.publish("inference/response", json.dumps(str(result)))
    except Exception as e:
        print(f"Understanding commands failed due to {e}")
        client.publish("inference/response", json.dumps(["ambiguous"]))

client = mqtt.Client()
client.on_message = on_message
client.connect("localhost", 1883, 60)
client.subscribe("inference/request")

client.loop_forever()
