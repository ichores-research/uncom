import time
t0=time.time()
import argparse
import os
import shutil
import tempfile
from pathlib import Path
import torch
from uncom.audio import AudioTranscriber, separate_audio
from uncom.image import (
    PointingDetector,
    ObjectDetector,
    Segmenter,
    DetectionResult,
    BoundingBox,
    DepthEstimator,
    annotate_action,
    annotate_image,
    extract_frame,
    load_image,
    pointed_result_index,
    voronoi_segmenting,
    line_plane_intersection,
    closest_to_fingertip,
    minimum_distante_to_vector_line
)
from uncom.text import CommandExtractor, check_relative_position
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from scipy.spatial import voronoi_plot_2d
import numpy as np 

def understand(video_path, output_dir, device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Torch on CPU does not support half precision
    torch_dtype = torch.float32 if device == "cpu" else "auto"

    torch.cuda.empty_cache()

    print("Device:", device)

    # Load big models (now will be done on a per-need base to save memory)
    transcriber = AudioTranscriber(device=device, torch_dtype=torch_dtype)
#    command_extractor = CommandExtractor(device=device, torch_dtype=torch_dtype)
#    object_detector = ObjectDetector(device=device, torch_dtype=torch_dtype)
#    segmenter = Segmenter(device=device, torch_dtype=torch_dtype)
#    hand_detector = PointingDetector()

    is_tmp = output_dir is None

    # Create a temporary directory if not specified otherwise clean the given directory
    if is_tmp:
        output_dir = tempfile.mkdtemp()
        output_dir = Path(output_dir)
    else:
        output_dir = Path(output_dir)
        shutil.rmtree(output_dir)
        output_dir.mkdir()

    print("Output directory:", output_dir)
    print("Input path:", video_path)

    # Copy the file to the temp dir
    tmp_video_path = shutil.copy(video_path, output_dir)
    print(tmp_video_path)
    # Extract audio from the video
    tmp_audio_path = separate_audio(tmp_video_path)

    print("Separated audio to", tmp_audio_path)

    # Transcribe the audio
    transcription = transcriber.transcribe(tmp_audio_path)
    # transcription = {'text': ' Take this small orange fruit and put it right of the bowl.', 
    #                  'chunks': [{'text': ' Take', 'timestamp': (1.74, 3.2)},
    #                             {'text': ' this', 'timestamp': (3.2, 3.5)},
    #                             {'text': ' small', 'timestamp': (3.5, 3.88)},
    #                             {'text': ' orange', 'timestamp': (3.88, 4.52)},
    #                             {'text': ' fruit', 'timestamp': (4.52, 4.94)},
    #                             {'text': ' and', 'timestamp': (4.94, 5.4)},
    #                             {'text': ' put', 'timestamp': (5.4, 5.84)},
    #                             {'text': ' it', 'timestamp': (5.84, 6.06)},
    #                             {'text': ' right', 'timestamp': (6.06, 6.5)},
    #                             {'text': ' of the', 'timestamp': (6.5, 6.8)},
    #                             {'text': ' bowl.', 'timestamp': (6.8, 7.18)}
    #                             ]
    #                 }
    del transcriber
    torch.cuda.empty_cache()

    # Load command extractor model     
    command_extractor = CommandExtractor(device=device, torch_dtype=torch_dtype)
    # Extract the command

    command = command_extractor.extract(transcription)

    print("Command:", command)
    command_path = output_dir / "command.json"
    command.save(command_path)
    print(f"Saved command to {command_path}")
    
    # unload command_extractor model 
    del command_extractor
    torch.cuda.empty_cache()

    # Extract relevant frames from the video
    object_frame_path = extract_frame(tmp_video_path, command.object.timestamp[1])
    target_frame_path = extract_frame(tmp_video_path, command.target.timestamp[1])

    print(f"Extracted {command.object.timestamp[1]}s frame from {object_frame_path}")
    print(f"Extracted {command.target.timestamp[1]}s frame from {target_frame_path}")

    # Load images of the extracted frames
    object_image = load_image(object_frame_path)
    target_image = load_image(target_frame_path)

    object_concrete = command.object.concrete
    target_concrete = command.target.concrete


    # Load oject detector model
    object_detector = ObjectDetector(device=device, torch_dtype=torch_dtype)
    # Detect objects in the corresponding frames
    
    object_results = object_detector.detect(object_image, command.object.text)
    target_results = []
    relative_position = check_relative_position(command.action.text+command.target.text)

    if relative_position:
        reference_object, position = command.target.text, relative_position
        target_results = object_detector.detect(target_image, reference_object)

    else:
        target_results = object_detector.detect(target_image, command.target.text)

    print(f"Detected {len(object_results)} object instances of '{command.object.text}'")
    if len(target_results)>0:
        print(f"Detected {len(target_results)} target instances of '{command.target.text}'")
    else:
        print(f"\n\n\n'{command.target.text}' could not be detected.\n\n\n")
    #TODO: Here is the moment to handle concrete vs non-concrete targets


    # load hand detector model 
    hand_detector = PointingDetector()

    # It is necessary to check if it is impossible to understand what the object and/or what the target is, 
    # that is, object/target is not concrete and  no pointing vector could be detected. Robot should ask for
    # new set of instructions
    object_pointing_detected = len(hand_detector.detect(object_frame_path))>0
    target_pointing_detected = len(hand_detector.detect(target_frame_path))>0

    impossible_task = (not (object_concrete or object_pointing_detected) or  # checks if the object is not concrrte and if no hands were detected
                       not (target_concrete or target_pointing_detected) or  # checks if the target is not concrrte and if no hands were detected
                       (object_concrete and len(object_results)==0) or  # checks if the object is concrete but could not be identified
                       (target_concrete and len(target_results)==0))  # checks if the targect is concrete but could not be identified
    
    if impossible_task:
        print("I am deeply sorry, but I failed to understandyour instructions, could you please explain it again?") 
        exit() 
    
    if object_concrete:
    # If there are multiple objects detected, detect the pointing direction and choose the most likely one
        if len(object_results) > 1:
            object_pointing_vec = hand_detector.detect(object_frame_path)
            print(f"Detected object pointing {object_pointing_vec}")
            pointed_object_idx = pointed_result_index(object_results, object_pointing_vec)
        else:
            pointed_object_idx = object_results[0]
    else:  # non-concrete object cases
        try:
            object_pointing_vec = hand_detector.detect(object_frame_path)
            object_results = object_detector.detect(object_image, "pickable objects") # TODO: we can further speed it up by croping the image to the pointed region
            print("Pickable objects: ", object_results)
            pointed_object_idx = pointed_result_index(object_results, object_pointing_vec)
            print("Inferred object to be picked: ", object_results[pointed_object_idx])

        except Exception as e:
            print("Failed to understand which object should be picked due to:",  e)
            exit()

    target_pointing_vec = hand_detector.detect(target_frame_path)
    
    # Target handling cases. There are 4 cases:
        # 1) target is a concrete object;
        # 2) target is described relatively to another object;
        # 3) target is an object described as "this" or "there";
        # 4) target is an empty space.
    
    #target_concrete = False
    # relative_position = True     
    area_target = False
    chosen_area = []
    if target_concrete:   

        pointed_target_idx = 0 
        if len(target_results) > 1:
            print(f"Detected target pointing {target_pointing_vec}")
            pointed_target_idx = pointed_result_index(target_results, target_pointing_vec)
        elif len(target_results) == 1:
            pointed_target_idx = 0
        else: 
            print("Failed to detect target object due to: ",  e)
            exit()

        if relative_position: # Case 2), relative to an object

            table_bb = object_detector.detect(target_image, "table")[0].box
            table_cells = voronoi_segmenting(table_bb.xmax, table_bb.ymax, 400, table_bb.xmin, table_bb.ymin)
            table_cells_regions = [[table_cells.vertices[p] for p in r] for r in table_cells.regions]
            table_cells_regions = [r for r in table_cells_regions if len(r)>0]
            table_cell_centers =  [np.array(r).mean(axis=0).tolist() for r in table_cells_regions]
            reference_center = [(target_results[pointed_target_idx].box.xmax+target_results[pointed_target_idx].box.xmin)/2,
                             (target_results[pointed_target_idx].box.ymax+target_results[pointed_target_idx].box.ymin)/2]
            print(target_results[pointed_target_idx].box, reference_center)
            
            other_objects = object_detector.detect(object_image, "objects")
            # other_objects_contours = [ cv2.findContours((o.mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) for o in other_objects if o.mask is not None]

            other_objects_bb = []
            for o in other_objects:
                other_objects_bb.append([[o.box.xmin, o.box.ymin],
                                         [o.box.xmax, o.box.ymax],
                                         [o.box.xmin, o.box.ymax],
                                         [o.box.xmax, o.box.ymin]])

            occupancy_grid = [0]*len(table_cells_regions)
            
            for i, tc in enumerate(table_cells_regions):
                for object in other_objects_bb:
                    if Polygon(object).intersects(Polygon(tc)):
                        occupancy_grid[i]=1
            
            grid = list(zip(occupancy_grid, table_cell_centers, table_cells_regions))
            grid = [g for g in grid if not g[0]]
            obj_height = (target_results[pointed_target_idx].box.ymax-target_results[pointed_target_idx].box.ymin)/2
            obj_width = (target_results[pointed_target_idx].box.xmax-target_results[pointed_target_idx].box.xmin)/2
            if position in ["left"]:
                grid = [g for g in grid if g[1][0]<reference_center[0]-obj_width]

            elif position in ["right"]:
                grid = [g for g in grid if g[1][0]>reference_center[0]+obj_width]
            
            elif position in ["in front", "in front of", "up", "above", "over", "higher"]:
                grid = [g for g in grid if  g[1][1]>reference_center[1]+obj_height]
            
            elif position in ["behind", "down", "under", "above", "lower"]:
                grid = [g for g in grid if g[1][1]<reference_center[0]-obj_height]
            
            else: 
                grid = [g for g in grid if  g[1][0]>reference_center[0]] # TODO: decide what to do when the relative position could not be understood
            #print("Reference `object and relative position: ", reference_object, relative_position)
            if len(grid)>0:
                area_target = True
                _, center , region = zip(*grid)
                distances = [np.sqrt( (c[0]-reference_center[0])**2+(c[1]-reference_center[1])**2 ) for c in center]
                decision = list(zip(distances, region))
                decision.sort(key=lambda x:x[0])
                print(decision[0][0], decision[0][1])
                # img = plt.imread(target_frame_path)
                # fig, ax = plt.subplots()
                # ax.imshow(img, extent=[0, 1920, 0, 1080],origin="lower")    
                # voronoi_plot_2d(table_cells, ax=ax)
                chosen_area = decision[0][1]
                # x, y = zip(*decision[0][1])
                # ax.scatter([reference_center[0]], [reference_center[1]])
                # ax.fill(list(x),list(y),"r",alpha=0.3)
                # ax.set_xlim((0, 1920))
                # ax.set_ylim((0, 1080))
                # ax.axis('off')
                # plt.show()
            else:
                print("No free area could be detected, please, try again.")
                exit()
        else:
            pass

    else:   
        target_results = object_detector.detect(object_image, "container") # TODO: we can further speed it up by croping the image to the pointed region
        print("Container objects: ", object_results)
        # target_results = [] #  TEST PURPOSES ONLY, comment/remove for final code.
        if len(target_results)>=1:  # case 1) or 3), we need to check if the user is pointing at an object.
            pointed_target_idx = pointed_result_index(target_results, target_pointing_vec)
            print("Inferred target object: ", target_results[pointed_target_idx])
        else: # if no objects are detected, target is an empty space; case 4)
            area_target = True
            # load depth_estimator
            depth_estimator = DepthEstimator()
            depths = depth_estimator.estimate_depth(target_frame_path).cpu()
            depth_estimator.render_depth(depths)
            # unload depth estimator
            del depth_estimator

            p1, p2 = target_pointing_vec
            p1_depth = depths[p1[1]][p1[0]] #  TODO: Verify if it is 0 and 1 or 1 and 0.
            p2_depth = depths[p2[1]][p2[0]] #  TODO: Verify if it is 0 and 1 or 1 and 0.
            p1 = p1.tolist()+[p1_depth]
            p2 = p2.tolist()+[p2_depth]

            print("Fingers: ", p1, p2)
            target_pointing_vec_3D = np.array(p2)-np.array(p1)

            ######################################################################################################
            #                   TODO: Transform this into a function to increase readability                     #
            ######################################################################################################
            table_bb = object_detector.detect(target_image, "table")[0].box
            table_cells = voronoi_segmenting(table_bb.xmax, table_bb.ymax, 400, table_bb.xmin, table_bb.ymin)
            table_cells_regions = [[table_cells.vertices[p] for p in r] for r in table_cells.regions]
            table_cells_regions = [r for r in table_cells_regions if len(r)>0]
            table_cell_centers =  [np.array(r).mean(axis=0).astype(np.uint16).tolist() for r in table_cells_regions]
            discard_outliers = zip(table_cells_regions, table_cell_centers)
            saved_voronois = []
            for c in discard_outliers:
                print(123, c[1][0],c[1][1])
                if c[1][0]<=1079 and c[1][1]<=1919:
                    print(45100, c[1][0],c[1][1])
                    saved_voronois.append(c)

            table_cells_regions, table_cell_centers = zip(*saved_voronois)
            table_cells_regions, table_cell_centers = list(table_cells_regions), list(table_cell_centers) 
            table_cell_centers_depth = []

            for i,c in enumerate(table_cell_centers):
                x = int(c[1])
                y = int(c[0])
                table_cell_centers_depth.append(depths[y,x].numpy().tolist())

            for i, c in enumerate(table_cell_centers):
                table_cell_centers[i] = c+[table_cell_centers_depth[i]]

            other_objects = object_detector.detect(object_image, "objects")
            other_objects_bb = []
            for o in other_objects:
                other_objects_bb.append([[o.box.xmin, o.box.ymin],
                                         [o.box.xmax, o.box.ymax],
                                         [o.box.xmin, o.box.ymax],
                                         [o.box.xmax, o.box.ymin]])

            occupancy_grid = [0]*len(table_cells_regions)
            for i, tc in enumerate(table_cells_regions):
                for object in other_objects_bb:
                    if Polygon(object).intersects(Polygon(tc)):
                        occupancy_grid[i]=1
        
            grid = list(zip(occupancy_grid, table_cell_centers, table_cells_regions))
            grid = [g for g in grid if not g[0]]
            ##########################################################################################################
            _, table_cell_centers, table_cells_regions = zip(*grid)


            # distances = pointed_area(target_pointing_vec_3D, np.array(p1.tolist()+[p1_depth]), table_cell_centers)
            # chosen_area = table_cells_regions[closest_to_fingertip(p1, table_cell_centers)]
            chosen_area = table_cells_regions[minimum_distante_to_vector_line(p2, target_pointing_vec_3D, table_cell_centers)]
            # line_plane_intersection(p1, p2, list(zip(table_cell_centers, table_cells_regions)))

            # grid = list(zip(distances, table_cells_regions))    
            # grid.sort(key=lambda x:x[0])
            # chosen_area = grid[0][1]

            img = plt.imread("/home/robot/Code/uncom-non-concrete-handling/output_dir/depth.png")
            fig, ax = plt.subplots()
            ax.scatter([p1[0]], [p1[1]], c="r")
            ax.scatter([p2[0]], [p2[1]], c="b")
            ax.axline((p1[0], p1[1]), (p2[0], p2[1]), color='purple', label="Infinite line")
            ax.imshow(img, extent=[0, 1920, 0, 1080],origin="lower")    
            voronoi_plot_2d(table_cells, ax=ax)
            for r in table_cells_regions:
                x, y = zip(*r)    
                ax.fill(list(x),list(y),"g",alpha=0.3)
            x, y = zip(*chosen_area)
            ax.fill(list(x),list(y),"r",alpha=0.8)
            ax.set_xlim((0, 1920))
            ax.set_ylim((0, 1080))
            ax.axis('off')
            plt.show()
            print(chosen_area)
    # unload object detector model 
    # unload hand_detector

    del object_detector
    del hand_detector
    torch.cuda.empty_cache()
    
    # load segmenter model 
    segmenter = Segmenter(device=device, torch_dtype=torch_dtype)

    # Segment only the relevant (pointed at) objects in the corresponding frames
    # Yes... quite a strange destruction expression
    [object_results[pointed_object_idx]] = segmenter.segment(
        object_image, [object_results[pointed_object_idx]]
    )

    if area_target:
        x, y = zip(*chosen_area)
        target_results = [DetectionResult(score=1.0, label='target.', box=BoundingBox(xmin=int(min(x)), ymin=int(min(y)), xmax=int(max(x)), ymax=int(max(y))), mask=np.array(chosen_area).astype(np.uint8))]
        pointed_target_idx = 0       
    else:
        [target_results[pointed_target_idx]] = segmenter.segment(target_image, [target_results[pointed_target_idx]])
       

    print(f"Segmented object '{command.object.text}'")
    print(f"Segmented target '{command.target.text}'")


    # unload segmenter_model 
    del segmenter
    torch.cuda.empty_cache()


    # Annotate object image
    annotated_object_image = annotate_image(
        object_image, object_results, object_pointing_vec, emph_idx=pointed_object_idx
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
    annotated_action_path = output_dir / "annotated_action.png"
    annotated_action.save(annotated_action_path)
    print(f"Saved annotated action image to {annotated_action_path}")

    # Clean up the temp dir if was used
    if is_tmp:
        shutil.rmtree(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument(
        "-o", "--output-dir", default=None, help="output directory path"
    )
    parser.add_argument("--device", default="auto", help="device to use")
    args = parser.parse_args()

    understand(args.video, args.output_dir, args.device)

print("final time", time.time()-t0)