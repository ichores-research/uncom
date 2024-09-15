def understand(video):
    frames, audio = separate_audio(video)
    transcription = transcribe(audio, word_tiestamps=True)
    # Extracts the command structure from the transcription
    command_struct = extract_command(transcription)

    object_frame = extract_frame(frames, command_struct.object.timestamp.end)
    target_frame = extract_frame(frames, command_struct.target.timestamp.end)

    # Detector returns bounding boxes of objects
    object_candidate_bboxes = detect_objects(object_frame, command_struct.object.text)

    object_pointing_vector = detect_pointing_vector(object_frame)
    target_pointing_vector = detect_pointing_vector(target_frame)

    is_target_relative = is_relative_position(command_struct.action.text, command_struct.target.text)

    if is_target_relative:
        reference_object = command_struct.target.text
        relative_position = get_relation(command_struct.action.text, command_struct.target.text)
        target_candidate_bboxes = detect_objects(target_frame, reference_object)
    else:
        target_candidate_bboxes = detect_objects(target_frame, command_struct.target.text)

    if command_struct.object.is_concrete:
        if len(object_candidate_bboxes) > 1:
            object_index = select_object(object_candidate_bboxes, pointing_vector)
        else:
            object_index = 0
    else:
        object_candidate_bboxes = detect_objects(object_frame, "pickable objects")
        object_index = select_object(object_candidate_bboxes, pointing_vector)

    is_target_area = False

    if command_struct.target.is_concrete:
        if len(target_candidate_bboxes) > 1:
            target_index = select_object(target_candidate_bboxes, pointing_vector)
        else:
            target_index = 0

        if is_target_relative:
            is_target_area = True
            table = detect_objects(target_frame, "table")[0]
            voronoi = voronoi_partition(table)
            objects = detect_objects(target_frame, "objects")
            mark_occupied_regions(voronoi, objects)
            # Choose empty cell closest to the relative position of the object
            cell = choose_cell(voronoi, relative_position)
    else:
        target_candidate_bboxes = detect_objects(target_frame, "container")

        if len(target_candidate_bboxes) > 1:
            target_index = select_object(target_candidate_bboxes, pointing_vector)
        else:
            is_target_area = True
            depth = estimate_depth(target_frame)
            pointing_vector_3d = project_2d_to_3d(pointing_vector, depth)
            table = detect_objects(target_frame, "table")[0]
            voronoi = voronoi_partition(table)
            objects = detect_objects(target_frame, "objects")
            mark_occupied_regions(voronoi, objects)
            # Choose empty cell closest to the intersection of the pointing vector with the table
            cell = choose_cell_3d(voronoi, pointing_vector_3d)


    object_bbox = object_candidate_bboxes[object_index]

    if is_target_area:
        target_bbox = cell.bounding_box
    else:
        target_bbox = target_candidate_bboxes[target_index]

    object_pixels = segment_object(object_frame, object_bbox.center)
    target_pixels = segment_object(target_frame, target_bbox.center)

    return {
        object_name: command_struct.object.text,
        target_name: command_struct.target.text,
        action_name: command_struct.action.text,
        object_bbox: object_bbox,
        object_pixels: object_pixels,
        target_bbox: target_bbox,
        target_pixels: target_pixels
    }
