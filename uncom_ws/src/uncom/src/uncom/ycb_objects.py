import json
import os
import numpy as np
import yaml

# TODO real table estiamtion
TABLE_HEURISTIC = {
  "table corners": {"left back": [-0.49, 0.19, 2.04],"right back": [0.45, 0.18, 1.97],"left front": [-0.42, 0.24, 0.96],"right front": [0.45, 0.31, 1.00]}
}


def load_grasp_annotations(grasp_data_dir, config_file_path):
    """
    Load grasp annotations from a directory and a configuration file.
    Config yaml file contains object names and their IDs.
    """

    with open(config_file_path, 'r') as config_file:
        config_data = yaml.safe_load(config_file)
    
    names = config_data.get('names', {})
    grasp_annotations = {}

    for obj_id, obj_name in names.items():
        filename = f"obj_{int(obj_id):06d}.npy"
        file_path = os.path.join(grasp_data_dir, filename)
        if os.path.exists(file_path):
            grasps = np.load(file_path)
            grasp_annotations[obj_name] = {'grasps': grasps}
    return grasp_annotations


def get_ycb_objects_info(dataset):
    """
    Load YCB objects information including their IDs, diameters, grasps, and mesh paths.
    The dataset parameter specifies the dataset to load (ycb_ichores or ycbv)
    """

    with open(f"/root/catkin_ws/src/uncom/config/{dataset}.yaml", 'r') as file:
        id_to_name = yaml.safe_load(file)["names"]
        name_to_id = {v: k for k, v in id_to_name.items()}

    with open(f"/root/catkin_ws/src/uncom/config/models_info_{dataset}.json", 'r') as file:
        models_info = json.load(file)
        # The object parameters need to be adjusted to coordinate scale
        # which is 1000 times smaller
        # this info is based on taks/ scripts from ichores pipeline
        diameters = { int(model_id) : round( models_info[model_id]["diameter"] / 1000 , 2) for model_id in models_info}

    grasp_annotations = load_grasp_annotations(f"/root/catkin_ws/src/uncom/data/datasets/ycb_ichores/grasp_annotations", f"/root/catkin_ws/src/uncom/config/ycb_ichores.yaml")

    objects_info = { 
        name  : {
            "id" : id, 
            "diameter": diameters[id], 
            "grasps": grasp_annotations.get(name, None)['grasps'],
            "mesh_path": f"/root/catkin_ws/src/uncom/data/datasets/ycb_ichores/models/obj_{int(id):06d}.ply"
        } for name, id in name_to_id.items()
    }
    return objects_info