from ultralytics import YOLO
import os
import sys

export_img_size = (704, 1280) # Width and height need to be multples of 32
export_device = 0 #"dla:0" is another option, but good luck getting it to work.
export_batch = 1
export_workspace = None
export_format = "onnx" #"engine"
export_int8 = False
export_nms = False
export_simplify = True
export_dynamic = False

default_models_dir = os.path.join(os.path.curdir, "models")
default_coco_path = os.path.join(default_models_dir, "coco.yaml")
default_lenna_path = os.path.join(default_models_dir, "Lenna.png")

def Detect(model_filepath: str, image_filepath: str=default_lenna_path):
    success = False
    print(f"Attempting inference run of .engine file at path: {model_filepath}")
    
    try:
        if os.path.isfile(model_filepath):
            model = YOLO(model_filepath, task="detect")
            result = model.predict(image_filepath)
            print(str(result))
            success = True
    except Exception as e:
        msg = f"Failed to run inference on image with model at path {model_filepath}. Error: {str(e)}"
        print(msg)
    return success

export_args: dict={
	"workspace": export_workspace,
	"format": export_format,
	"batch": export_batch,
	"int8": export_int8,
	"data": default_coco_path,
	"imgsz": export_img_size,
	"device": export_device,
    "simplify": export_simplify,
    "nms": export_nms,
    "dynamic": export_dynamic
}

def ExportEngine(
    pt_filepath: str, engine_filepath: str, coco_filepath: str=default_coco_path):
    success = False
    print(f"Attempting export of .engine model file from .pt model file at path: {pt_filepath}")
    print(f"Export arguments:\n{str(export_args)}\n")
    try:
        model = YOLO(pt_filepath)
        model.export(
            workspace=export_workspace,
            format=export_format,
            batch=export_batch,
            int8=export_int8,
            data=default_coco_path,
            imgsz=export_img_size,
            device=export_device,
            simplify=export_simplify,
            nms=export_nms,
            dynamic=export_dynamic
        )
        success = os.path.isfile(engine_filepath)
    except Exception as e:
        msg = f"Unable to export .engine from .pt at path {pt_filepath}. Error: {str(e)}"
        print(msg)
    return success

if __name__ == "__main__":
    model_path: str = "models/yolo11l-seg.pt"
    models_dir: str = ""
    model_paths: dict = {}
    argc = len(sys.argv)
    if argc > 1:
        if os.path.isfile(sys.argv[1]):
            model_path =  sys.argv[1]
        if os.path.isdir(sys.argv[1]):
            models_dir = sys.argv[1]
    if model_path == "" and models_dir == "":
        models_dir = default_models_dir
    if models_dir != "":
        filenames = os.listdir(models_dir)
        for pt_filename in filenames:
            pt_filepath = os.path.join(models_dir, pt_filename)
            if os.path.isfile(pt_filepath):
                name_split = pt_filename.split('.')
                ext = name_split[-1]
                if ext.lower() == "pt":
                    model_name = name_split[0]
                    engine_filename = f"{model_name}.engine"
                    engine_filepath = os.path.join(models_dir, engine_filename)
                    if not os.path.isfile(engine_filepath):
                        model_paths[model_name] = [pt_filepath, engine_filepath]
                    else:
                        msg = f"Skipping file {pt_filename}, engine file already exists at path: {engine_filepath}!"
                        print(msg)      
    elif model_path != "":
        abs_path = os.path.abspath(model_path)
        models_dir == os.path.dirname(abs_path)
        pt_filename = os.path.basename(model_path)
        name_split = pt_filename.split('.')
        if name_split[-1] == "pt":
            model_name = name_split[0]
            engine_filename = f"{model_name}.engine"
            engine_filepath = os.path.join(models_dir, engine_filename)
            model_paths[model_name] = [model_path, engine_filepath]
    model_names = list(model_paths.keys())
    num_models = len(model_names)
    if num_models > 0:
        print(f"Found {num_models} .pt model files to convert in directory at path: {models_dir}")
        for model_name in model_names:
            [pt_filepath, engine_filepath] = model_paths[model_name]
            if not os.path.isfile(engine_filepath):
                if ExportEngine(pt_filepath, engine_filepath):
                    print(f"Successfully exported .engine file to path: {engine_filepath}")
                else:
                    print(f"Halting, engine export failed, suggest removing erroneous file: {engine_filepath}")
                    break
            if os.path.isfile(engine_filepath) and Detect(engine_filepath):
                print(f"Engine export valided, congratulations!")
            else:
                print(f"Halting, engine validation failed, suggest removing erroneous file: {engine_filepath}")
                break
            
print("Exiting, goodbye.")
exit(0)
