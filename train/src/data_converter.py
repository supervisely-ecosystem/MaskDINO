import supervisely as sly
import numpy as np
import os
import functools
from detectron2.data import DatasetCatalog, MetadataCatalog
from pathlib import Path
from PIL import Image


TRAIN_DATASET_NAME = "main_train"
VALIDATION_DATASET_NAME = "main_validation"


def reset_detectron_dataset(name):
    if name in DatasetCatalog:
        DatasetCatalog.remove(name)
    if name in MetadataCatalog:
        MetadataCatalog.remove(name)


def string2number(s):
    return int.from_bytes(s.encode(), "little")


def prepare_sem_seg_file(image, ann, save_path, classes):
    sem_seg = np.zeros(image.shape, dtype=np.uint8)
    labels = ann.labels
    for label in labels:
        class_name = label.obj_class.name
        class_idx = classes.index(class_name)
        geometry = label.geometry
        geometry.draw(bitmap=sem_seg, color=[class_idx, class_idx, class_idx])
    sem_seg = sem_seg[:, :, 0]
    mask = Image.fromarray(sem_seg)
    mask.save(save_path)


def convert_data_to_detectron(dataset, project_meta, seg_path, classes):
    records = []
    for item_name, image_path, ann_path in dataset.items():
        record = {"file_name": image_path, "image_id": string2number(item_name)}
        image = sly.image.read(image_path)
        width, height = image.shape[1], image.shape[0]
        record["height"] = height
        record["width"] = width
        ann = sly.Annotation.load_json_file(ann_path, project_meta)
        base_name = Path(item_name).with_suffix(".png")
        sem_seg_file_name = os.path.join(seg_path, base_name)
        prepare_sem_seg_file(image, ann, sem_seg_file_name, classes)
        record["sem_seg_file_name"] = sem_seg_file_name
        records.append(record)
    return records


def configure_datasets(train):
    project_path = os.path.join(train.work_dir, "sly_project")
    project = sly.Project(directory=project_path, mode=sly.OpenMode.READ)
    project_meta = project.meta

    train_ds_path = os.path.join(project_path, "train")
    val_ds_path = os.path.join(project_path, "val")

    train_ds, val_ds = sly.Dataset(train_ds_path, sly.OpenMode.READ), sly.Dataset(
        val_ds_path, sly.OpenMode.READ
    )

    train_seg_path = os.path.join(train.work_dir, "sly_project_seg/train")
    if not os.path.exists(train_seg_path):
        os.makedirs(train_seg_path)
    val_seg_path = os.path.join(train.work_dir, "sly_project_seg/val")
    if not os.path.exists(val_seg_path):
        os.makedirs(val_seg_path)

    get_train = functools.partial(
        convert_data_to_detectron, train_ds, project_meta, train_seg_path, train.classes
    )
    get_validation = functools.partial(
        convert_data_to_detectron, val_ds, project_meta, val_seg_path, train.classes
    )

    reset_detectron_dataset(TRAIN_DATASET_NAME)
    reset_detectron_dataset(VALIDATION_DATASET_NAME)

    DatasetCatalog.register(TRAIN_DATASET_NAME, get_train)
    DatasetCatalog.register(VALIDATION_DATASET_NAME, get_validation)

    MetadataCatalog.get(TRAIN_DATASET_NAME).stuff_classes = train.classes
    MetadataCatalog.get(TRAIN_DATASET_NAME).ignore_label = 255
    MetadataCatalog.get(VALIDATION_DATASET_NAME).stuff_classes = train.classes
    MetadataCatalog.get(VALIDATION_DATASET_NAME).ignore_label = 255
    MetadataCatalog.get(VALIDATION_DATASET_NAME).evaluator_type = "sem_seg"
