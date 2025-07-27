from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from maskdino import add_maskdino_config, MaskDINO
import supervisely as sly
from detectron2.data import MetadataCatalog
from supervisely.nn.inference import CheckpointInfo, ModelSource, RuntimeType, Timer
import numpy as np
from typing import List
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
import torch
from detectron2.structures import ImageList
import yaml
import os
from torch.nn import functional as F
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.memory import retry_if_cuda_oom


class MaskDinoModel(sly.nn.inference.SemanticSegmentation):
    FRAMEWORK_NAME = "MaskDINO"
    MODELS = "models/models.json"
    APP_OPTIONS = "serve/src/app_options.yaml"
    INFERENCE_SETTINGS = "serve/src/inference_settings.yaml"

    def load_model(
        self,
        model_files: dict,
        model_info: dict,
        model_source: str,
        device: str,
        runtime: str,
    ):
        config_path = model_files["config"]
        self.cfg = get_cfg()
        add_deeplab_config(self.cfg)
        add_maskdino_config(self.cfg)
        self.cfg.merge_from_file(config_path)

        checkpoint_path = model_files["checkpoint"]
        # if sly.is_development():
        #     checkpoint_path = "." + checkpoint_path
        self.cfg.MODEL.WEIGHTS = checkpoint_path
        self.cfg.MODEL.DEVICE = device
        self.device = device

        if runtime == RuntimeType.PYTORCH:
            self.cfg.freeze()
            self.model = MaskDINO(self.cfg)
            self.model.to(self.device)
            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(self.cfg.MODEL.WEIGHTS)
            self.model.eval()

        if model_source == ModelSource.PRETRAINED:
            self.checkpoint_info = CheckpointInfo(
                checkpoint_name=os.path.basename(checkpoint_path),
                model_name=model_info["meta"]["model_name"],
                architecture=self.FRAMEWORK_NAME,
                checkpoint_url=model_info["meta"]["model_files"]["checkpoint"],
                model_source=model_source,
            )
            dataset = model_info["Dataset"]
            if dataset == "ADE20K":
                self.classes = MetadataCatalog.get("ade20k_sem_seg_val").stuff_classes
            elif dataset == "Cityscapes":
                self.classes = MetadataCatalog.get(
                    "cityscapes_fine_sem_seg_val"
                ).stuff_classes
        else:
            self.classes = torch.load(checkpoint_path)["class_names"]

        obj_classes = [sly.ObjClass(name, sly.Bitmap) for name in self.classes]
        conf_tag = sly.TagMeta("confidence", sly.TagValueType.ANY_NUMBER)
        self._model_meta = sly.ProjectMeta(
            obj_classes=obj_classes, tag_metas=[conf_tag]
        )

        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST],
            self.cfg.INPUT.MAX_SIZE_TEST,
        )
        with open(self.INFERENCE_SETTINGS) as settings_file:
            self._custom_inference_settings = yaml.safe_load(settings_file)

    def predict_benchmark(self, images_np: List[np.ndarray], settings: dict = None):
        return self._predict_pytorch(images_np, settings)

    def _preprocess_input(self, images_np: List[np.ndarray]):
        orig_shapes = [
            {"height": img.shape[0], "width": img.shape[1]} for img in images_np
        ]
        images = [self.aug.get_transform(img).apply_image(img) for img in images_np]
        images = [
            torch.as_tensor(img.astype("float32").transpose(2, 0, 1)) for img in images
        ]
        images = [img.to(self.device) for img in images]
        images = [(x - self.model.pixel_mean) / self.model.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.model.size_divisibility)
        return images, orig_shapes

    def _predict_pytorch(self, images_np: List[np.ndarray], settings: dict = None):
        # 1. Preprocess
        with Timer() as preprocess_timer:
            images, orig_shapes = self._preprocess_input(images_np)
        # 2. Inference
        with Timer() as inference_timer:
            with torch.no_grad():
                features = self.model.backbone(images.tensor)
                outputs, _ = self.model.sem_seg_head(features)
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            mask_box_results = outputs["pred_boxes"]
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            processed_results = []
            for (
                mask_cls_result,
                mask_pred_result,
                mask_box_result,
                orig_shape,
                image_size,
            ) in zip(
                mask_cls_results,
                mask_pred_results,
                mask_box_results,
                orig_shapes,
                images.image_sizes,
            ):
                height = orig_shape["height"]
                width = orig_shape["width"]

                postprocess_before_inference = settings.get(
                    "postprocess_before_inference", False
                )

                if postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                r = retry_if_cuda_oom(self.model.semantic_inference)(
                    mask_cls_result, mask_pred_result
                )
                if not postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(
                        r, image_size, height, width
                    )
                processed_results.append(r)

        # 3. Postprocess
        with Timer() as postprocess_timer:
            predictions = self._format_predictions(processed_results, settings)

        benchmark = {
            "preprocess": preprocess_timer.get_time(),
            "inference": inference_timer.get_time(),
            "postprocess": postprocess_timer.get_time(),
        }
        return predictions, benchmark

    def _format_predictions(self, results, settings):
        predictions = []
        for result in results:
            result = result.argmax(dim=0).cpu().numpy().astype(np.uint8)
            predictions.append([sly.nn.PredictionSegmentation(result)])
        return predictions
