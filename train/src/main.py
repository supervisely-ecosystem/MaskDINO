import subprocess
import sys

subprocess.run(["sh", "install_multiscale_attn.sh"])
subprocess.run([sys.executable, "-m", "pip", "install", "Pillow==10.1.0"], check=True)
from train.src.custom_train_app import CustomTrainApp
from dotenv import load_dotenv
from train.src.data_converter import configure_datasets
from train.src.model_configuration import configure_trainer
from serve.src.maskdino_model import MaskDinoModel
import supervisely as sly
import os
import torch


load_dotenv("supervisely.env")
load_dotenv("local.env")


train = CustomTrainApp(
    framework_name="MaskDINO",
    models="models/models.json",
    hyperparameters="train/src/hyperparameters.yaml",
    app_options="train/src/app_options.yaml",
    work_dir="train_data",
)
train.register_inference_class(MaskDinoModel)


def clean_data():
    # delete app data since it is no longer needed
    sly.fs.remove_dir("train_data")
    sly.fs.remove_dir("train_output")


train.app.call_before_shutdown(clean_data)


@train.start
def start_training():
    # prepare training dataset
    configure_datasets(train)
    # prepare model
    trainer = configure_trainer(train)
    # train model
    output_dir = "./train_output"
    train.start_tensorboard(output_dir)
    trainer.train()
    # save class names to checkpoint
    best_checkpoint_path = os.path.join(output_dir, "model_final.pth")
    best_checkpoint_dict = torch.load(best_checkpoint_path)
    best_checkpoint_dict["class_names"] = train.classes
    torch.save(best_checkpoint_dict, best_checkpoint_path)
    # generate experiment info
    config_path = os.path.join(output_dir, "training_config.yaml")
    experiment_info = {
        "model_name": train.model_name,
        "model_files": {"config": config_path},
        "checkpoints": output_dir,
        "best_checkpoint": "model_final.pth",
        "task_type": "semantic segmentation",
    }
    return experiment_info
