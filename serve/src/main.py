import subprocess

subprocess.run(["sh", "install_multiscale_attn.sh"])
from dotenv import load_dotenv
import supervisely as sly
from serve.src.maskdino_model import MaskDinoModel


if sly.is_development():
    load_dotenv("local.env")
    load_dotenv("supervisely.env")

model = MaskDinoModel(
    use_gui=True,
    use_serving_gui_template=True,
)
model.serve()
