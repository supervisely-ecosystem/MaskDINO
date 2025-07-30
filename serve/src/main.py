import subprocess
import sys

subprocess.run(["sh", "install_multiscale_attn.sh"])
subprocess.run([sys.executable, "-m", "pip", "install", "Pillow==10.1.0"], check=True)
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
