import logging
import os
from typing import Union
import sys

logging.basicConfig(level=logging.WARNING) 

# 2. Create a specific logger for YOUR app
logger = logging.getLogger("my_app")
logger.setLevel(logging.INFO) # Your app will log INFO and above
logger.propagate = False # This prevents logs from "bubbling up" to the root logger

# 3. Add a FileHandler specifically to your logger
log_path = os.path.join(os.path.dirname(__file__), "app.log")
file_handler = logging.FileHandler(log_path, encoding="utf-8")
formatter = logging.Formatter("{asctime} - {levelname} - {message}", style="{", datefmt="%Y-%m-%d %H:%M")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def log_err(err_msg: Union[str | BaseException]) -> None:
    logger.error(str(err_msg))

def log_info(msg: str) -> None:
    logger.info(msg)

def get_file_name(pdf_path: str) -> str:
    return pdf_path.split("\\")[-1] if "win" in sys.platform else pdf_path.split("/")[-1]

def save_png(agent):
    png_data = agent.get_graph().draw_mermaid_png()
    with open(os.path.join("graph_pngs","graph.png"),"wb") as png:
        png.write(png_data)
    print("Image saved!")