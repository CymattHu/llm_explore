import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# add parent directory to sys.path
sys.path.insert(0, parent_dir)
import chroma
from knowledge_base_entry import KnowledgeBaseEntry
from gemini import Gemini
from image_processor import ImageProcessor
from utils import utils
import json
import bbox_selection
from tkinter import filedialog


image_path = "/home/chunyu.hu/Documents/Gemini/llm_as_planner/VLM/test/realsense_picture/build0_color.png"

prompt = "Detect the cylinder tea bottle bounding boxes of the picture"



gemini_bot = Gemini()
image_processor= ImageProcessor(image_path=image_path)
chroma_DB = chroma.ChromaDB()
utils_ = utils()

# get incorrect bounding boxes
image = image_processor.open_image()
gemimi_response = gemini_bot.generate_response(prompt, image)
json_output = utils_._load_json(gemimi_response.text)

print(json_output)

editor = bbox_selection.BoundingBoxEditor(image_path, None)
correct_bbox = editor.save_bboxes()

knowledge_base_entry = KnowledgeBaseEntry(chroma_DB.images_embedding(image_path).tolist(),
                                  json.loads(json_output)[0]["box_2d"],
                                  correct_bbox)


knowledge_base_entry.save_to_json("DB/test.json")