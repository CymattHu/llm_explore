from utils import utils
from gemini import Gemini
from image_processor import ImageProcessor
import os
import json
from sam import SAMSegmenter
import numpy as np
from PIL import Image,ImageDraw, ImageFont, ImageColor

gemini_bot = Gemini()
utils_ = utils()
sam_checkpoint ="/home/chunyu.hu/Documents/Gemini/llm_as_planner/sam_vit_h_4b8939.pth"
sam_segmenter = SAMSegmenter(sam_checkpoint,model_type="vit_h")

Image_folder =os.path.dirname(os.path.abspath(__file__))+"/test/server_slot"

prompt = "Detect rectangle HDD slot bounding boxes in the picture"

images_path = utils_._get_image_files(Image_folder)
for image_path in images_path:
    image_processor = ImageProcessor(image_path=image_path)
    image = image_processor.open_image()
    gemimi_response = gemini_bot.generate_response(prompt, image)
    print(f"Response: {gemimi_response.text}")
    json_output = utils_._load_json(gemimi_response.text)
    pixel_bbox= utils_._convert_gemini_bbox_to_pixels_bbox(json_output, image)
    if json_output is None:
        print("Failed to parse JSON output.")
        continue
    try:
        image_processor.plot_bounding_boxes(json.loads(json_output))
        # mask,scores, logits = sam_segmenter.segment_with_bbox(image,tuple(pixel_bbox))
        # best_mask = mask[np.argmax(scores)]
        # mask_image = Image.fromarray(best_mask)
        # image_with_mask = Image.composite(image, Image.new('RGB', image.size, (255, 0, 0)), mask_image)
        # image_with_mask.show()
    except Exception as e:
        print(f"Error while plotting bounding boxes: {e}")