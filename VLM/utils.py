import os
from typing import List, Dict
import json



class utils:
    def __init__(self):
        pass
    def _get_image_files(self,folder_path) -> List[str]:
        """acquire image files from the folder."""
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"failed to read folder: {folder_path}")
        supported_formats = {".jpg", ".jpeg", ".png"}
        return [os.path.join(folder_path, file_name) 
               for file_name in os.listdir(folder_path)
               if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))]

    def _load_json(self,json_input) -> List[Dict]:
        """load json file."""
        try:
            if not json_input or not isinstance(json_input, str):
                raise ValueError("Invalid input: Expected a non-empty string")

            # parse JSON code block
            lines = json_input.splitlines()
            for i, line in enumerate(lines):
                if line.strip() == "```json":  # remove possible spaces
                    json_input = "\n".join(lines[i + 1:])  # remove "```json" before content
                    json_input = json_input.split("```")[0]  # remove "```" after content
                    break  # found "```json" and immediately exit the loop

            # ensure the final result is not an empty string
            if not json_input.strip():
                raise ValueError("Parsed JSON content is empty")

            return json_input

        except Exception as e:
            print(f"Error while parsing JSON output: {e}")
            return None  # return empty string ""
    def _convert_gemini_bbox_to_pixels_bbox(self,json_output, image):
        """
        Convert the 2D bounding box coordinates in a json output to pixel coordinates based on the image size.
        
        :param json_output: JSON output containing the 2D bounding box in normalized coordinates (list of dicts).
        :param image: PIL Image object to get the width and height for scaling.
        :return: Converted bounding box as (x1, y1, x2, y2) in pixel coordinates.
        """
        try:
            bboxes_pixel = []
            
            # Parse the JSON string to a Python list
            items = json.loads(json_output)

            # Calculate scaling factors based on the image size
            scale_w, scale_h = image.width / 1000, image.height / 1000

            for item in items:
                bbox = item.get("box_2d", None)
                if bbox is not None:
                    y1, x1, y2, x2 = bbox
                    x1, y1 = int(x1 * scale_w), int(y1 * scale_h)
                    x2, y2 = int(x2 * scale_w), int(y2 * scale_h)
                    bboxes_pixel.append((x1, y1, x2, y2))

            if not bboxes_pixel:
                raise ValueError("No 'box_2d' found in any items of the JSON output.")

            return bboxes_pixel


        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing JSON or extracting 'box_2d': {e}")
            return None
    
