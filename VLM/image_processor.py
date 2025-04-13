import random
from typing import List, Dict, Tuple
from PIL import Image,ImageDraw, ImageFont, ImageColor

class ImageProcessor:
    def __init__(self, image_path: str):
        """Initialize the ImageProcessor with an image path."""
        self.image_path = image_path
        self.image = Image.open(self.image_path).convert("RGB")
        if self.image is None:
            raise ValueError(f"failed to load image: {image_path}")
        # self.image.thumbnail([1024,1024], Image.Resampling.LANCZOS)
    
    def open_image(self):
        """Open the image using OpenCV."""
        return self.image

    def plot_bounding_boxes(self,boxes: List[Dict[str, Tuple[int, int, int, int]]], color: Tuple[int, int, int] = None, thickness: int = 2):
        """
        Draw bounding boxes on the image and display it.
        :param boxes: include a list of dictionaries, each with "box_2d" (x_min, y_min, x_max, y_max) and "label" fields
        :param color: bounding box color (R, G, B), if not specified, random colors are generated for different classes
        :param thickness: bounding box line thickness
        """
        image_copy = self.image.copy()
        width, height = image_copy.size
        color_map = {} 
        font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=70)
        

        for item in boxes:
            scale_w, scale_h = width / 1000, height / 1000
            y1, x1, y2, x2 = (int(coord * scale) for coord, scale in zip(item["box_2d"], [scale_h, scale_w, scale_h, scale_w]))
            label = item["label"]

            # if color is None, generate a random color for each label
            if color is None:
                if label not in color_map:
                    if image_copy.mode == "RGBA":
                        # color_map[label] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                        color_map[label] = (255, 0, 0)
                    else:
                        # color_map[label] = random.randint(0, 255) 
                        color_map[label] = 255             
                box_color = color_map[label]
            else:
                box_color = color
            # draw the bounding box
            draw = ImageDraw.Draw(image_copy)
            abs_x1, abs_x2 = min(x1, x2), max(x1, x2)
            abs_y1, abs_y2 = min(y1, y2), max(y1, y2)
            draw.rectangle([abs_x1, abs_y1, abs_x2, abs_y2], outline=box_color, width=thickness)
            draw.text((abs_x1, abs_y1 - 5), label, fill=box_color, font=font)
        image_copy.show()
            
