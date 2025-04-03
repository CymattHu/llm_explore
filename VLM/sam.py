import torch
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import numpy as np

class SAMSegmenter:
    def __init__(self, model_checkpoint,model_type="vit_h"):
        """
        Initialize the SAMSegmenter class with a SAM model loaded from a checkpoint.
        
        :param model_checkpoint: Path to the SAM model checkpoint
        """
        # Load the model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = sam_model_registry[model_type](checkpoint=model_checkpoint).to(device)
        self.predictor = SamPredictor(self.model)

    def segment_with_bbox(self, image, bbox):
        """
        Segment the image using a bounding box as input.
        
        :param image_path: Path to the input image
        :param bbox: A tuple (x_min, y_min, x_max, y_max) representing the bounding box
        :return: A segmentation mask
        """
        # Load and preprocess the image
        # image = Image.open(image_path)
        image = np.array(image)

        # Set the image for prediction
        self.predictor.set_image(image)
        input_label = np.array([1])
        
        bbox_input = np.array(bbox)
        
        # Get the segmentation mask based on the bounding box
        segmentation_mask = self.predictor.predict(box=bbox_input,point_labels=input_label)

        return segmentation_mask
