from utils import utils
from gemini import Gemini
from image_processor import ImageProcessor
import os
import json
import numpy as np
from PIL import Image,ImageDraw, ImageFont, ImageColor
import cv2
from lightglue import LightGlue, SuperPoint
import torch
from torchvision import transforms


def match_template(target, bboxes,template, threshold=0.1):
    # pil_target = Image.fromarray(target)
    # pil_template = Image.fromarray(template)
    transform = transforms.ToTensor()
    target = transform(target)
    template = transform(template)
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
        extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor

    matcher = LightGlue(features="superpoint").eval().to(device)
    feats_template = extractor.extract(template.to(device))
    template_h, template_w = template.shape[-2:]
    results = []
    for i,bbox in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, bbox) 
        patch = target[..., y1:y2, x1:x2] 
        feats_patch = extractor.extract(patch.to(device))
        with torch.no_grad():
                matches = matcher({"image0": feats_template, "image1": feats_patch})
                num_matches = (matches["matches"][0] > -1).sum().item()
                print(f"Number of matches: {num_matches}")
                
                if num_matches > 100:
                    matched_indices = matches['matches0']
                    template_matched_keypoints = feats_template['keypoints'][matched_indices[:, 0]]
                    patch_matched_keypoints = feats_patch['keypoints'][matched_indices[:, 1]]
                    template_matched_keypoints = template_matched_keypoints.squeeze(0) 
                    patch_matched_keypoints = patch_matched_keypoints.squeeze(0)
                    mean_template_coords = template_matched_keypoints.float().mean(dim=0)
                    mean_patch_coords = patch_matched_keypoints.float().mean(dim=0)
                    revised_coords = mean_patch_coords - mean_template_coords
                    results.append([revised_coords[0].item()+x1, revised_coords[1].item()+y1, revised_coords[0].item()+x1+template_w, revised_coords[1].item()+y1+template_h])
    return results


def match_template_with_homography(target, bboxes, template, threshold=0.1):
    transform = transforms.ToTensor()
    target = transform(target)
    template = transform(template)

    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
        matcher = LightGlue(features="superpoint").eval().to(device)

    feats_template = extractor.extract(template.to(device))
    template_keypoints = feats_template['keypoints'][0].cpu().numpy()
    template_h, template_w = template.shape[-2:]

    results = []

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, bbox)
        patch = target[..., y1:y2, x1:x2]
        feats_patch = extractor.extract(patch.to(device))
        patch_keypoints = feats_patch['keypoints'][0].cpu().numpy()

        with torch.no_grad():
            matches = matcher({"image0": feats_template, "image1": feats_patch})
            matches0 = matches["matches0"][0].cpu().numpy()  

            valid_mask = matches0 > -1
            if valid_mask.sum() < 100:
                continue 

            src_pts = patch_keypoints[matches0[valid_mask]] 
            dst_pts = template_keypoints[valid_mask]         
            # calue the homography matrix
            H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is None:
                continue  
            # Get the projected points
            projected_points = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), H)
            projected_points = projected_points.reshape(-1, 2)
            #get original patch coordinates
            patch_x_coords = src_pts[:, 0]
            patch_y_coords = src_pts[:, 1]
            patch_x_min, patch_y_min = patch_x_coords.min(), patch_y_coords.min()
            #get template coordinates
            template_x_coords = dst_pts[:, 0]
            template_y_coords = dst_pts[:, 1]
            template_x_min, template_y_min = template_x_coords.min(), template_y_coords.min()
            template_x_max, template_y_max = template_x_coords.max(), template_y_coords.max()
            #get projected points coordinates
            projected_x_coords = projected_points[:, 0]
            projected_y_coords = projected_points[:, 1]
            projected_x_min, projected_y_min = projected_x_coords.min(), projected_y_coords.min()
            projected_x_max, projected_y_max = projected_x_coords.max(), projected_y_coords.max()
            
            #calulate the scale of projection
            scale_x = (projected_x_max - projected_x_min) / (template_x_max - template_x_min)
            scale_y = (projected_y_max - projected_y_min) / (template_y_max - template_y_min)
            #calulate the revised bounding box coordinates
            results.append([
                x1+patch_x_min-projected_x_min,
                y1+patch_y_min-projected_y_min,
                x1+patch_x_min-projected_x_min+template_w*scale_x,
                y1+patch_y_min-projected_y_min+template_h*scale_y,
            ])

    return results

gemini_bot = Gemini()
utils_ = utils()

Image_folder =os.path.dirname(os.path.abspath(__file__))+"/test/dataset"

template = Image.open(os.path.dirname(os.path.abspath(__file__))+"/test/template.png").convert("RGB")

prompt = "Detect ethernet plug bounding boxes in the picture"

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
        # revised_bboxes=match_template(image, pixel_bbox, template)
        revised_bboxes = match_template_with_homography(image, pixel_bbox, template)
        print(f"Revised bounding boxes: {revised_bboxes}")
        revised_json_output = []
        for idx, box in enumerate(revised_bboxes):
            scale_w, scale_h = image.width / 1000, image.height / 1000
            new_box = [box[1], box[0], box[3], box[2]]  # 转换为 gemini 格式 (y_min, x_min, y_max, x_max)
            new_box = [int(coord / scale) for coord, scale in zip(new_box, [scale_h, scale_w, scale_h, scale_w])]
            revised_json_output.append({
                    "box_2d": new_box,
                    "label": str(idx+1)})
        print(f"Updated JSON output: {revised_json_output}")
        json_str = json.dumps(revised_json_output, indent=2)
        image_processor.plot_bounding_boxes(json.loads(json_str))
                
    except Exception as e:
        print(f"Error while plotting bounding boxes: {e}")