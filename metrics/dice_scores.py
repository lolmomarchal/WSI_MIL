# imports
import numpy as np
from scipy.ndimage import label, binary_fill_holes
import xml.etree.ElementTree as ET
import openslide
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import PIL
from PIL import Image


# ===================== ANNOTATION PROCESSING ==============================
def read_xml_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotations = []
    for annotation in root.findall(".//Annotation"):
        coordinates = []
        for coord in annotation.findall(".//Coordinate"):
            x = int(float(coord.attrib["X"]))
            y = int(float(coord.attrib["Y"]))
            coordinates.append((x,y))
        # get region name
        annotations.append({
                "name": annotation.attrib["Name"],
                "type": annotation.attrib["Type"],
                "group": annotation.attrib["PartOfGroup"],
                "color": annotation.attrib["Color"],
                "coordinates": np.array(coordinates)
            })
    return annotations
  
def get_separate_regions(annotations):
    # get tumor vs excluded regions 
    tumor = []
    exclusion = []
    for annotation in annotations:
        if annotation["group"] == "Tumor":
            tumor.append(annotation)
        else:
            exclusion.append(annotation)
    return exclusion, tumor
  
def convert_coordinates( coordinates, scale = 32):
    return coordinates//scale
  
def create_segmentation_mask(exclusion, tumor, slide_dimensions):
    mask = np.zeros(slide_dimensions[::-1], dtype=np.uint8) 
    def fill_polygon(annotations,color):
        for annotation in annotations:
            coords = convert_coordinates(annotation["coordinates"])
            cv2.fillPoly(mask, [coords], color = color)
    # for tumor region
    fill_polygon(tumor, color = 255)
    # for exclusion region
    fill_polygon(exclusion, color = 0)
    return mask
def create_mask_results(df, slide_dimensions, size = 512, scale = 32 ,label_column = "instance_label"):
    predictions = pd.read_csv(df)
    mask = np.zeros(slide_dimensions[::-1], dtype=np.uint8) 
    adjusted_size = size//scale
    x_coords = (predictions["x"] // scale).to_numpy()
    y_coords = (predictions["y"] // scale).to_numpy()
    labels = predictions[label_column].to_numpy()
    for x, y, label in zip(x_coords, y_coords, labels):
        mask[y:y+adjusted_size, x:x+adjusted_size] = 255 if label == 1 else 0
    
    return mask

# ========================== BORDERS =======================================
  def preprocess_mask(mask, min_object_size=512, max_hole_size=10):

    binary_mask = (mask > 0).astype(np.uint8)
    filled_mask = binary_fill_holes(binary_mask).astype(np.uint8)
    
    inverted_mask = 1 - filled_mask  
    labeled_holes, num_holes = label(inverted_mask)
    for hole_idx in range(1, num_holes + 1):
        if np.sum(labeled_holes == hole_idx) <= max_hole_size:
            filled_mask[labeled_holes == hole_idx] = 1  
    labeled_objects, num_objects = label(filled_mask)
    cleaned_mask = np.zeros_like(filled_mask)
    for obj_idx in range(1, num_objects + 1):
        if np.sum(labeled_objects == obj_idx) >= min_object_size:
            cleaned_mask[labeled_objects == obj_idx] = 1

    return (cleaned_mask * 255).astype(np.uint8)

def get_borders(result_mask):
    result_borders = np.zeros_like(result_mask)
    padded_mask = np.pad(result_mask, pad_width=1, mode="constant", constant_values=0)

    # neighbor regions
    left = padded_mask[1:-1, :-2]    
    right = padded_mask[1:-1, 2:]    
    top = padded_mask[:-2, 1:-1]     
    bottom = padded_mask[2:, 1:-1]   
    current = padded_mask[1:-1, 1:-1]  # current pixel

    #part of the shape but adjacent to the background
    border_condition = (
        (current == 255) & 
        ((left == 0) | (right == 0) | (top == 0) | (bottom == 0))  
    )

    result_borders[border_condition] = 255

    return result_borders

# ============================== DICE COEFF =============================
def dice_coefficient(y_pred, y_label):
    # first check that both are the same type + size
    assert y_pred.shape == y_label.shape, "Arrays must be the same size."
    assert y_pred.dtype == y_pred.dtype
    # equation for dice coeff is (2*intersection area)/(total area)
    intersection = np.sum(y_pred & y_label)
    return (2*intersection)/(np.sum(y_label) + np.sum(y_pred))
# ============================== MAIN ====================================
    
def main(xml_file, slide_path, df, column, output_path = None):
    # get written annotations
    annotations = read_xml_annotations(xml_file)
    exclusion, tumor = get_separate_regions(annotations)
    # get slide downscale
    slide = openslide.OpenSlide(slide_path)
    w,h = slide.dimensions
    w_d = w//32
    h_d = h//32
    thumbnail = slide.get_thumbnail((w_d, h_d))
    mask = create_segmentation_mask(exclusion, tumor, (w_d, h_d))
    results_mask = create_mask_results(df, (w_d, h_d),label_column = column )
    dice_coeff = dice_coefficient(results_mask, mask)
    dice_coeff_borders = dice_coefficient(get_borders(mask), get_borders(preprocess_mask(result_mask)))
    if output_path is not None:
        cv2.imwrite(os.path.join(output_path, "ground_truth_annotations.png"), mask)
        cv2.imwrite(os.path.join(output_path, "predicted_annotations.png"), results_mask)
    
    return dice_coeff, mask, results_mask
    
