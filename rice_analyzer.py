# --- rice_analyzer.py ---
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Define constants
MIN_PIXEL_AREA = 50       # Lowered to capture small foreign debris
RICE_TYPICAL_LENGTH_MM = 5.0
IMAGE_SIZE = (150, 150)
FOREIGN_MATTER_SOL_THRESHOLD = 0.55 # If area/convex_hull_area is less than 55%, it's very irregular (like a scribble)

def classify_grain_cnn(grain_crop, cnn_model, class_names):
    """Classifies a single cropped grain image using the loaded CNN model."""
    try:
        img_resized = cv2.resize(grain_crop, IMAGE_SIZE)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_array = image.img_to_array(img_rgb)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = cnn_model.predict(img_array, verbose=0)
        
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_index]
        confidence = predictions[0][predicted_class_index]
        
        if confidence < 0.70:
            return "Good", confidence

        return predicted_class, confidence
    except Exception as e:
        return "Good", 0.0

def analyze_rice_image(img_cv, scale_factor, cnn_model, class_names):
    """
    Performs image processing and classification on the rice sample using CV and CNN.
    """
    analysis_img = img_cv.copy()
    hsv_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # 1. Segmentation
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 70, 255])
    mask = cv2.inRange(hsv_img, lower_white, upper_white)
    blurred_mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, thresh = cv2.threshold(blurred_mask, 150, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
    sure_bg = cv2.dilate(opening, kernel, iterations=5)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    markers = cv2.connectedComponents(sure_fg)[1]
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(analysis_img, markers)

    # 2. Measure all grains and classify
    total_grains = 0
    total_length_pixels = 0
    total_width_pixels = 0
    broken_count = 0
    dd_count = 0
    strip_count = 0
    tip_count = 0
    foreign_count = 0

    for marker_id in np.unique(markers):
        if marker_id <= 1: continue

        mask_single = np.zeros(gray_img.shape, dtype=np.uint8)
        mask_single[markers == marker_id] = 255
        contours, _ = cv2.findContours(mask_single, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            contour = contours[0]
            area = cv2.contourArea(contour)
            
            # Use the lower MIN_PIXEL_AREA threshold for the initial filter
            if area < MIN_PIXEL_AREA:
                continue

            rect = cv2.minAreaRect(contour)
            width_px, length_px = rect[1]
            if width_px > length_px: length_px, width_px = width_px, length_px

            length_mm = length_px * scale_factor
            width_mm = width_px * scale_factor

            # Filter out non-rice size items that survived MIN_PIXEL_AREA (e.g., tiny specks)
            if length_mm < 1.0 and width_mm < 1.0:
                continue

            total_grains += 1
            total_length_pixels += length_px
            total_width_pixels += width_px
            ratio = length_mm / width_mm if width_mm > 0 else 0
            
            # --- CLASSIFICATION FLAGS ---
            is_broken = False
            is_dd = False
            is_strip = False
            is_tip = False
            is_foreign = False
            color = (0, 255, 0) # Default: GREEN (Good)
            
            # Calculate Solidity for irregularity check
            solidity = 1.0
            if area > 0:
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
            
            # 1. Foreign Matter Check (Geometric Filter)
            # a. Extreme Aspect Ratio
            # b. Extreme Irregularity (Scribbles/Outlines)
            if ratio > 6.0 or ratio < 1.0 or solidity < FOREIGN_MATTER_SOL_THRESHOLD:
                is_foreign = True
                color = (255, 0, 0) # BLUE
            
            # 2. Broken Rice Check (Geometric Definition) - Only if not Foreign
            elif ratio < 1.75 and length_mm < RICE_TYPICAL_LENGTH_MM: 
                is_broken = True
                color = (0, 0, 255) # RED
            
            # 3. CNN Check for Defects - Only if Whole and not Foreign
            elif cnn_model is not None and len(class_names) > 0:
                
                # Crop the grain for CNN
                x, y, w, h = cv2.boundingRect(contour)
                buffer = 5
                x_c = max(0, x - buffer)
                y_c = max(0, y - buffer)
                w_c = min(img_cv.shape[1], x + w + buffer) - x_c
                h_c = min(img_cv.shape[0], y + h + buffer) - y_c
                grain_crop = img_cv[y_c:y_c+h_c, x_c:x_c+w_c]
                
                predicted_class, confidence = classify_grain_cnn(grain_crop, cnn_model, class_names)

                # Map CNN result to classification flags
                if predicted_class == "DD":
                    is_dd = True
                    color = (255, 255, 0) # YELLOW
                elif predicted_class == "Strip":
                    is_strip = True
                    color = (0, 255, 255) # CYAN
                elif predicted_class == "Tip":
                    is_tip = True
                    color = (255, 0, 255) # MAGENTA
                elif predicted_class == "Foreign_Matters":
                    is_foreign = True
                    color = (255, 0, 0) # BLUE


            # --- COUNTING AND VISUALIZATION ---
            if is_foreign:
                foreign_count += 1
            elif is_broken:
                broken_count += 1
            elif is_dd:
                dd_count += 1
            elif is_strip:
                strip_count += 1
            elif is_tip:
                tip_count += 1

            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(analysis_img, [box], 0, color, 2)

    # 4. Calculate Final Metrics (Code remains unchanged)
    total_defects = broken_count + dd_count + strip_count + tip_count + foreign_count
    good_rice = total_grains - total_defects
    if good_rice < 0: good_rice = 0
    
    if total_grains > 0:
        avg_length_mm = (total_length_pixels / total_grains) * scale_factor
        avg_width_mm = (total_width_pixels / total_grains) * scale_factor
        avg_ratio = avg_length_mm / avg_width_mm
        broken_percent = (broken_count / total_grains) * 100
        dd_percent = (dd_count / total_grains) * 100
    else:
        avg_length_mm, avg_width_mm, avg_ratio, broken_percent, dd_percent = 0, 0, 0, 0, 0

    results_dict = {
        'total_grains': total_grains,
        'avg_ratio': avg_ratio,
        'avg_length_mm': avg_length_mm,
        'avg_width_mm': avg_width_mm,
        'good_rice': good_rice,
        'broken_count': broken_count,
        'dd_count': dd_count,
        'strip_count': strip_count,
        'tip_count': tip_count,
        'foreign_count': foreign_count,
        'broken_percent': broken_percent,
        'dd_percent': dd_percent
    }

    return analysis_img, results_dict, total_grains