import cv2
import numpy as np

# Vibrant color palette (BGR format for OpenCV)
COLOR_PALETTE = [
    (255, 42, 4),     # 0  - #042AFF   Deep Blue
    (235, 219, 11),   # 1  - #0BDBEB   Aqua Blue
    (183, 223, 0),    # 3  - #00DFB7   Teal Green
    (104, 31, 17),    # 4  - #111F68   Navy Blue
    (221, 111, 255),  # 5  - #FF6FDD   Pink/Magenta
    (79, 68, 255),    # 6  - #FF444F   Red-Pink
    (0, 237, 204),    # 7  - #CCED00   Lime Yellow-Green
    (68, 243, 0),     # 8  - #00F344   Neon Green
    (255, 0, 189),    # 9  - #BD00FF   Purple
    (255, 180, 0),    # 10 - #00B4FF   Sky Blue
    (186, 0, 221),    # 11 - #DD00BA   Magenta Purple
    (255, 255, 0),    # 12 - #00FFFF   Cyan
    (0, 192, 38),     # 13 - #26C000   Bright Green
    (147, 20, 255),   # 14 - #FF1493   DeepPink
]


def get_color(i):
    """Get color from palette"""
    return COLOR_PALETTE[i % len(COLOR_PALETTE)]


def save_yolo_annotations(boxes, imgw, imgh, class_id, filename):
    """
    Convert bounding boxes to YOLO format and append if file exists, otherwise create new.

    Parameters:
    - boxes: list of tuples [(x1, y1, x2, y2), ...] in pixel coordinates
    - imgw: width of the image
    - imgh: height of the image
    - class_id: integer class label
    - filename: name of the .txt file
    """
    # Prepare YOLO lines
    lines = []
    for (x1, y1, x2, y2) in boxes:
        center_x = ((x1 + x2) / 2) / imgw
        center_y = ((y1 + y2) / 2) / imgh
        width = (x2 - x1) / imgw
        height = (y2 - y1) / imgh
        lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

    # create annotation file
    with open(filename, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved {len(boxes)} annotations to {filename}")


def draw_box_and_masks(img_cv,
                       results,
                       show_boxes=True,
                       show_masks=True,
                       mask_alpha=0.35,
                       line_width=4,
                       show_conf=False,
                       show_label=True,
                       label=None,
                       save_yolo=False,
                       filename="None",
                       class_id = 0):
    """
    Fast and awesome visualization for SAM3 results

    Args:
        img_cv: Input image (BGR format)
        results: Dictionary with 'boxes', 'masks', 'scores'
        show_boxes: Whether to draw bounding boxes
        show_masks: Whether to draw masks
        mask_alpha: Transparency of mask overlay (0-1)
        show_conf: Whether to show confidence scores
        show_label: Whether to show label text
        line_width: Line width for bounding boxes (affects text size and padding)
        label: Bounding box or mask label.
        save_yolo: Save annotations in YOLO format
        filename: YOLO annotation TXT output filename
        class_id: Class_ID for YOLO.
    """
    if not show_boxes and not show_masks:
        return img_cv

    result = img_cv.copy()
    total_objects = len(results["scores"])
    h, w = result.shape[:2]

    # Adaptive font scale based on bbox size and line width
    font_scale = 0.31 * line_width + 0.10
    text_thickness = max(1, line_width // 2)  # Text thickness scales with line width

    for i in range(total_objects):
        color = get_color(i)

        # Draw masks first (so boxes appear on top)
        if show_masks:
            mask = results["masks"][i].squeeze(0).cpu().numpy()
            if mask.shape != (h, w):  # Handle mask dimensions
                if mask.shape == (w, h):
                    mask = mask.T
                else:
                    mask = cv2.resize(mask.astype(np.uint8), (w, h),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
            mask_bool = mask.astype(bool)

            overlay = result.copy()
            overlay[mask_bool] = color  # Apply semi-transparent overlay
            result = cv2.addWeighted(result, 1 - mask_alpha, overlay, mask_alpha, 0)

            # Draw contour for clarity
            contours, _ = cv2.findContours(mask_bool.astype(np.uint8) * 255,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, color, max(1, line_width // 2))

        # Draw bounding boxes
        if show_boxes:
            box = results["boxes"][i].cpu().numpy()
            x1, y1, x2, y2 = box.astype(int)
            prob = results["scores"][i].item()  # confidence score

            cv2.rectangle(result, (x1, y1), (x2, y2), color, max (1, (line_width-2)))  # bbox plotting

            if show_label:
                label = f"{label}:{prob:.2f}" if show_conf else f"{label}"

                (text_w, text_h), baseline = cv2.getTextSize(label,
                                                             cv2.FONT_HERSHEY_SIMPLEX,
                                                             font_scale,
                                                             text_thickness)
                padding = line_width * 2
                label_x, label_y = x1, y1 - padding - baseline  # Position label at top-left corner of box
                if label_y - text_h < 0:  # If label goes above image, put it inside the box
                    label_y = y1 + text_h + padding
                label_x = max(0, min(label_x, w - text_w - 2 * padding))

                # Draw label background
                cv2.rectangle(result,
                              (label_x, label_y - text_h - padding),
                              (label_x + text_w + 2 * padding, label_y + baseline + padding),
                              color, -1)

                # Draw label text
                cv2.putText(result, label, (label_x + padding, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (255, 255, 255), text_thickness, cv2.LINE_AA)

    # save YOLO annotations
    if save_yolo:
        save_yolo_annotations(results["boxes"], w, h, class_id, filename=filename)

    return result
