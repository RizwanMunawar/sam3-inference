# SAM 3: Segment Anything 3

![sam3-architecture](assets/sam3-architecture.png)

SAM 3 is a unified foundation model for promptable segmentation in images and videos. It can detect, 
segment, and track objects using text or visual prompts such as points, boxes, and masks. Compared to its predecessor 
[SAM 2](https://github.com/facebookresearch/sam2), SAM 3 introduces the ability to exhaustively segment all instances of an open-vocabulary concept specified 
by a short text phrase or exemplars. Unlike prior work, SAM 3 can handle a vastly larger set of open-vocabulary prompts. 

https://github.com/user-attachments/assets/10f40eb1-379b-49a5-8f6f-4f3897f7851f

## Features supported (in this repo):

- [Inference on image](#inference-on-image-) 🚀
- [Auto annotation in YOLO format](#auto-annotation-) 🥳
- [Inference on video](#inference-on-video) 😍

## Prerequisites

- ✅ Python 3.12 or higher
- ✅ PyTorch 2.7 or higher
- ✅ CUDA 12.2 or greater (not necessarily required, you can also use CPU)

## Installation 👨‍💻

1. **Create a new virtual environment:**

    ```bash
    python3 -m venv "sam3test"
    source sam3test/bin/active
    ```

2. **Install PyTorch with CUDA support (CUDA>=12.2):**

    ```bash
    pip install torch==2.7.0 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu126
    ```

3. **Clone the repository and install the package:**

    ```bash
    git clone https://github.com/RizwanMunawar/sam3-inference
    cd sam3-inference
    pip install -e .
    ```

⚠️ **Note:** Access to the `sam3.pt` checkpoint must be requested via the SAM 3 Hugging Face [repository](https://huggingface.co/facebook/sam3).
Once your request is approved, you’ll be able to download and use the `sam3.pt` model for inference with the example shown below.

## Inference on Image 🎉

![image-inference-readme-demo.jpg](/assets/image-inference-demo.jpg)

```python
import cv2
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualize.utils import draw_box_and_masks

label_to_predict = "white dog"  # this will be used as prompt for inference.

url = "assets/images/dogs.jpg"
image = Image.open(url)  # Image load

# SAM3 model load
processor = Sam3Processor(build_sam3_image_model(checkpoint_path="sam3.pt"))

# Run inference with text prompt
results = processor.set_text_prompt(state=processor.set_image(image), 
                                    prompt=label_to_predict)

# Visualization
result_image = draw_box_and_masks(cv2.imread(url, cv2.COLOR_RGB2BGR),  # PIL -> OpenCV
                                  results=results,
                                  show_boxes=True,
                                  show_masks=True,
                                  line_width=4,
                                  label=label_to_predict)

cv2.imwrite("sam3_results.png", result_image)  # Save (optional)
```

## Auto annotation 🔥

![sam3 auto annotation workflow](assets/sam3-autoannotate.png)

```python
import os
import cv2
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualize.utils import draw_box_and_masks

# SAM3 model load (cpu inference also supported)
processor = Sam3Processor(build_sam3_image_model(checkpoint_path="sam3.pt"))

images_dir = "assets/images"
yolo_ann_dir = "assets/images/yolo_labels"
if not os.path.exists(yolo_ann_dir):
    os.mkdir(yolo_ann_dir)

# Auto annotation
label_to_predict = "bird"
for i, img in enumerate(os.listdir(images_dir)):
    url = os.path.join(images_dir, img)
    image = Image.open(url)  # Image load

    # Run inference with text prompt
    results = processor.set_text_prompt(state=processor.set_image(image),
                                        prompt=label_to_predict)

    # Visualization and auto annotation in YOLO format.
    result_image = draw_box_and_masks(
        cv2.imread(url, cv2.COLOR_RGB2BGR), # PIL -> OpenCV
        results=results,                    # SAM3 predictions
        show_boxes=True,                    # Display bounding boxes on output image
        show_masks=True,                    # Display masks on output image
        mask_alpha=0.4,                     # Adjust mask overlay value, range [0.0 - 1.0]
        show_conf=True,                     # Bool: display object confidence score.
        line_width=4,                       # Int: Adjust label, box, and mask fontsize.
        label=label_to_predict,             # Str: Bounding box/mask label
        save_yolo=True,                     # Bool: Write annotations in YOLO format.
        filename=os.path.join(yolo_ann_dir, img[:-4]+".txt"),  # Str: Annotation file name.
        class_id=0                          # Object ID, i.e., person class ID 0.
    )
    print(f"{i+1} Images processed, annotations saved in {yolo_ann_dir}")
```

## Inference on video 😍

⚠️  Currently, video processing runs frame-by-frame. This means the model does not retain object information from previous frames yet. 

```python
import cv2
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualize.utils import draw_box_and_masks

# === Settings ===
label_to_predict = "dog"
input_video = "path/to/video.mp4"
output_video = "output_sam3.avi"
model_path = "sam3.pt"

# === LOAD MODEL ===
print("[INFO] Loading SAM3 model...")
processor = Sam3Processor(build_sam3_image_model(checkpoint_path=model_path))

# === VIDEO CAPTURE ===
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("Error opening video file")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

frame_count = 0

# === PROCESS VIDEO ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"[INFO] Processing frame {frame_count}")

    # OpenCV (BGR) -> PIL (RGB)
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Run inference
    state = processor.set_image(image_pil)
    results = processor.set_text_prompt(state=state, prompt=label_to_predict)

    # Draw bbox + mask
    output_frame = draw_box_and_masks(frame, results=results, show_boxes=True,
                                      show_masks=True, line_width=3, label=label_to_predict)

    writer.write(output_frame)  # Write processed frame
  
# === CLEANUP ===
cap.release()
writer.release()
cv2.destroyAllWindows()
```

## License

This project is licensed under the SAM License - see the [LICENSE](LICENSE) file for details.

## References

- [SAM3 offical implementation](https://github.com/facebookresearch/sam3)
- [OpenCV repository](https://github.com/opencv/opencv)
