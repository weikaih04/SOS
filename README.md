<h1 align="center">
  SOS: Synthetic Object Segments Improve Detection, Segmentation, and Grounding
</h1>

<h2 align="center">
  <!-- <a href="https://generate-any-scene.github.io/">🌐 Website</a> | -->
  <!-- <a href="https://arxiv.org/abs/2412.08221">📑 Paper</a> | -->
  <a href="https://huggingface.co/collections/weikaih/sos-synthetic-object-segments-improves-detection-segmentat-682679751d20faa20800033c">🤗  Dataset</a>
</h2>

<p align="center"><small>
  Weikai Huang<sup>1</sup>, Jieyu Zhang<sup>1</sup>,
  Taoyang Jia<sup>1</sup>, Chenhao Zheng<sup>1</sup>, Ziqi Gao<sup>1</sup>,
  Jae Sung Park<sup>1</sup>, Ranjay Krishna<sup>1,2</sup><br>
  <sup>1</sup>&nbsp University of Washington &nbsp 
  <sup>2</sup>Allen Institute for AI
</small></p>




<p align="center">
  <img src="./assets/teaser.png" alt="Text-to-Image Results" width="800">
</p>
<p align="center">A scalable pipeline for composing high-quality synthetic object segments into richly annotated images for object detection, instance segmentation, and visual grounding.</p>

# Installation

*Notice*: We provide only minimal guidance for the core parts of the codebase for: image composing, relighting and blending, and referring expression generation. The full documentation (with an accompanying arXiv paper) covering additional tasks and case studies will be released soon.

## Environment Setup
Follow the steps below to set up the environment and use the repository:
```bash
# Clone the repository
git clone https://github.com/weikaih04/SOS
cd ./SOS

# Create and activate a Python virtual environment:
conda create -n sos python==3.10
conda activate sos

# Install the required dependencies for composing images with synthetic object segments:
pip install -r requirements.txt

# If you want to perform relighting and blending:
conda create -n sos-relight python==3.10
conda activate sos-relight
pip install -r requirements_relight_and_blend.txt

# If you want to generating referring expression:
conda create -n sos-ref python==3.10
create activate sos-ref
pip install -r requirements_referring_expression_generation.txt
```


# Data Preparation

## For object segments dataset:
You can download the all the object segments dataset from: https://huggingface.co/collections/weikaih/sos-synthetic-object-segments-improves-detection-segmentat-682679751d20faa20800033c


## For background dataset:
If you want to relight images and didn't direclty pasting object segments into the background, just use the a random image as the background and set the `hasBackground` to false in the `generate_batch.py`
You can download the BG-20K from this repo: https://github.com/JizhiziLi/GFM.git

# Usage

## Composing synthetic images:
We provide the script to composing images with synthetic segments:
If you want to generate the images for relightening and blending that only contains the foreground object segments for the religting and blending later
```
python scripts/generate_with_batch.py \
    --num_processes 100 # depands on your cpus \
    --total_images 100000 \
    --filtering_setting filter_0 \
    --image_save_path "/output/dastaset_name/train" \
    --mask_save_path "/output/dastaset_name/panoptic_train" \
    --annotation_path "/output/dastaset_name/annotations" \
    --json_save_path "/output/dastaset_name/annotations/panoptic_train.json" 
```

If you want to generate the images that direclty paste the object onto the background, uncommend the `with bg process_image_worker` function in the `scripts/generate_with_batch.py` 


## Relighting and Blending
You can relight and blend the images with: `relighting_and_blending/inference.py` 

Currently it support google cloud storage access and local file system, 
you can run it with:
```
python relighting_and_blending/inference.py \
  --dataset_path "$DATASET_PATH" \
  --output_data_path "$OUTPUT_DATA_PATH" \
  --num_splits "$NUM_SPLITS" \
  --split "$SPLIT" \
  --index_json_path "" \
  --illuminate_prompts_path "$ILLUMINATE_PROMPTS_PATH" \
  --record_path "$RECORD_PATH"
```

## Referring Expression Generation
You can generate referring expressions with: `referring_expression_generation/inference.py` 


Currently it support google cloud storage access and local file system, 
you can run it with:
```
python inference.py "${TOTAL_JOBS}" "${JOB_INDEX}" "${INPUT_FILE}" "${OUTPUT_DIR}"
```



# Method
<p align="center">
  <img src="./assets/pipeline.png" alt="Text-to-Image Results" width="800">
</p>

1. **Object Segments Generation**  
   – Prompt a large diffusion model (FLUX-1) to render single-object images on a plain background.  
   – Extract clean masks with a segmentation model (DIS).  
   – Build a library of 20 M segments over both frequent (LVIS/COCO) and general categories.

2. **Object Selection & Layout Generation**  
   – Sample 5–20 segments per image, matching real-photo object-count distributions.  
   – Balanced‐category sampling to avoid head-class bias.  
   – Assign each segment to small/medium/large bins (40%/35%/25%) and enforce limited overlap.

3. **Relighting & Blending**  
   – **Global Relighting**: Apply IC-Light diffusion to harmonize illumination and suppress hard-edge artifacts.  
   – **Mask-Area-Weighted Blending**: Re-blend each segment with a learned weight ωᵢ ∈ [0,1] (higher for small objects) to preserve fine details and color fidelity.
   **Blending Comparison on LVIS-Mini**  
   - **Naive Paste**: direct alpha paste (hard edges, color mismatch)  
   - **IC-Light Only**: global relighting → AP = 36.3  
   - **IC-Light + Blending**: + mask-area-weighted re-blend → AP = 38.6 (**+2.3**)

4. **Ground Truth Generation**  
   – Compute final masks by subtracting occlusions from later-placed segments.  
   – Extract tight bounding boxes from each final mask.  
   – Generate 9+ referring expressions per image (attribute-, spatial-, and mixed-type) by prompting a language model with segment metadata.

<p align="center">
     <img src="./assets/comparison.png" alt="Relighting and Blending Comparison" width="800">
</p>


# Results

## Task 1: Open-Vocabulary Object Detection

<p align="center">
  <img src="./assets/ovd.png" alt="Text-to-Image Results" width="800">
</p>

- **Small amount of SOS efficiently brings strong gain.**  
  With only 50 K synthetic images, SOS boosts LVIS AP from 20.1 → 29.8 (+ 9.7) and AP<sub>rare</sub> from 10.1 → 23.5 (+ 13.4) 

- **Scaling up SOS data leads to better performance.**  
  Doubling to 100 K yields AP 31.0 (+ 1.2) and further scaling to 400 K yields AP 31.4 (+ 1.6) on LVIS and OdinW-35 mAP 22.8 (+ 1.8)  

- **SOS is complementary to real datasets.**  
  Mixing 100 K SOS with COCO + GRIT + V3Det raises LVIS AP from 31.9 → 33.2 (+ 1.3) and AP<sub>rare</sub> from 23.6 → 29.8 (+ 6.2) 

## Task 2: Visual Grounding

<p align="center">
  <img src="./assets/ref.png" alt="Text-to-Image Results" width="800">
</p>

- **Existing large detection and grounding datasets yield only marginal improvements.**  
  Adding V3Det or 20 M GRIT examples to Object365 + GoldG brings at most + 0.5 P@1 on gRefCOCO and + 1.4 mAP on DoD (FULL) 

- **SOS provides diverse, high-quality referring expressions that yield strong gains.**  
  SOS-50K improves gRefCOCO no-target accuracy by + 4.6 (89.3 → 93.9) and DoD (FULL) mAP by + 1.0; scaling to SOS-100K further adds + 8.4 no-target accuracy and + 3.8 mAP 

## Task 3: Instance Segmentation

<p align="center">
  <img src="./assets/results.png" alt="Text-to-Image Results" width="800">
</p>


- **SOS continuously improves LVIS segmentation.**  
  Fine-tuning APE on 50 K SOS then LVIS raises AP<sub>rare</sub> from 40.87 → 44.70 (+ 3.83), overall AP from 46.96 → 48.48 (+ 1.52), and AP<sub>frequent</sub> by + 0.31

## Task 4: Small-Vocabulary, Limited-Data Regimes
- **SOS excels in low-data regimes.**  
  Augmenting 1% of COCO with SOS yields a + 6.59 AP gain; this boost grows by ~ 3 points at 10%, 50%, and 100% COCO scales  

## Task 5: Intra-Class Referring Expression

<p align="center">
  <img src="./assets/intra-class.png" alt="Text-to-Image Results" width="800">
</p>

- **Targeted SOS data fixes intra-class shortcuts.**  
  Fine-tuning on 100 K SOS-SFC + SOS-SGC raises Average Gap by + 3.1 (37.5 → 40.6) and boosts Positive Gap Ratio to 90%  


### Ablation Study
- **Layout choice matters.**  
  Our layout (AP 9.16) outperforms random (9.07) and COCO-based (8.60).  

- **Relighting & blending are critical.**  
  Adding relighting & blending yields a + 39.7 % AP uplift (9.16 → 12.79).  

- **Segment quality impacts results.**  
  Real segments alone AP 7.03 → + Subject200K 12.06 → + SOS 12.79.  


<!-- *For figures, tables, and full details see the [paper PDF](./paper.pdf) and supplementary materials.*   -->

# Contact
* Weikai Huang: weikaih@cs.washington.edu
* Jieyu Zhang: jieyuz2@cs.washingtong.edu

# Citation
Bibtex: stay tuned for Arxiv!