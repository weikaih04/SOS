# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

#!/usr/bin/env python
"""
Modified version of inference_with_blending.py with GCS integration.
Supports downloading input images (and related JSON/masks) from GCS and uploading outputs to GCS,
similar to the provided flux pipeline code.
"""

import os
import re
import json
import time
import argparse
import numpy as np
from tqdm import tqdm
import tempfile

from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import torch
import safetensors.torch as sf

# Diffusers and related libraries
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from torch.hub import download_url_to_file

# Custom functions from your previous code (e.g., blending utilities)
from gradio_demo import (
    hooked_unet_forward, 
    encode_prompt_pair, 
    pytorch2numpy, 
    numpy2pytorch, 
    resize_and_center_crop, 
    resize_without_crop,
)

# --- GCS Utility Functions ---
from google.cloud import storage

def parse_gcs_path(gcs_path):
    """
    Given a GCS path in the form gs://bucket_name/path/to/file,
    returns a tuple (bucket_name, blob_path).
    """
    if not gcs_path.startswith("gs://"):
        raise ValueError("Not a valid GCS path")
    path = gcs_path[5:]
    parts = path.split("/", 1)
    bucket = parts[0]
    blob = parts[1] if len(parts) > 1 else ""
    return bucket, blob

def download_from_gcs(gcs_path, local_path):
    """
    Downloads a file from a GCS path to a local file.
    """
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    print(f"Downloaded {gcs_path} to {local_path}")

def upload_to_gcs(local_path, gcs_path):
    """
    Upload a local file to a GCS path.
    """
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to {gcs_path}")

def gcs_blob_exists(gcs_path):
    """
    Check if a blob exists at the given GCS path.
    """
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.exists()

def list_gcs_files(gcs_path, suffix=""):
    """
    List file names (relative to the prefix) from a GCS path with an optional suffix filter.
    """
    bucket_name, prefix = parse_gcs_path(gcs_path)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(client.list_blobs(bucket, prefix=prefix))
    file_names = []
    for blob in blobs:
        if blob.name.endswith(suffix):
            # Remove the prefix portion for clarity (if needed)
            relative_name = blob.name[len(prefix):].lstrip("/")
            file_names.append(relative_name)
    return file_names

def build_color_mask_gcs_path(data_path, file_id, original_folder="train", new_folder="panoptic_train"):
    """
    Builds the GCS path for the color mask by replacing the folder name
    in the blob portion without altering the bucket name.
    """
    if not data_path.startswith("gs://"):
        raise ValueError("Not a valid GCS path")
    # Remove the "gs://" prefix.
    path_without_prefix = data_path[5:]
    # Split into bucket and blob path.
    bucket, blob_path = path_without_prefix.split("/", 1)
    # Replace only the intended folder segment in blob_path.
    new_blob_path = blob_path.replace(original_folder, new_folder, 1)
    return f"gs://{bucket}/{new_blob_path}/{file_id}.png"

# --- Diffusion and Relightening Setup ---

from enum import Enum

class BGSource(Enum):
    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light" 
    TOP = "Top Light"
    BOTTOM = "Bottom Light"

# Set up Stable Diffusion components
sd15_name = 'stablediffusionapi/realistic-vision-v51'
tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")

# RMBG and UNet modifications
from briarmbg import BriaRMBG
rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

with torch.no_grad():
    new_conv_in = torch.nn.Conv2d(8, unet.conv_in.out_channels,
                                  unet.conv_in.kernel_size,
                                  unet.conv_in.stride,
                                  unet.conv_in.padding)
    new_conv_in.weight.zero_()
    new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    new_conv_in.bias = unet.conv_in.bias
    unet.conv_in = new_conv_in

unet_original_forward = unet.forward
unet.forward = hooked_unet_forward

# Load model offset and merge with UNet
model_path = './models/iclight_sd15_fc.safetensors'
if not os.path.exists(model_path):
    download_url_to_file(
        url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors',
        dst=model_path
    )
sd_offset = sf.load_file(model_path)
sd_origin = unet.state_dict()
sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
unet.load_state_dict(sd_merged, strict=True)
del sd_offset, sd_origin, sd_merged

# Set device and send models to device
device = torch.device('cuda')
text_encoder = text_encoder.to(device=device, dtype=torch.float16)
vae = vae.to(device=device, dtype=torch.bfloat16)
unet = unet.to(device=device, dtype=torch.float16)
rmbg = rmbg.to(device=device, dtype=torch.float32)

# SDP attention processor
unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# Define schedulers
ddim_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
euler_a_scheduler = EulerAncestralDiscreteScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    steps_offset=1
)
dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
    steps_offset=1
)

# Pipelines for text-to-image and image-to-image
t2i_pipe = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)
i2i_pipe = StableDiffusionImg2ImgPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)

# --- Utility Functions for Image Parsing and Blending ---

def parse_rgba(img, sigma=0.0):
    """
    Given a RGBA image (as a NumPy array), returns the blended RGB image using the alpha channel.
    """
    assert img.shape[2] == 4, "Input image must have 4 channels (RGBA)."
    rgb = img[:, :, :3]
    alpha = img[:, :, 3].astype(np.float32) / 255.0
    # set alpha to be like if it is transparent then 0, otherwise 1
    # alpha = np.where(alpha < 0.5, 0.0, 1.0)
    result = 127 + (rgb.astype(np.float32) - 127 + sigma) * alpha[:, :, None]
    # temporarily 
    return result.clip(0, 255).astype(np.uint8), alpha

def blend_images_with_mask_rank_sigmoid(old_image, new_image, color_mask, alpha_min=0.3, alpha_max=0.9, steepness=10):
    """
    Blends new_image into old_image according to a color mask using a sigmoid function based on segment area.
    """
    if not isinstance(old_image, Image.Image):
        old_image = Image.fromarray(old_image)
    if not isinstance(new_image, Image.Image):
        new_image = Image.fromarray(new_image)
    if not isinstance(color_mask, Image.Image):
        color_mask = Image.fromarray(color_mask)

    old_image = old_image.convert("RGBA")
    new_image = new_image.convert("RGBA")
    color_mask = color_mask.convert("RGBA")

    old_np = np.array(old_image).astype(np.float32)
    new_np = np.array(new_image).astype(np.float32)
    mask_np = np.array(color_mask).astype(np.float32)
    mask_rgb = mask_np[..., :3]
    total_pixels = mask_rgb.shape[0] * mask_rgb.shape[1]

    unique_colors = np.unique(mask_rgb.reshape(-1, 3), axis=0)
    unique_colors = [c for c in unique_colors if not np.allclose(c, [0, 0, 0])]

    color_to_norm_area = {}
    for color in unique_colors:
        region = ((mask_rgb[..., 0] == color[0]) &
                  (mask_rgb[..., 1] == color[1]) &
                  (mask_rgb[..., 2] == color[2]))
        area = np.sum(region)
        norm_area = area / total_pixels
        color_to_norm_area[tuple(color)] = norm_area

    segments_sorted = sorted(color_to_norm_area.items(), key=lambda x: x[1])
    total_segments = len(segments_sorted)
    color_to_alpha = {}
    for rank, (color, norm_area) in enumerate(segments_sorted):
        normalized_rank = rank / (total_segments - 1) if total_segments > 1 else 0
        sigmoid_value = 1 / (1 + np.exp(steepness * (normalized_rank - 0.5)))
        alpha = alpha_min + (alpha_max - alpha_min) * sigmoid_value
        color_to_alpha[color] = alpha
        print(f"Segment color: {color}, Norm Area: {norm_area:.6f}, Rank: {rank}, Normalized Rank: {normalized_rank:.3f}, Alpha: {alpha}")

    # Convert images from RGBA numpy arrays to Lab color space for lightness-only blending
    rgb_old = old_np[..., :3] / 255.0
    rgb_new = new_np[..., :3] / 255.0
    lab_old = rgb2lab(rgb_old)
    lab_new = rgb2lab(rgb_new)
    lab_out = lab_old.copy()

    # Blend only the L channel per segment while preserving original a/b channels
    for color, alpha in color_to_alpha.items():
        region = ((mask_rgb[..., 0] == color[0]) &
                  (mask_rgb[..., 1] == color[1]) &
                  (mask_rgb[..., 2] == color[2]))
        # Update lightness channel
        lab_out[..., 0][region] = alpha * lab_old[..., 0][region] + (1 - alpha) *  lab_new[..., 0][region]
        # Lightly blend chroma channels (a and b) to avoid unrealistic color shifts
        lab_out[..., 1][region] = 0.25 * alpha * lab_old[..., 1][region] + (1 - 0.25 * alpha) * lab_new[..., 1][region]
        lab_out[..., 2][region] = 0.25 * alpha * lab_old[..., 2][region] + (1 - 0.25 * alpha) * lab_new[..., 2][region]
        # lab_out[..., 1][region] = lab_old[..., 1][region]
        # lab_out[..., 2][region] = lab_old[..., 2][region]
        lab_out[..., 1][region] = lab_new[..., 1][region]
        lab_out[..., 2][region] = lab_new[..., 2][region]
    # Convert Lab back to RGB
    rgb_out = lab2rgb(lab_out)
    rgb_out_uint8 = (rgb_out * 255).clip(0, 255).astype(np.uint8)
    blended_image = Image.fromarray(rgb_out_uint8)
    return blended_image

# --- Inference Functions ---

@torch.inference_mode()
def process(input_fg, prompt, image_width, image_height, num_samples,
            seed, steps, a_prompt, n_prompt, cfg,
            highres_scale, highres_denoise, lowres_denoise, bg_source):
    bg_source = BGSource(bg_source)
    input_bg = None
    if bg_source == BGSource.NONE:
        pass
    elif bg_source == BGSource.LEFT:
        gradient = np.linspace(255, 0, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.RIGHT:
        gradient = np.linspace(0, 255, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.TOP:
        gradient = np.linspace(255, 0, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.BOTTOM:
        gradient = np.linspace(0, 255, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    else:
        raise ValueError("Wrong initial latent!")

    rng = torch.Generator(device=device).manual_seed(int(seed))
    fg = resize_and_center_crop(input_fg, image_width, image_height)
    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    conds, unconds = encode_prompt_pair(positive_prompt=prompt + ', ' + a_prompt, negative_prompt=n_prompt)

    if input_bg is None:
        latents = t2i_pipe(
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=steps,
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type='latent',
            guidance_scale=cfg,
            cross_attention_kwargs={'concat_conds': concat_conds},
        ).images.to(vae.dtype) / vae.config.scaling_factor
    else:
        bg = resize_and_center_crop(input_bg, image_width, image_height)
        bg_latent = numpy2pytorch([bg]).to(device=vae.device, dtype=vae.dtype)
        bg_latent = vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor
        latents = i2i_pipe(
            image=bg_latent,
            strength=lowres_denoise,
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=int(round(steps / lowres_denoise)),
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type='latent',
            guidance_scale=cfg,
            cross_attention_kwargs={'concat_conds': concat_conds},
        ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels)
    pixels = [resize_without_crop(
        image=p,
        target_width=int(round(image_width * highres_scale / 64.0) * 64),
        target_height=int(round(image_height * highres_scale / 64.0) * 64))
        for p in pixels]
    pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
    latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
    latents = latents.to(device=unet.device, dtype=unet.dtype)
    image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8
    fg = resize_and_center_crop(input_fg, image_width, image_height)
    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    latents = i2i_pipe(
        image=latents,
        strength=highres_denoise,
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=image_width,
        height=image_height,
        num_inference_steps=int(round(steps / highres_denoise)),
        num_images_per_prompt=num_samples,
        generator=rng,
        output_type='latent',
        guidance_scale=cfg,
        cross_attention_kwargs={'concat_conds': concat_conds},
    ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample
    return pytorch2numpy(pixels)

@torch.inference_mode()
def process_relight(input_fg, prompt, image_width, image_height, num_samples,
                     seed, steps, a_prompt, n_prompt, cfg,
                     highres_scale, highres_denoise, lowres_denoise, bg_source):
    input_fg, _ = parse_rgba(input_fg)
    results = process(input_fg, prompt, image_width, image_height, num_samples,
                      seed, steps, a_prompt, n_prompt, cfg,
                      highres_scale, highres_denoise, lowres_denoise, bg_source)
    return input_fg, results

def adjust_dimensions(width, height, max_dim=1024, divisible_by=8):
    # For simplicity, we return a fixed dimension (or you can implement your resizing logic)
    return 1024, 1024

# --- Main Inference Workflow with GCS Integration ---

def main(args):
    data_path = args.dataset_path
    output_data_path = args.output_data_path
    illuminate_prompts_path = args.illuminate_prompts_path
    record_path = args.record_path

    # Determine if the dataset, index JSON, and output_data_path reside on GCS.
    input_on_gcs = data_path.startswith("gs://")
    output_on_gcs = output_data_path.startswith("gs://")
    index_on_gcs = args.index_json_path is not None and args.index_json_path.startswith("gs://")
    illuminate_on_gcs = illuminate_prompts_path.startswith("gs://") if illuminate_prompts_path else False

    # Create a local temporary directory to use when downloading files from GCS.
    temp_dir = tempfile.mkdtemp()

    # If the illumination prompts are on GCS, download them locally.
    if illuminate_on_gcs:
        local_illuminate_path = os.path.join(temp_dir, "illumination_prompt.json")
        download_from_gcs(illuminate_prompts_path, local_illuminate_path)
        illuminate_prompts_path = local_illuminate_path

    with open(illuminate_prompts_path, "r") as f:
        illuminate_prompts = json.load(f)

    records = {}
    split_index = args.split
    num_splits = args.num_splits

    # Prepare list of filenames based on index JSON if provided; otherwise list .png files in dataset.
    if args.index_json_path:
        # If index JSON is on GCS, download it.
        if index_on_gcs:
            local_index_path = os.path.join(temp_dir, "index.json")
            download_from_gcs(args.index_json_path, local_index_path)
            index_json_path = local_index_path
        else:
            index_json_path = args.index_json_path

        with open(index_json_path, 'r') as f:
            all_filenames = json.load(f)
        if not isinstance(all_filenames, list):
            raise ValueError("The index JSON file must contain a list of filenames.")
        splits = np.array_split(all_filenames, num_splits)
        split_filenames = list(splits[split_index])
        print(f"Processing split {split_index + 1}/{num_splits} with {len(split_filenames)} images from index JSON.")
    else:
        if input_on_gcs:
            # List .png files from the GCS path.
            bucket_name, prefix = parse_gcs_path(data_path)
            files = list_gcs_files(data_path, suffix=".png")
            # Optionally, sort numerically if filenames are numbers.
            pattern = re.compile(r'^(\d+)\.png$')
            file_numbers = []
            for f in files:
                m = pattern.match(os.path.basename(f))
                if m:
                    file_numbers.append((int(m.group(1)), f))
            file_numbers.sort(key=lambda x: x[0])
            sorted_filenames = [f for _, f in file_numbers]
            splits = np.array_split(sorted_filenames, num_splits)
            split_filenames = list(splits[split_index])
            print(f"Processing split {split_index + 1}/{num_splits} with {len(split_filenames)} images from GCS.")
        else:
            # Local directory listing.
            all_files = [f for f in os.listdir(data_path) if f.endswith(".png")]
            pattern = re.compile(r'^(\d+)\.png$')
            filtered_files = []
            for f in all_files:
                m = pattern.match(f)
                if m:
                    numeric_value = int(m.group(1))
                    filtered_files.append((numeric_value, f))
            filtered_files.sort(key=lambda x: x[0])
            sorted_filenames = [f for _, f in filtered_files]
            splits = np.array_split(sorted_filenames, num_splits)
            split_filenames = list(splits[split_index])
            print(f"Processing split {split_index + 1}/{num_splits} with {len(split_filenames)} images (local).")

    # Prepare the output destination.
    if not output_on_gcs:
        os.makedirs(output_data_path, exist_ok=True)

    # Process each file.
    for fg_name in tqdm(split_filenames, desc="Processing images"):
        # For input image, if residing on GCS, construct full GCS path and download to temp file.
        if input_on_gcs:
            full_fg_path = os.path.join(data_path, fg_name)
            local_fg_path = os.path.join(temp_dir, os.path.basename(fg_name))
            download_from_gcs(full_fg_path, local_fg_path)
        else:
            local_fg_path = os.path.join(data_path, fg_name)

        # Open the foreground image.
        try:
            input_fg = np.array(Image.open(local_fg_path))
        except Exception as e:
            print(f"Error opening image {local_fg_path}: {e}")
            continue

        # Determine output file name.
        file_id = os.path.splitext(os.path.basename(fg_name))[0]
        if output_on_gcs:
            output_blob = f"{file_id}.jpg"
            full_output_path = os.path.join(output_data_path, output_blob).replace("\\", "/")
            if gcs_blob_exists(full_output_path):
                print(f"Skipping '{fg_name}': Output blob already exists on GCS.")
                continue
        else:
            output_path = os.path.join(output_data_path, f"{file_id}.jpg")
            if os.path.exists(output_path):
                print(f"Skipping '{fg_name}': Output file '{output_path}' already exists.")
                continue

        # Dynamically adjust dimensions.
        orig_height, orig_width = input_fg.shape[:2]
        image_width, image_height = adjust_dimensions(orig_width, orig_height, max_dim=1024, divisible_by=8)
        print(f"Processing '{fg_name}': Adjusted dimensions: {image_width} x {image_height}")

        # Select a random prompt from illumination prompts.
        prompt = np.random.choice(illuminate_prompts)
        bg_source = np.random.choice([BGSource.NONE, BGSource.NONE, BGSource.NONE, BGSource.NONE,
                                      BGSource.LEFT, BGSource.RIGHT, BGSource.TOP, BGSource.BOTTOM])
        seed = 123456
        steps = 25
        a_prompt = "not obvious objects in the background, best quality, don't significantly change foreground objects, keep its semantic meaning"
        n_prompt = "have obvious objects in the background, lowres, bad anatomy, bad hands, cropped, worst quality, change foreground objects, don't keep its semantic meaning"
        cfg = 2.0
        highres_scale = 1.0
        highres_denoise = 0.5
        lowres_denoise = 0.9
        num_samples = 1

        # Process relighting.
        input_fg, results = process_relight(
            input_fg=input_fg,
            prompt=prompt,
            image_width=image_width,
            image_height=image_height,
            num_samples=num_samples,
            seed=seed,
            steps=steps,
            a_prompt=a_prompt,
            n_prompt=n_prompt,
            cfg=cfg,
            highres_scale=highres_scale,
            highres_denoise=highres_denoise,
            lowres_denoise=lowres_denoise,
            bg_source=bg_source.value  # Pass Enum string value if needed
        )

        # For blending, determine the color mask path.
        if input_on_gcs:
            # Use the helper function to build the color mask path properly.
            color_mask_path = build_color_mask_gcs_path(data_path, file_id)
            local_mask_path = os.path.join(temp_dir, f"{file_id}_mask.png")
            download_from_gcs(color_mask_path, local_mask_path)
        else:
            color_mask_path = os.path.join(data_path.replace("train", "panoptic_train"), f"{file_id}.png")
            local_mask_path = color_mask_path

        try:
            color_mask = Image.open(local_mask_path)
        except Exception as e:
            print(f"Error opening color mask {local_mask_path}: {e}")
            continue

        blended_image = blend_images_with_mask_rank_sigmoid(
            old_image=results[0],
            new_image=input_fg,
            color_mask=color_mask
        )

        # Save the output.
        if output_on_gcs:
            local_output_file = os.path.join(temp_dir, f"{file_id}.jpg")
            blended_image.save(local_output_file)
            upload_to_gcs(local_output_file, full_output_path)
            os.remove(local_output_file)
        else:
            blended_image.save(output_path)
            print(f"Saved relit image to '{output_path}'")

        # Record details.
        records[fg_name] = {
            "output_path": full_output_path if output_on_gcs else output_path,
            "prompt": prompt,
            "bg_source": bg_source.value,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "highres_scale": highres_scale,
            "highres_denoise": highres_denoise,
            "lowres_denoise": lowres_denoise
        }

    # (Optional) Save or update your records file.
    # if record_path:
    #     if record_path.startswith("gs://"):
    #         local_record_path = os.path.join(temp_dir, "record.json")
    #     else:
    #         local_record_path = record_path
    #
    #     with open(local_record_path, 'w') as f:
    #         json.dump(records, f, indent=4)
    #     if record_path.startswith("gs://"):
    #         upload_to_gcs(local_record_path, record_path)
    #         os.remove(local_record_path)
    #     print("Processing complete.")

if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser(description="Relight images using Stable Diffusion pipelines with GCS integration.")
        parser.add_argument('--dataset_path', type=str, required=True,
                            help="Path to the segment dataset. Can be a local path or a GCS path (e.g., gs://bucket/path).")
        parser.add_argument('--output_data_path', type=str, required=True,
                            help="Path to save the output data. Can be a local path or a GCS path.")
        parser.add_argument('--num_splits', type=int, default=1, help="Number of splits to create")
        parser.add_argument('--split', type=int, default=0, help="Split index to process (0-indexed)")
        parser.add_argument('--index_json_path', type=str, default=None,
                            help="Path to the JSON file containing image filenames; supports GCS paths.")
        parser.add_argument('--illuminate_prompts_path', type=str, required=True,
                            help="Path to the JSON file containing illumination prompts; supports GCS paths.")
        parser.add_argument('--record_path', type=str, default=None,
                            help="Path to the JSON file where records are saved; supports GCS paths.")
        return parser.parse_args()

    args = parse_args()
    main(args)