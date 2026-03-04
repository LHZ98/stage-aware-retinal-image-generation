#!/usr/bin/env python3
"""
Generate images from a directory of segmentation masks using a trained
segmentation-guided diffusion checkpoint. Does not require the main.py
dataset layout (seg_dir/all/test/); reads mask images directly.

Usage:
  python run_generate_from_masks.py \
    --ckpt_dir ddpm-aptos2019-valtestft-256-segguided \
    --mask_dir debug/crop_test_masks16_mask01_single \
    --mask_glob "*_mask01_256.png" \
    --batch_size 8 \
    --out_suffix _generated_256
"""
import os
import sys
import argparse
import glob
from PIL import Image
import torch
from torchvision import transforms

import diffusers
from training import TrainingConfig
from eval import SegGuidedDDPMPipeline, SegGuidedDDIMPipeline


def parse_args():
    p = argparse.ArgumentParser(description="Generate images from mask directory using a seg-guided diffusion ckpt")
    p.add_argument("--ckpt_dir", type=str, required=True,
                   help="Path to checkpoint dir (e.g. ddpm-aptos2019-valtestft-256-segguided), must contain unet/config.json and unet/diffusion_pytorch_model.safetensors")
    p.add_argument("--mask_dir", type=str, required=True,
                   help="Directory containing mask images")
    p.add_argument("--mask_glob", type=str, default="*_mask01_256.png",
                   help="Glob for mask files under mask_dir (default: *_mask01_256.png)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--model_type", type=str, default="DDPM", choices=["DDPM", "DDIM"])
    p.add_argument("--num_segmentation_classes", type=int, default=2,
                   help="Number of segmentation classes including background (default: 2 for binary)")
    p.add_argument("--out_suffix", type=str, default="_generated_256",
                   help="Suffix for output filenames: mask name 'XX_name_mask01_256.png' -> 'XX_name{out_suffix}.png'")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Output directory; default is mask_dir")
    p.add_argument("--num_inference_steps", type=int, default=None,
                   help="DDIM steps (default: 50); DDPM uses 1000")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ref_dir", type=str, default=None,
                   help="Optional: directory of reference images (same base names as masks) for ref-conditioned model")
    return p.parse_args()


def load_mask_as_tensor(path, size=256):
    """Load a mask image and return tensor 1xHxW in [0, 1/255, 2/255, ...] for class indices."""
    pil = Image.open(path).convert("L")
    tr = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    x = tr(pil)
    # Dataset expects class indices 0,1,2,... as pixel values; ToTensor() gives 0, 1/255, 2/255 for 0,1,2
    # If mask is 0/255 binary, we get 0 and 1; we need foreground as 1/255 so (255*seg).int() = 1
    if x.max() > 0.5:
        x = (x > 0.5).float() / 255.0 + (x <= 0.5).float() * 0.0
    return x


def main():
    args = parse_args()
    mask_dir = os.path.abspath(args.mask_dir)
    ckpt_dir = os.path.abspath(args.ckpt_dir)
    out_dir = os.path.abspath(args.out_dir or mask_dir)
    unet_path = os.path.join(ckpt_dir, "unet")
    if not os.path.isdir(unet_path):
        print("Checkpoint unet dir not found:", unet_path)
        sys.exit(1)
    mask_pattern = os.path.join(mask_dir, args.mask_glob)
    mask_paths = sorted(glob.glob(mask_pattern))
    if not mask_paths:
        print("No mask files found:", mask_pattern)
        sys.exit(1)
    print("Found", len(mask_paths), "masks under", mask_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = TrainingConfig(
        image_size=256,
        dataset="from_masks",
        segmentation_guided=True,
        segmentation_channel_mode="single",
        num_segmentation_classes=args.num_segmentation_classes,
        eval_batch_size=args.batch_size,
        output_dir=ckpt_dir,
        model_type=args.model_type,
    )

    # Load UNet: support both .safetensors and .bin (same as main.py)
    use_safetensors = os.path.isfile(os.path.join(unet_path, "diffusion_pytorch_model.safetensors"))
    if use_safetensors:
        print("Loading UNet from diffusion_pytorch_model.safetensors ...")
    else:
        print("Loading UNet from diffusion_pytorch_model.bin ...")
    unet = diffusers.UNet2DModel.from_pretrained(unet_path, use_safetensors=use_safetensors)
    unet = unet.to(device)
    # Ref-conditioned model (in_channels=7 = 3 noise + 1 seg + 3 ref)
    if getattr(unet.config, "in_channels", 4) == 7:
        config.use_ref_conditioning = True
        config.ref_downsize = 64
        print("Checkpoint uses ref conditioning; will use zero ref if --ref_dir not set.")
    if args.model_type == "DDPM":
        scheduler = diffusers.DDPMScheduler(num_train_timesteps=1000)
        pipeline = SegGuidedDDPMPipeline(unet=unet, scheduler=scheduler, eval_dataloader=None, external_config=config)
    else:
        scheduler = diffusers.DDIMScheduler(num_train_timesteps=1000)
        pipeline = SegGuidedDDIMPipeline(unet=unet, scheduler=scheduler, eval_dataloader=None, external_config=config)
    # Pipeline keeps eval_dataloader/external_config as plain attrs; skip .to(device) (unet already on device)

    num_steps = args.num_inference_steps
    if num_steps is None:
        num_steps = 50 if args.model_type == "DDIM" else 1000

    os.makedirs(out_dir, exist_ok=True)
    n = len(mask_paths)
    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        paths_batch = mask_paths[start:end]
        seg_tensors = [load_mask_as_tensor(p, config.image_size) for p in paths_batch]
        seg_batch = {
            "seg_all": torch.stack(seg_tensors).to(device),
            "image_filenames": [os.path.basename(p) for p in paths_batch],
        }
        if getattr(config, "use_ref_conditioning", False):
            ref_list = []
            for p in paths_batch:
                base = os.path.basename(p)
                name_no_ext = base[:-4] if base.lower().endswith(".png") else base
                if args.ref_dir:
                    ref_path = os.path.join(os.path.abspath(args.ref_dir), name_no_ext + ".png")
                    if not os.path.isfile(ref_path):
                        ref_path = os.path.join(os.path.abspath(args.ref_dir), base)
                    if os.path.isfile(ref_path):
                        ref_pil = Image.open(ref_path).convert("RGB")
                        tr = transforms.Compose([
                            transforms.Resize((config.image_size, config.image_size)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5] * 3, [0.5] * 3),
                        ])
                        ref_list.append(tr(ref_pil))
                    else:
                        ref_list.append(torch.zeros(3, config.image_size, config.image_size))
                else:
                    ref_list.append(torch.zeros(3, config.image_size, config.image_size))
            seg_batch["ref_images"] = torch.stack(ref_list).to(device)
        batch_size_cur = len(paths_batch)
        out_images = pipeline(
            batch_size=batch_size_cur,
            seg_batch=seg_batch,
            num_inference_steps=num_steps,
        ).images
        for i, (path, pil_img) in enumerate(zip(paths_batch, out_images)):
            base = os.path.basename(path)
            # e.g. 00_e4dcca36ceb4_mask01_256.png -> 00_e4dcca36ceb4_generated_256.png
            name_no_ext = base[: -len(".png")] if base.lower().endswith(".png") else base
            if "_mask01_256" in name_no_ext:
                out_name = name_no_ext.replace("_mask01_256", args.out_suffix) + ".png"
            else:
                out_name = name_no_ext + args.out_suffix + ".png"
            out_path = os.path.join(out_dir, out_name)
            pil_img.save(out_path)
        print("Saved batch {}-{} -> {}".format(start, end, out_dir))
    print("Done. Generated", n, "images in", out_dir)


if __name__ == "__main__":
    main()
