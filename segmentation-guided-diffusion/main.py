import os
import csv
from argparse import ArgumentParser

# torch imports
import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

# HF imports
import diffusers
from diffusers.optimization import get_cosine_schedule_with_warmup
import datasets

# custom imports
from training import TrainingConfig, train_loop
from eval import evaluate_generation, evaluate_sample_many


def _flat_image_to_mask_path(img_path, mask_root, mask_subdir):
    """From image path like .../input_nnunet/Neo_Normal__10_0000.png return mask path
    .../pred_masks_by_class_denoised/Neo_Normal/10.png. Returns None if mask not found."""
    f = os.path.basename(img_path)
    if "__" not in f or not f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        return None
    prefix, rest = f.split("__", 1)
    num = rest.split("_")[0]
    seg_path = os.path.join(mask_root, mask_subdir, prefix, num + ".png")
    return seg_path if os.path.isfile(seg_path) else None


def main(
    mode,
    img_size,
    num_img_channels,
    dataset,
    img_dir,
    seg_dir,
    model_type,
    segmentation_guided,
    segmentation_channel_mode,
    num_segmentation_classes,
    train_batch_size,
    eval_batch_size,
    num_epochs,
    resume_epoch=None,
    use_ablated_segmentations=False,
    eval_shuffle_dataloader=True,

    # ref conditioning (requires use_crop_data + label_dir for same-label ref)
    use_ref_conditioning=False,
    ref_downsize=64,
    use_crop_data=False,
    label_dir=None,
    # flat HVDROPDB layout: img_dir/img_subdir/*.png, mask at mask_subdir/<prefix>/<num>.png
    use_flat_data=False,
    img_subdir="input_nnunet",
    mask_subdir="pred_masks_by_class_denoised",

    # arguments only used in eval
    eval_mask_removal=False,
    eval_blank_mask=False,
    eval_sample_size=1000,
    cuda_device=None,
):
    # GPUs
    if cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on {}'.format(device))

    # load config
    output_dir = '{}-{}-{}'.format(model_type.lower(), dataset, img_size)  # the model namy locally and on the HF Hub
    if segmentation_guided:
        output_dir += "-segguided"
        assert seg_dir is not None, "must provide segmentation directory for segmentation guided training/sampling"

    if use_ablated_segmentations or eval_mask_removal or eval_blank_mask:
        output_dir += "-ablated"
    if use_ref_conditioning:
        output_dir += "-ref"
        assert use_crop_data and label_dir, "use_ref_conditioning requires use_crop_data and label_dir (for same-label ref)"

    print("output dir: {}".format(output_dir))

    if mode == "train":
        evalset_name = "val"
        assert img_dir is not None, "must provide image directory for training"
    elif "eval" in mode:
        evalset_name = "test"

    print("using evaluation set: {}".format(evalset_name))

    config = TrainingConfig(
        image_size = img_size,
        dataset = dataset,
        segmentation_guided = segmentation_guided,
        segmentation_channel_mode = segmentation_channel_mode,
        num_segmentation_classes = num_segmentation_classes,
        train_batch_size = train_batch_size,
        eval_batch_size = eval_batch_size,
        num_epochs = num_epochs,
        output_dir = output_dir,
        model_type=model_type,
        resume_epoch=resume_epoch,
        use_ablated_segmentations=use_ablated_segmentations,
        use_ref_conditioning=use_ref_conditioning,
        ref_downsize=ref_downsize,
    )

    load_images_as_np_arrays = False
    if num_img_channels not in [1, 3]:
        load_images_as_np_arrays = True
        print("image channels not 1 or 3, attempting to load images as np arrays...")

    if config.segmentation_guided:
        # Support layouts:
        # 1) original: seg_dir/<seg_type>/<train|test|val>...
        # 2) CROP flat: <CROP_ROOT>/<train_images|...>/<train_masks|...>
        # 3) flat HVDROPDB: img_dir/img_subdir/*.png, mask at (seg_dir|img_dir)/mask_subdir/<prefix>/<num>.png
        if use_flat_data:
            seg_types = ["mask"]
        else:
            crop_mask_root = None
        if not use_flat_data and use_crop_data and seg_dir is not None:
            candidates = []
            cur = seg_dir
            for _ in range(3):
                if cur and cur not in candidates:
                    candidates.append(cur)
                parent = os.path.dirname(cur) if cur else None
                if not parent or parent == cur:
                    break
                cur = parent
            for cand in candidates:
                ok = True
                for split_sub in ["train_images", "test_images", "val_images"]:
                    mask_sub = split_sub.replace("_images", "_masks")
                    if not os.path.isdir(os.path.join(cand, split_sub, mask_sub)):
                        ok = False
                        break
                if ok:
                    crop_mask_root = cand
                    break

        if not use_flat_data:
            if crop_mask_root is not None:
                seg_types = ["mask"]
                print("Detected CROP flat mask layout at {}".format(crop_mask_root))
            else:
                seg_types = os.listdir(seg_dir)
        seg_paths_train = {} 
        seg_paths_eval = {}
        label_dict = {}  # id_code -> diagnosis (for use_crop_data + use_ref_conditioning)

        mask_root = (seg_dir if seg_dir is not None else img_dir)

        if use_flat_data and img_dir is not None:
            # Flat HVDROPDB: img_dir/img_subdir/*.png, mask at mask_root/mask_subdir/<prefix>/<num>.png
            img_scan = os.path.join(img_dir, img_subdir)
            assert os.path.isdir(img_scan), "flat data image dir not found: {}".format(img_scan)
            img_paths_all = []
            seg_paths_all = {"mask": []}
            for f in sorted(os.listdir(img_scan)):
                if not f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    continue
                img_path = os.path.join(img_scan, f)
                seg_path = _flat_image_to_mask_path(img_path, mask_root, mask_subdir)
                if seg_path is not None:
                    img_paths_all.append(img_path)
                    seg_paths_all["mask"].append(seg_path)
            np.random.seed(0)
            n = len(img_paths_all)
            perm = np.random.permutation(n)
            n_train = int(0.9 * n) or max(1, n - 1)
            train_idx, eval_idx = perm[:n_train], perm[n_train:]
            img_paths_train = [img_paths_all[i] for i in train_idx]
            img_paths_eval = [img_paths_all[i] for i in eval_idx]
            seg_paths_train["mask"] = [seg_paths_all["mask"][i] for i in train_idx]
            seg_paths_eval["mask"] = [seg_paths_all["mask"][i] for i in eval_idx]
            print("Flat HVDROPDB: {} train, {} eval (total {} with valid mask)".format(
                len(img_paths_train), len(img_paths_eval), n))
        elif use_crop_data and img_dir is not None:
            # CROP: merge train_images, test_images, val_images; optional labels from CSVs
            if label_dir is not None:
                for csv_name in ["train_1.csv", "valid.csv", "test.csv"]:
                    csv_path = os.path.join(label_dir, csv_name)
                    if os.path.isfile(csv_path):
                        with open(csv_path, "r") as f:
                            r = csv.DictReader(f)
                            for row in r:
                                label_dict[row["id_code"].strip()] = int(row["diagnosis"].strip())
                print("Loaded {} labels from {}".format(len(label_dict), label_dir))

            img_paths_all = []
            seg_paths_all = {seg_type: [] for seg_type in seg_types}
            for sub in ["train_images", "test_images", "val_images"]:
                img_sub = os.path.join(img_dir, sub)
                if not os.path.isdir(img_sub):
                    continue
                # Support nested: CROP/train_images/train_images/*.png
                img_sub_inner = os.path.join(img_sub, sub)
                if os.path.isdir(img_sub_inner):
                    scan_dir = img_sub_inner
                    seg_extra = sub  # seg under seg_dir/seg_type/sub/sub/f
                else:
                    scan_dir = img_sub
                    seg_extra = ""
                for f in sorted(os.listdir(scan_dir)):
                    if not f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        continue
                    img_path = os.path.join(scan_dir, f)
                    img_paths_all.append(img_path)
                    for seg_type in seg_types:
                        if crop_mask_root is not None:
                            mask_sub = sub.replace("_images", "_masks")
                            seg_sub = os.path.join(crop_mask_root, sub, mask_sub)
                            seg_path = os.path.join(seg_sub, f)
                        elif seg_dir == img_dir:
                            seg_path = img_path
                        else:
                            seg_sub = os.path.join(seg_dir, seg_type, sub)
                            if seg_extra and os.path.isdir(os.path.join(seg_dir, seg_type, sub, seg_extra)):
                                seg_sub = os.path.join(seg_dir, seg_type, sub, seg_extra)
                            seg_path = os.path.join(seg_sub, f)
                        seg_paths_all[seg_type].append(seg_path)

            # 90% train, 10% eval (fixed seed)
            np.random.seed(0)
            n = len(img_paths_all)
            perm = np.random.permutation(n)
            n_train = int(0.9 * n) or max(1, n - 1)
            train_idx, eval_idx = perm[:n_train], perm[n_train:]
            img_paths_train = [img_paths_all[i] for i in train_idx]
            img_paths_eval = [img_paths_all[i] for i in eval_idx]
            for seg_type in seg_types:
                seg_paths_train[seg_type] = [seg_paths_all[seg_type][i] for i in train_idx]
                seg_paths_eval[seg_type] = [seg_paths_all[seg_type][i] for i in eval_idx]
            print("CROP: {} train, {} eval".format(len(img_paths_train), len(img_paths_eval)))
        else:
            # original: train/ and val/ (or evalset_name) subdirs
            # train set
            if img_dir is not None: 
                img_dir_train = os.path.join(img_dir, "train")
                img_paths_train = [os.path.join(img_dir_train, f) for f in os.listdir(img_dir_train)]
                for seg_type in seg_types:
                    seg_paths_train[seg_type] = [os.path.join(seg_dir, seg_type, "train", f) for f in os.listdir(img_dir_train)]
            else:
                for seg_type in seg_types:
                    seg_paths_train[seg_type] = [os.path.join(seg_dir, seg_type, "train", f) for f in os.listdir(os.path.join(seg_dir, seg_type, "train"))]

            # eval set
            if img_dir is not None: 
                img_dir_eval = os.path.join(img_dir, evalset_name)
                img_paths_eval = [os.path.join(img_dir_eval, f) for f in os.listdir(img_dir_eval)]
                for seg_type in seg_types:
                    seg_paths_eval[seg_type] = [os.path.join(seg_dir, seg_type, evalset_name, f) for f in os.listdir(img_dir_eval)]
            else:
                for seg_type in seg_types:
                    seg_paths_eval[seg_type] = [os.path.join(seg_dir, seg_type, evalset_name, f) for f in os.listdir(os.path.join(seg_dir, seg_type, evalset_name))]

        if img_dir is not None:
            dset_dict_train = {
                    **{"image": img_paths_train},
                    **{"seg_{}".format(seg_type): seg_paths_train[seg_type] for seg_type in seg_types}
                }
            dset_dict_eval = {
                    **{"image": img_paths_eval},
                    **{"seg_{}".format(seg_type): seg_paths_eval[seg_type] for seg_type in seg_types}
            }
        else:
            dset_dict_train = {
                    **{"seg_{}".format(seg_type): seg_paths_train[seg_type] for seg_type in seg_types}
                }
            dset_dict_eval = {
                    **{"seg_{}".format(seg_type): seg_paths_eval[seg_type] for seg_type in seg_types}
            }

        if img_dir is not None:
            dset_dict_train["image_filename"] = [os.path.basename(f) for f in dset_dict_train["image"]]
            dset_dict_eval["image_filename"] = [os.path.basename(f) for f in dset_dict_eval["image"]]
        else:
            dset_dict_train["image_filename"] = [os.path.basename(f) for f in dset_dict_train["seg_{}".format(seg_types[0])]]
            dset_dict_eval["image_filename"] = [os.path.basename(f) for f in dset_dict_eval["seg_{}".format(seg_types[0])]]

        if use_crop_data and label_dict:
            dset_dict_train["label"] = [label_dict.get(os.path.splitext(os.path.basename(p))[0], 0) for p in dset_dict_train["image"]]
            dset_dict_eval["label"] = [label_dict.get(os.path.splitext(os.path.basename(p))[0], 0) for p in dset_dict_eval["image"]]

        dataset_train = datasets.Dataset.from_dict(dset_dict_train)
        dataset_eval = datasets.Dataset.from_dict(dset_dict_eval)

        # load the images
        if not load_images_as_np_arrays and img_dir is not None:
            dataset_train = dataset_train.cast_column("image", datasets.Image())
            dataset_eval = dataset_eval.cast_column("image", datasets.Image())

        for seg_type in seg_types:
            dataset_train = dataset_train.cast_column("seg_{}".format(seg_type), datasets.Image())

        for seg_type in seg_types:
            dataset_eval = dataset_eval.cast_column("seg_{}".format(seg_type), datasets.Image())

    else:
        if img_dir is not None:
            img_dir_train = os.path.join(img_dir, "train")
            img_paths_train = [os.path.join(img_dir_train, f) for f in os.listdir(img_dir_train)]

            img_dir_eval = os.path.join(img_dir, evalset_name)
            img_paths_eval = [os.path.join(img_dir_eval, f) for f in os.listdir(img_dir_eval)]

            dset_dict_train = {
                    **{"image": img_paths_train}
                }

            dset_dict_eval = {
                    **{"image": img_paths_eval}
                }

            # add image filenames to dataset
            dset_dict_train["image_filename"] = [os.path.basename(f) for f in dset_dict_train["image"]]
            dset_dict_eval["image_filename"] = [os.path.basename(f) for f in dset_dict_eval["image"]]

            dataset_train = datasets.Dataset.from_dict(dset_dict_train)
            dataset_eval = datasets.Dataset.from_dict(dset_dict_eval)

            # load the images
            if not load_images_as_np_arrays:
                dataset_train = dataset_train.cast_column("image", datasets.Image())
                dataset_eval = dataset_eval.cast_column("image", datasets.Image())

    # training set preprocessing
    if not load_images_as_np_arrays:
        preprocess = transforms.Compose(
            [
                transforms.Resize((config.image_size, config.image_size)),
                # transforms.RandomHorizontalFlip(), # flipping wouldn't result in realistic images
                transforms.ToTensor(),
                transforms.Normalize(
                    num_img_channels * [0.5], 
                    num_img_channels * [0.5]),
            ]
        )
    else:
        # resizing will be done in the transform function
        preprocess = transforms.Compose(
            [
                transforms.Normalize(
                    num_img_channels * [0.5], 
                    num_img_channels * [0.5]),
            ]
        )

    if num_img_channels == 1:
        PIL_image_type = "L"
    elif num_img_channels == 3:
        PIL_image_type = "RGB"
    else:
        PIL_image_type = None

    if config.segmentation_guided:
        preprocess_segmentation = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
        )

        def transform(examples):
            if img_dir is not None:
                if not load_images_as_np_arrays:
                    images = [preprocess(image.convert(PIL_image_type)) for image in examples["image"]]
                else:
                    # load np array as torch tensor, resize, then normalize
                    images = [
                        preprocess(F.interpolate(torch.tensor(np.load(image)).unsqueeze(0).float(), size=(config.image_size, config.image_size)).squeeze()) for image in examples["image"]
                        ]

            images_filenames = examples["image_filename"]

            segs = {}
            for seg_type in seg_types:
                segs["seg_{}".format(seg_type)] = [preprocess_segmentation(image.convert("L")) for image in examples["seg_{}".format(seg_type)]]
            out = {**{"images": images}, **segs, **{"image_filenames": images_filenames}}
            if "label" in examples:
                out["labels"] = examples["label"]
            if img_dir is not None:
                return out
            else:
                return {**segs, **{"image_filenames": images_filenames}}
            
        dataset_train.set_transform(transform)
        dataset_eval.set_transform(transform)

    else:
        if img_dir is not None:
            def transform(examples):
                if not load_images_as_np_arrays:
                    images = [preprocess(image.convert(PIL_image_type)) for image in examples["image"]]
                else:
                    images = [
                        preprocess(F.interpolate(torch.tensor(np.load(image)).unsqueeze(0).float(), size=(config.image_size, config.image_size)).squeeze()) for image in examples["image"]
                        ]
                images_filenames = examples["image_filename"]
                #return {"images": images, "image_filenames": images_filenames}
                return {"images": images, **{"image_filenames": images_filenames}}
        
            dataset_train.set_transform(transform)
            dataset_eval.set_transform(transform)

    if ((img_dir is None) and (not segmentation_guided)):
        train_dataloader = None
        # just make placeholder dataloaders to iterate through when sampling from uncond model
        eval_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.zeros(config.eval_batch_size, num_img_channels, config.image_size, config.image_size)),
            batch_size=config.eval_batch_size,
            shuffle=eval_shuffle_dataloader
        )
    else:
        train_dataloader = torch.utils.data.DataLoader(
                dataset_train, 
                batch_size=config.train_batch_size, 
                shuffle=True
                )

        eval_dataloader = torch.utils.data.DataLoader(
                dataset_eval, 
                batch_size=config.eval_batch_size, 
                shuffle=eval_shuffle_dataloader
                )

    # define the model
    in_channels = num_img_channels
    if config.segmentation_guided:
        assert config.num_segmentation_classes is not None
        assert config.num_segmentation_classes > 1, "must have at least 2 segmentation classes (INCLUDING background)" 
        if config.segmentation_channel_mode == "single":
            in_channels += 1
        elif config.segmentation_channel_mode == "multi":
            in_channels = len(seg_types) + in_channels
    if getattr(config, "use_ref_conditioning", False):
        in_channels += 3

    model = diffusers.UNet2DModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=in_channels,  # the number of input channels, 3 for RGB images
        out_channels=num_img_channels,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D"
        ),
    )

    if (mode == "train" and resume_epoch is not None) or "eval" in mode:
        if mode == "train":
            print("resuming from model at training epoch {}".format(resume_epoch))
        elif "eval" in mode:
            print("loading saved model...")
        model_dir = os.path.join(config.output_dir, "unet")
        use_safetensors = os.path.isfile(os.path.join(model_dir, "diffusion_pytorch_model.safetensors"))
        if use_safetensors:
            print("loading safetensors checkpoint...")
        else:
            print("loading .bin checkpoint...")
        model = model.from_pretrained(model_dir, use_safetensors=use_safetensors)

    model = nn.DataParallel(model)
    model.to(device)

    # define noise scheduler
    if model_type == "DDPM":
        noise_scheduler = diffusers.DDPMScheduler(num_train_timesteps=1000)
    elif model_type == "DDIM":
        noise_scheduler = diffusers.DDIMScheduler(num_train_timesteps=1000)

    if mode == "train":
        # training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * config.num_epochs),
        )

        # train
        train_loop(
            config, 
            model, 
            noise_scheduler, 
            optimizer, 
            train_dataloader, 
            eval_dataloader, 
            lr_scheduler, 
            device=device
            )
    elif mode == "eval":
        """
        default eval behavior:
        evaluate image generation or translation (if for conditional model, either evaluate naive class conditioning but not CFG,
        or with CFG),
        possibly conditioned on masks.

        has various options.
        """
        evaluate_generation(
            config, 
            model, 
            noise_scheduler,
            eval_dataloader, 
            eval_mask_removal=eval_mask_removal,
            eval_blank_mask=eval_blank_mask,
            device=device
            )

    elif mode == "eval_many":
        """
        generate many images and save them to a directory, saved individually
        """
        evaluate_sample_many(
            eval_sample_size,
            config,
            model,
            noise_scheduler,
            eval_dataloader,
            device=device
            )

    else:
        raise ValueError("mode \"{}\" not supported.".format(mode))


if __name__ == "__main__":
    # parse args:
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--num_img_channels', type=int, default=1)
    parser.add_argument('--dataset', type=str, default="breast_mri")
    parser.add_argument('--img_dir', type=str, default=None)
    parser.add_argument('--seg_dir', type=str, default=None)
    parser.add_argument('--model_type', type=str, default="DDPM")
    parser.add_argument('--segmentation_guided', action='store_true', help='use segmentation guided training/sampling?')
    parser.add_argument('--segmentation_channel_mode', type=str, default="single", help='single == all segmentations in one channel, multi == each segmentation in its own channel')
    parser.add_argument('--num_segmentation_classes', type=int, default=None, help='number of segmentation classes, including background')
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--resume_epoch', type=int, default=None, help='resume training starting at this epoch')

    # novel options
    parser.add_argument('--use_ablated_segmentations', action='store_true', help='use mask ablated training and any evaluation? sometimes randomly remove class(es) from mask during training and sampling.')

    # other options
    parser.add_argument('--eval_noshuffle_dataloader', action='store_true', help='if true, don\'t shuffle the eval dataloader')

    # args only used in eval
    parser.add_argument('--eval_mask_removal', action='store_true', help='if true, evaluate gradually removing anatomies from mask and re-sampling')
    parser.add_argument('--eval_blank_mask', action='store_true', help='if true, evaluate sampling conditioned on blank (zeros) masks')
    parser.add_argument('--eval_sample_size', type=int, default=1000, help='number of images to sample when using eval_many mode')

    # ref conditioning (CROP + aptos2019 labels)
    parser.add_argument('--use_ref_conditioning', action='store_true', help='add 3-channel reference (same-label, downscale then upsample)')
    parser.add_argument('--ref_downsize', type=int, default=64, help='ref downscale size before upsample (default 64)')
    parser.add_argument('--use_crop_data', action='store_true', help='use CROP layout: train_images, test_images, val_images + merge for train')
    parser.add_argument('--label_dir', type=str, default=None, help='path to aptos2019 CSVs (id_code, diagnosis) for ref same-label')
    parser.add_argument('--use_flat_data', action='store_true', help='use flat HVDROPDB layout: img_dir/img_subdir + mask_subdir/<prefix>/<num>.png')
    parser.add_argument('--img_subdir', type=str, default='input_nnunet', help='image subdir under img_dir when use_flat_data')
    parser.add_argument('--mask_subdir', type=str, default='pred_masks_by_class_denoised', help='mask root subdir when use_flat_data (masks at mask_root/<mask_subdir>/<prefix>/<num>.png)')
    parser.add_argument('--cuda_device', type=int, default=None, help='CUDA device index (e.g. 2 for cuda:2)')

    args = parser.parse_args()

    main(
        args.mode,
        args.img_size,
        args.num_img_channels,
        args.dataset,
        args.img_dir,
        args.seg_dir,
        args.model_type,
        args.segmentation_guided,
        args.segmentation_channel_mode,
        args.num_segmentation_classes,
        args.train_batch_size,
        args.eval_batch_size,
        args.num_epochs,
        args.resume_epoch,
        args.use_ablated_segmentations,
        not args.eval_noshuffle_dataloader,

        use_ref_conditioning=args.use_ref_conditioning,
        ref_downsize=args.ref_downsize,
        use_crop_data=args.use_crop_data,
        label_dir=args.label_dir,
        use_flat_data=args.use_flat_data,
        img_subdir=args.img_subdir,
        mask_subdir=args.mask_subdir,

        eval_mask_removal=args.eval_mask_removal,
        eval_blank_mask=args.eval_blank_mask,
        eval_sample_size=args.eval_sample_size,
        cuda_device=args.cuda_device,
    )
