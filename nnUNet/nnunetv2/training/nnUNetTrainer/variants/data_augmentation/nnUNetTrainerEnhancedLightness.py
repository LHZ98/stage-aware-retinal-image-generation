"""
Enhanced nnUNetTrainer with stronger lightness/brightness augmentation for domain shift.
For retinal fundus and other imaging where strong lighting augmentation is needed.
"""

from typing import Union, Tuple, List
import numpy as np
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerEnhancedLightness(nnUNetTrainer):
    """
    增强版Trainer，专门针对domain shift问题增强lightness扰动
    
    主要增强：
    1. 更强的brightness扰动范围（0.5-1.5，默认0.75-1.25）
    2. 更高的brightness应用概率（0.3，默认0.15）
    3. 更强的contrast扰动范围（0.5-1.5，默认0.75-1.25）
    4. 更高的gamma扰动概率和应用范围
    5. 添加额外的brightness变换以增加多样性
    """
    
    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        """
        重写训练时的数据增强，增强lightness相关变换
        基于父类方法，但增强brightness、contrast和gamma变换
        """
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False, mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        
        # 增强的brightness扰动（范围更大，概率更高）
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.5, 1.5)),  # 默认0.75-1.25，现在扩大到0.5-1.5
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.3  # 默认0.15，现在提高到0.3
        ))
        
        # 增强的contrast扰动
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.5, 1.5)),  # 默认0.75-1.25，现在扩大到0.5-1.5
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.25  # 默认0.15，现在提高到0.25
        ))
        
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        
        # 增强的gamma扰动（两个方向都增强）
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.5, 2.0)),  # 默认0.7-1.5，现在扩大到0.5-2.0
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.2  # 默认0.1，现在提高到0.2
        ))
        
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.5, 2.0)),  # 默认0.7-1.5，现在扩大到0.5-2.0
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.4  # 默认0.3，现在提高到0.4
        ))
        
        # 添加额外的轻微brightness扰动以增加多样性
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.9, 1.1)),  # 轻微扰动
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.5  # 高概率应用轻微扰动
        ))
        
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        p_per_sample=0.4,
                        key="data",
                        strel_size=(1, 5),
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        key="data",
                        p_per_sample=0.2,
                        fill_with_other_class_p=0.0,
                        dont_do_if_covers_more_than_X_percent=0.15
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            transforms.append(ConvertSegmentationToRegionsTransform(regions=regions))

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(deep_supervision_scales, 0, 0, input_key="target",
                                                          output_key="target"))
        transforms.append(ComposeTransforms(transforms))
        return transforms[-1]

