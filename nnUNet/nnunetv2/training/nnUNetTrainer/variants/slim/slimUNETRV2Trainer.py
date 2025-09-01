from typing import List, Tuple, Union
import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nets.SlimUNETRV2 import SlimUNETRV2


class SlimUNETRV2_128Trainer(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        model = SlimUNETRV2(
            in_chans=num_input_channels,
            out_chans=num_output_channels,
            kernel_sizes=[4, 2, 2, 2],
            num_slices_list=[64, 32, 16, 8],
        )
        self.configuration_manager.patch_size = (128, 128, 128)
        return model


class SlimUNETRV2_96Trainer(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        model = SlimUNETRV2(
            in_chans=num_input_channels,
            out_chans=num_output_channels,
            kernel_sizes=[2, 2, 2, 2],
            num_slices_list=[16, 8, 4, 2],
        )
        self.configuration_manager.patch_size = (96, 96, 96)
        return model
