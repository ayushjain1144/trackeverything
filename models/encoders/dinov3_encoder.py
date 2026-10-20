# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import einops
import ipdb
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import nn

from .dinov3_adapter import DINOv3_Adapter
from .dinov3_model import (
    DINOV3_MODEL_FUNC,
    DINOV3_OUTPUT_DIM,
    Weights,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_TO_NUM_LAYERS,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)


st = ipdb.set_trace


BACKBONE_INTERMEDIATE_LAYERS = {
    "dinov3_vits16": [2, 5, 8, 11],
    "dinov3_vitb16": [2, 5, 8, 11],
    "dinov3_vitl16": [4, 11, 17, 23],
    "dinov3_vit7b16": [9, 19, 29, 39],
}


class Dinov3Encoder(nn.Module):
    """
    Tokenize an image snippet using DinoV3.
    """

    def __init__(
        self,
        dinov3_name: str = "dinov3_vitb16",
        pretrained: bool = True,
        freeze: bool = False,
        output_dim: int = 768,  # ignored if add_linear_layer = False
        out_patch_size: int = 16,
        return_features_directly: bool = False,
        use_vit_adapter: bool = True,
    ):
        super().__init__()
        self.freeze = freeze
        self.return_features_directly = return_features_directly
        self.n_layers = MODEL_TO_NUM_LAYERS[dinov3_name]
        self.output_dim = output_dim
        self.use_vit_adapter = use_vit_adapter
        assert dinov3_name in DINOV3_MODEL_FUNC.keys(), f"{dinov3_name} not supported"

        self.register_buffer(
            "mean",
            torch.as_tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).float(),
        )
        self.register_buffer(
            "std", torch.as_tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).float()
        )

        self.embedding_dim = (
            output_dim  # this is expected variable in init for downstream modules
        )

        weights_path = os.environ.get("DINOV3_WEIGHTS_PATH", "")
        if weights_path and os.path.exists(weights_path):
            weights = weights_path
        else:
            weights = Weights.LVD1689M
        self.model = DINOV3_MODEL_FUNC[dinov3_name](
            pretrained=pretrained, weights=weights
        )
        # else:
        #     from transformers import AutoModel
        #     if dinov3_name == "dinov3_vits16":
        #         pretrained_model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
        #     elif dinov3_name == "dinov3_vits16plus":
        #         pretrained_model_name = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
        #     elif dinov3_name == "dinov3_vitb16":
        #         pretrained_model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m" 
        #     else:
        #         raise ValueError(f"Unknown arch_type: {dinov3_name}")
        #     self.model = AutoModel.from_pretrained(pretrained_model_name, device_map=None)
        #     embed_dim = self.model.config.hidden_size
        #     self.model.embed_dim = embed_dim
        #     self.model.patch_size = 16
            
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad_(False)
            self.model.eval()

        # dinov3 adapter
        if use_vit_adapter:
            self.model = DINOv3_Adapter(
                self.model,
                interaction_indexes=BACKBONE_INTERMEDIATE_LAYERS[dinov3_name],
                freeze_backbone=freeze,
            )

        self.dim_out = DINOV3_OUTPUT_DIM[dinov3_name]

        if not self.return_features_directly:
            self.projector_layer = nn.Linear(self.dim_out, output_dim)
        else:
            # set the embedding dim to the output dim of the encoder
            self.embedding_dim = self.dim_out

        logger.info(
            f"DinoV3 InputTokenizer {dinov3_name}, is frozen {freeze}, dim_out of {self.dim_out}"
        )
        self.out_patch_size = out_patch_size

    def feat_dim(self):
        return self.embedding_dim

    def patch_size(self):
        return self.out_patch_size

    def post_process(self, feats, B, T, out_size=None, rot_k=0):
        """
        Post processing to convert Dino features, e.g. feature interpolation to the desired size,
        handling Aria image rotation, the linear mapping to increase feature dimension.

        Args:
            feats: [B x T x C x H x W]
            B: batch size
            T: number of frames
            out_size: (h, w) token feature map output size, if None, don't resize the feature map size.
        """
        if out_size is not None:
            # resize to desired size
            feats = F.interpolate(feats, out_size, mode="bilinear")
        if rot_k != 0:
            feats = torch.rot90(feats, rot_k, [-2, -1])
        # to token sequence BxNxC
        feats = einops.rearrange(feats, "(b t) c h w -> b t h w c", b=B, t=T)
        return feats

    def forward(
        self, img: torch.Tensor, normalize_img: bool = True, rot_k: int = 0
    ) -> torch.Tensor:
        """
        Args:
            img: [B x T x C x H x W] A sequence / snippet of Image Frames
        """
        assert img.dim() == 5, f"expecting BxTxCxHxW but got {img.shape}"
        B, T, C, H, W = img.shape
        assert C in [1, 3], "must be either 1 or 3 channel input (BxCxHxW)"
        if C == 1:
            # Fake RGB by repeating gray channel.
            img = img.repeat(1, 1, 3, 1, 1)
        assert H % 16 == 0 and W % 16 == 0, "expecting H and W divisible by 16"

        if rot_k != 0:
            img = torch.rot90(img, rot_k, [-1, -2])
        # get batch image for resnet
        img = einops.rearrange(img, "b t c h w -> (b t) c h w")

        # assert (
        #     img.min() >= -1e-3 and img.max() <= 1.0 + 1e-3
        # ), "input image not in [0,1]"
        img = (img - self.mean) / self.std
        
        if not self.use_vit_adapter:
            with torch.inference_mode():
                feats = self.model.get_intermediate_layers(
                    img,
                    n=range(self.n_layers),
                    reshape=True,
                    norm=True,
                )
                feats = feats[-1]  # (b t), c, hp, wp
        else:
            feats = self.model(img)

            # keep the highest resolution feature
            feats = feats["1"]
        feats = self.post_process(feats, B, T, out_size=None, rot_k=rot_k)

        if not self.return_features_directly:
            feats = self.projector_layer(feats)

        feats = einops.rearrange(feats, "b t h w c -> b t c h w")
        return feats
