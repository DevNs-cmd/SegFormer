import torch
import torch.nn as nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation


class SegFormerModel(nn.Module):
    """Wrapper around HuggingFace SegFormer with configurable class count."""

    def __init__(self, backbone: str, num_classes: int, ignore_index: int = 255, use_pretrained: bool = True):
        super().__init__()
        if "segformer-b2" not in backbone.lower() and "segformer-b3" not in backbone.lower():
            raise ValueError("Backbone must be a SegFormer-B2 or SegFormer-B3 HuggingFace model id.")

        self.ignore_index = ignore_index

        if use_pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                backbone,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            config = SegformerConfig.from_pretrained(backbone)
            config.num_labels = num_classes
            self.model = SegformerForSemanticSegmentation(config)

        # Ensure label maps are consistent with arbitrary class counts.
        self.model.config.id2label = {i: f"class_{i}" for i in range(num_classes)}
        self.model.config.label2id = {f"class_{i}": i for i in range(num_classes)}

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).logits

    def freeze_backbone(self) -> None:
        for name, param in self.model.named_parameters():
            if "decode_head" not in name:
                param.requires_grad = False

    def unfreeze_all(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True
