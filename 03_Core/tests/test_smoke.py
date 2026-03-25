import unittest

import torch

from PCBVision.models.factory import get_model
from PCBVision.utils.loss_functions import FocalLoss, DiceLoss, HybridLoss


class SmokeTests(unittest.TestCase):
    def test_models_forward_rgb(self):
        x = torch.randn(2, 3, 256, 256)
        for name in ["unet", "attention_unet", "resunet", "deeplabv3+", "linknet"]:
            m = get_model(name, in_channels=3, out_channels=4, pretrained=False)
            m.eval()
            y = m(x)
            self.assertEqual(tuple(y.shape), (2, 4, 256, 256))

    def test_models_forward_hsi(self):
        x = torch.randn(1, 214, 128, 128)
        for name in ["unet", "resunet", "deeplabv3+"]:
            m = get_model(name, in_channels=214, out_channels=4, pretrained=False)
            m.eval()
            y = m(x)
            self.assertEqual(tuple(y.shape), (1, 4, 128, 128))

    def test_losses_forward(self):
        logits = torch.randn(2, 4, 64, 64)
        targets = torch.randint(0, 4, (2, 64, 64), dtype=torch.long)

        focal = FocalLoss(alpha=torch.tensor([1.0, 1.0, 1.0, 1.0]))
        dice = DiceLoss()
        hybrid = HybridLoss(alpha=torch.tensor([1.0, 1.0, 1.0, 1.0]))

        self.assertTrue(torch.isfinite(focal(logits, targets)))
        self.assertTrue(torch.isfinite(dice(logits, targets)))
        self.assertTrue(torch.isfinite(hybrid(logits, targets)))


if __name__ == "__main__":
    unittest.main()
