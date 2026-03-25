
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean', ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: (N, C, H, W) logits
        # targets: (N, C, H, W) one-hot float or (N, H, W) indices
        
        # If targets are (N, H, W), convert to one-hot for consistency if needed, 
        # BUT standard CE supports indices.
        # The error was "ignore_index not supported for floating point target".
        # This implies targets passed were Float (one-hot).
        
        if targets.dim() == 4: # One-hot (N, C, H, W)
            # Use binary_cross_entropy_with_logits for soft labels / one-hot
            # But Focal Law assumes class dimension.
            # Manually compute CE: -sum(target * log_softmax(input))
            
            log_pt = F.log_softmax(inputs, dim=1)
            ce_loss = - (targets * log_pt).sum(dim=1) # (N, H, W)
            
            # Create pt for Focal
            pt = torch.exp(log_pt)
            pt = (targets * pt).sum(dim=1) # get prob of target class
            
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
            
            # Ignore index handling for one-hot is tricky unless we have a mask.
            # Assuming one-hot targets are valid (sum to 1). If ignore_index was used to zero out, sum is 0.
            # Let's assume passed targets are valid.
            
        else: # Indices (N, H, W)
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index, weight=self.alpha)
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, reduction='mean', ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: (N, C, H, W) logits
        # targets: (N, C, H, W) one-hot or (N, H, W) indices
        
        num_classes = inputs.shape[1]
        
        # Apply Softmax to get probabilities
        inputs = F.softmax(inputs, dim=1)
        
        # One-hot encode targets if needed
        if targets.dim() == 3:
            targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        else:
            targets_one_hot = targets
        
        # Calculate intersection and union
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice
        
        # Mask out ignore_index (if handled by one-hot - usually one-hot fails if ignore_index is large)
        # Assuming inputs and targets are valid 0...C-1
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss

class HybridLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, smooth=1e-6, focal_weight=0.5, dice_weight=0.5, ignore_index=255):
        super(HybridLoss, self).__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, ignore_index=ignore_index)
        self.dice = DiceLoss(smooth=smooth, ignore_index=ignore_index)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        loss = self.focal_weight * self.focal(inputs, targets) + self.dice_weight * self.dice(inputs, targets)
        return loss
