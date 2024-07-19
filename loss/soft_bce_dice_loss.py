import torch 
import torch.nn as nn 

class SoftDiceBCELoss(nn.Module):
    def __init__(self, smooth=0.7):
        super(SoftDiceBCELoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def feed_forward(self, logits, targets):
        num = targets.size(0)
        
        logits = logits.contiguous()
        targets = targets.contiguous()

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = m1 * m2

        score = (
            2.0
            * (intersection.sum(1) + self.smooth)
            / (m1.sum(1) + m2.sum(1) + self.smooth)
        )
        score = 1 - score.sum() / num
        return score
    
    def forward(self, logits, targets):
        
        dice_loss = self.feed_forward(logits, targets)
        bce_loss = self.bce(logits, targets)
        
        return dice_loss / 2 + bce_loss / 2