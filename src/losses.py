import torch
import torch.nn as nn

class Losses(nn.Module):
    def __init__(self, loss_type, batch_size, real_label=1., fake_label=0.):
        super(Losses, self).__init__()
        assert loss_type in ['hinge', 'bce'], '現在設定できる損失関数はhingeかbceです'

        self.loss_type = loss_type
        self.bce = nn.BCEWithLogitsLoss()
        self.real_labels = torch.full((batch_size,), real_label, dtype=torch.float)
        self.fake_labels = torch.full((batch_size,), fake_label, dtype=torch.float)

    def __call__(self, x, net_type):
        assert net_type in ['generator', 'discriminator_real', 'discriminator_fake']
        if self.loss_type == 'hinge':
            if net_type == 'generator':
                return -torch.mean(x)
            elif net_type == 'discriminator_real':
                minval = torch.min(x - 1, self.fake_labels)
                return -torch.mean(minval)
            elif net_type == 'discriminator_fake':
                minval = torch.min(-x - 1, self.fake_labels)
                return -torch.mean(minval)
        elif self.loss_type == 'bce':
            if net_type == 'discriminator_fake':
                return self.bce(x, self.fake_labels)
            else: return self.bce(x, self.real_labels)
