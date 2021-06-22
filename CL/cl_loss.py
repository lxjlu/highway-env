import torch
import torch.nn as nn
import torch.nn.functional as F


class SupLoss(nn.Module):

    def __init__(self, temperature=0.5):
        super(SupLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        featuers: hidden vector of shap [bsz, 128]
        labels: ground truth of shap [bsz]
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')

        # 这里mask是什么, [bsz, bsz] 对角为True其他为False
        mask_label = torch.eq(labels, labels.T).float().to(device)
        # mask_pos = mask_label.masked_select(~torch.eye(batch_size).bool()).view(batch_size, -1)

        sim = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        sim_label = sim * mask_label
        sim_pos = sim_label * (~torch.eye(batch_size).bool())
        sim_neg = sim * (~mask_label.bool())

        # log
        exp_sim_pos = torch.exp(sim_pos)
        exp_sim_neg = torch.exp(sim_neg)

        log_prob = torch.log(exp_sim_pos.sum(1, keepdim=True) / exp_sim_neg.sum(1, keepdim=True))
        mean_log_prob = log_prob / (mask_label.sum(1, keepdim=True) - 1)

        # loss
        loss = -mean_log_prob
        loss = loss.view(batch_size, -1).mean()

        return loss, mean_log_prob








