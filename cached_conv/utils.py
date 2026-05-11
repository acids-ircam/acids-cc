import torch.nn as nn

class Branches(nn.Module):

    def __init__(self, *branches, delays=None, cumulative_delay=0, stride=1):
        super().__init__()
        self.branches = nn.ModuleList(branches)

        if delays is None:
            delays = list(map(lambda x: x.cumulative_delay, self.branches))

        max_delay = max(delays)

        self.cumulative_delay = int(cumulative_delay * stride) + max_delay

    def forward(self, x):
        outs = []
        for branch in self.branches:
            outs.append(branch(x))
        return outs
