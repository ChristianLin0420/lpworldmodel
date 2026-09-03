"""Small MLP heads that consume a LATENT and a GOAL latent (round 5, T4 / V4).

One class, used twice: `V(z, g)` (T4, `d_out = 1`) and `pi(z, g)` (V4,
`d_out = act_dim_raw`). Both read only latents, so nothing here can see
`states.pth` even in principle.

Two deliberate choices, both about being reconstructible at PLAN time:

* the parameters live in a plain ``nn.Sequential`` named ``net``, so the state
  dict keys are exactly ``net.0.weight`` / ``net.0.bias`` / ``net.2.*`` /
  ``net.4.*`` and the head's three widths (3D -> H -> H -> d_out) can be read
  straight off the checkpoint without any config;
* the input is ``[z, g, z - g]`` rather than ``[z, g]``. The difference is
  already computable from the concatenation, so this adds no information -- it
  adds the INDUCTIVE BIAS that the answer depends on the displacement, which is
  what a time-to-goal and a goal-reaching action both are. Fan-in is 3D, which
  is why train.py optimises these heads with plain AdamW and NOT
  `mup_param_groups`: muP would scale their LR by base_width / (3D) and make the
  head's learning rate a function of the swept width, confounding T4 with the
  wave6/wave7 width x LR factorial.
"""

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        self.d_in, self.d_hid, self.d_out = int(d_in), int(d_hid), int(d_out)
        self.net = nn.Sequential(
            nn.Linear(int(d_in), int(d_hid)),
            nn.GELU(),
            nn.Linear(int(d_hid), int(d_hid)),
            nn.GELU(),
            nn.Linear(int(d_hid), int(d_out)),
        )

    def forward(self, z, g):
        """z, g: (..., D) -> (..., d_out). D must be d_in // 3."""
        return self.net(torch.cat([z, g, z - g], dim=-1))
