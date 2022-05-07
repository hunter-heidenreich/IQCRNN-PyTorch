import torch
import torch.nn as nn


class MLPCritic(nn.Module):
    
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            n_layers: int,
            nonlinearity: str = "tanh",
            device=None,
            dtype=None,
            use_trunc_normal=False,
    ):
        super(MLPCritic, self).__init__()

        self._in_dim = input_size
        self._hid_dim = hidden_size

        self._n_layers = n_layers
        self._nl = nonlinearity

        if self._nl == 'tanh':
            self._act = torch.tanh
        else:
            # identity
            self._act = lambda x: x

        self._init_std = 1e-4
        self._use_trunc_normal = use_trunc_normal

        self._device = device
        self._dtype = dtype

        # self._init_std = 1

        self._in_lin = nn.Linear(
            self._in_dim,
            self._hid_dim,
            bias=True,
            device=self._device,
            dtype=self._dtype
        )
        if self._use_trunc_normal:
            nn.init.trunc_normal_(self._in_lin.weight, std=self._init_std)
            nn.init.zeros_(self._in_lin.bias)

        hid_layers = []
        for _ in range(self._n_layers - 1):
            lx = nn.Linear(
                self._hid_dim,
                self._hid_dim,
                bias=True,
                device=self._device,
                dtype=self._dtype
            )
            if self._use_trunc_normal:
                nn.init.trunc_normal_(lx.weight, std=self._init_std)
                nn.init.zeros_(lx.bias)

            hid_layers.append(lx)
        self._hidden_layers = nn.ModuleList(hid_layers)

        self._out_lin = nn.Linear(
            self._hid_dim,
            1,  # critic predicts a scalar
            bias=True,
            device=self._device,
            dtype=self._dtype
        )
        if self._use_trunc_normal:
            nn.init.trunc_normal_(self._out_lin.weight, std=self._init_std)
            nn.init.zeros_(self._out_lin.bias)

    def forward(self, obs):
        z = self._in_lin(obs)
        h = self._act(z)

        for lx in self._hidden_layers:
            z = lx(h)
            h = self._act(z)

        return self._out_lin(h)
