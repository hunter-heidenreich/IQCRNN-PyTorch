import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tfx

from torch.distributions import Normal


class RobustRNNCell(nn.Module):

    def __init__(
            self,
            input_size: int,   # y(k)
            hidden_size: int,  # w(k)/v(k)
            state_size: int,   # xi(k)
            output_size: int,  # u(k)
            bias: bool = True,
            nonlinearity: str = "tanh",
            device=None,
            dtype=None,
            use_trunc_normal=False,
    ):
        super(RobustRNNCell, self).__init__()

        self._ny = input_size
        self._nh = hidden_size
        self._nx = state_size
        self._nu = output_size

        self._init_std = 1e-4
        self._use_trunc_normal = use_trunc_normal

        self._bias = bias
        self._nl = nonlinearity
        if self._nl == 'tanh':
            self._act = torch.tanh
        elif self._nl == 'relu':
            self._act = tfx.relu
        else:
            # identity function
            self._act = lambda x: x

        self._device = device
        self._dtype = dtype

        #  xi(k+1) = AK  xi(k) + BK1 w(k) + BK2 y(k)
        #  u(k)    = CK1 xi(k) + DK1 w(k) + DK2 y(k)
        #  v(k)    = CK2 xi(k) + DK3 y(k)
        #  w(k)    = phi(v(k))
        #
        #  xi: hidden state
        #  y:  input
        #  v:  after dense
        #  w:  after activation
        #  u:  output

        # xi
        self._W_xi = nn.Linear(
            self._nx + self._nh + self._ny,  # input size
            self._nx,  # output_size
            bias=self._bias,
            device=self._device,
            dtype=self._dtype,
        )
        if self._use_trunc_normal:
            nn.init.trunc_normal_(self._W_xi.weight, std=self._init_std)
            if self._bias:
              nn.init.zeros_(self._W_xi.bias)

        # u
        self._W_u = nn.Linear(
            self._nx + self._nh + self._ny,  # input size
            self._nu,  # output_size
            bias=self._bias,
            device=self._device,
            dtype=self._dtype,
        )
        if self._use_trunc_normal:
            nn.init.trunc_normal_(self._W_u.weight, std=self._init_std)
            if self._bias:
                nn.init.zeros_(self._W_u.bias)

        # v
        self._W_v = nn.Linear(
            self._nx + self._ny,  # input size
            self._nh,  # output_size
            bias=self._bias,
            device=self._device,
            dtype=self._dtype,
        )
        if self._use_trunc_normal:
            nn.init.trunc_normal_(self._W_v.weight, std=self._init_std)
            if self._bias:
                nn.init.zeros_(self._W_v.bias)

        # log std. dev.
        # self.log_std = nn.Parameter(
        #     data=torch.full((self._nu,), np.log(0.2)),
        #     requires_grad=True
        # )
        self.log_std = nn.Linear(
            1,
            self._nu,
            bias=False,
            device=self._device,
            dtype=self._dtype,
        )
        nn.init.constant_(self.log_std.weight, np.log(0.2))

    def forward(
            self,
            y,        # ([B]atch, [I]nput), or just (I,)
            xi=None,  # ([B]atch, [S]tate), or just (S,)
    ):
        assert y.dim() in (1, 2)

        # Init to all zeros if not specified
        if xi is None:
            xi = torch.zeros((self._nx,) if y.dim() == 1 else (y.size()[0], self._nx))

        #  v(k)    = CK2 xi(k) + DK3 y(k)
        xi_y = torch.cat((xi, y), dim=-1)  # (B, S + I), (S + I,)
        v = self._W_v(xi_y)  # (B, [H]idden), (H,)

        #  w(k)    = phi(v(k))
        w = self._act(v)  # (B, H), (H,)

        xi_w_y = torch.cat((xi, w, y), dim=-1)  # (B, S + H + I), (S + H + I,)

        #  u(k)    = CK1 xi(k) + DK1 w(k) + DK2 y(k)
        u = self._W_u(xi_w_y)  # (B, [O]utput), (O,)

        #  xi(k+1) = AK  xi(k) + BK1 w(k) + BK2 y(k)
        xi_ = self._W_xi(xi_w_y)  # (B, S), (S,)

        log_std = self.log_std(torch.ones(1) if y.dim() == 1 else torch.ones((u.size(0), 1)))

        return u, xi_, log_std

    def get_weights(self):
        return {
            'AK': self._W_xi.weight.data[:, :self._nx].numpy(),
            'BK1': self._W_xi.weight.data[:,
                   self._nx:self._nx + self._nh].numpy(),
            'BK2': self._W_xi.weight.data[:, self._nx + self._nh:].numpy(),

            'CK1': self._W_u.weight.data[:, :self._nx].numpy(),
            'DK1': self._W_u.weight.data[:,
                   self._nx:self._nx + self._nh].numpy(),
            'DK2': self._W_u.weight.data[:, self._nx + self._nh:].numpy(),

            'CK2': self._W_v.weight.data[:, :self._nx].numpy(),
            'DK3': self._W_v.weight.data[:, self._nx:].numpy(),
        }

    def set_weights(
            self,
            AK, BK1, BK2,
            CK1, DK1, DK2,
            CK2, DK3,
    ):
        with torch.no_grad():
            self._W_xi.weight.data = torch.cat((AK, BK1, BK2), dim=-1)
            self._W_u.weight.data = torch.cat((CK1, DK1, DK2), dim=-1)
            self._W_v.weight.data = torch.cat((CK2, DK3), dim=-1)


class RobRNNActor(RobustRNNCell):

    def forward(
            self,
            y,        # ([B]atch, [I]nput), or just (I,)
            xi=None,  # ([B]atch, [S]tate), or just (S,)
            eps=1e-8,
    ):
        u, xi_, log_std = super(RobRNNActor, self).forward(y, xi=xi)

        dist = Normal(u, (eps + log_std).exp())
        action = dist.sample()
        log_p = dist.log_prob(action)

        return u, xi_, dist, action, log_p


class RobustRNNCellTilde(nn.Module):

    def __init__(
            self,
            input_size: int,   # y(k)
            hidden_size: int,  # w(k)/v(k)/z(k)
            state_size: int,   # xi(k)
            output_size: int,  # u(k)
            Aphi=None,
            Bphi=None,
            bias: bool = True,
            nonlinearity: str = "tanh",
            device=None,
            dtype=None,
            use_trunc_normal=False,
    ):
        super(RobustRNNCellTilde, self).__init__()

        self._ny = input_size
        self._nh = hidden_size
        self._nx = state_size
        self._nu = output_size

        self._init_std = 1e-4
        self._use_trunc_normal = use_trunc_normal

        self._bias = bias
        self._nl = nonlinearity
        if self._nl == 'tanh':
            self._act = torch.tanh
        elif self._nl == 'relu':
            self._act = tfx.relu
        else:
            # identity function
            self._act = lambda x: x

        self._Aphi = Aphi
        if self._Aphi is None:
            self._Aphi = torch.zeros(1)

        self._Bphi = Bphi
        if self._Bphi is None:
            self._Bphi = torch.ones(1)

        self._device = device
        self._dtype = dtype

        #  xi(k+1) = AK  xi(k) + BK1 w(k) + BK2 y(k)
        #  u(k)    = CK1 xi(k) + DK1 w(k) + DK2 y(k)
        #  v(k)    = CK2 xi(k) + DK3 y(k)
        #  w(k)    = phi(v(k))
        #
        #  xi: hidden state
        #  y:  input
        #  v:  after dense
        #  w:  after activation
        #  u:  output

        # xi
        self._W_xi = nn.Linear(
            self._nx + self._nh + self._ny,  # input size
            self._nx,  # output_size
            bias=self._bias,
            device=self._device,
            dtype=self._dtype,
        )
        if self._use_trunc_normal:
            nn.init.trunc_normal_(self._W_xi.weight, std=self._init_std)
            if self._bias:
                nn.init.zeros_(self._W_xi.bias)

        # u
        self._W_u = nn.Linear(
            self._nx + self._nh + self._ny,  # input size
            self._nu,  # output_size
            bias=self._bias,
            device=self._device,
            dtype=self._dtype,
        )
        if self._use_trunc_normal:
            nn.init.trunc_normal_(self._W_u.weight, std=self._init_std)
            if self._bias:
                nn.init.zeros_(self._W_u.bias)

        # v
        self._W_v = nn.Linear(
            self._nx + self._ny,  # input size
            self._nh,  # output_size
            bias=self._bias,
            device=self._device,
            dtype=self._dtype,
        )
        if self._use_trunc_normal:
            nn.init.trunc_normal_(self._W_v.weight, std=self._init_std)
            if self._bias:
                nn.init.zeros_(self._W_v.bias)

        # log std. dev.
        # self.log_std = nn.Parameter(data=torch.full((self._nu,), np.log(0.2)))
        self.log_std = nn.Linear(
            1,
            self._nu,
            bias=False,
            device=self._device,
            dtype=self._dtype,
        )
        nn.init.constant_(self.log_std.weight, np.log(0.2))

    def forward(
            self,
            y,        # ([B]atch, [I]nput), or just (I,)
            xi=None,  # ([B]atch, [S]tate), or just (S,)
    ):
        assert y.dim() in (1, 2)

        # Init to all zeros if not specified
        if xi is None:
            xi = torch.zeros((self._nx,) if y.dim() == 1 else (y.size()[0], self._nx))

        #  v(k)    = CK2 xi(k) + DK3 y(k)
        xi_y = torch.cat((xi, y), dim=-1)  # (B, S + I), (S + I,)
        v = self._W_v(xi_y)  # (B, [H]idden), (H,)

        #  z(k)    = phi~ (v(k))
        z = self.new_activation(v)  # (B, H), (H,)

        #  w(k)    = (B-A/2) z(k)
        if self._Aphi.dim() > 1:
            w = ((self._Bphi - self._Aphi) / 2) @ z  # (B, H), (H,)
        else:
            w = z * (self._Bphi - self._Aphi) / 2  # (B, H), (H,)

        xi_w_y = torch.cat((xi, w, y), dim=-1)  # (B, S + H + I), (S + H + I,)

        #  u(k)    = CK1 xi(k) + DK1 w(k) + DK2 y(k)
        u = self._W_u(xi_w_y)  # (B, [O]utput), (O,)

        #  xi(k+1) = AK  xi(k) + BK1 w(k) + BK2 y(k)
        xi_ = self._W_xi(xi_w_y)  # (B, S), (S,)

        log_std = self.log_std(
            torch.ones(1) if y.dim() == 1 else torch.ones((u.size(0), 1)))

        return u, xi_, log_std

    def new_activation(self, v):
        w = self._act(v)

        if self._Aphi.dim() > 1:
            z = (2 * torch.inverse(self._Bphi - self._Aphi)) @ (
                        w - ((self._Aphi + self._Bphi) / 2) @ v)
        else:
            z = 2 / (self._Bphi - self._Aphi) * (
                        w - (self._Aphi + self._Bphi) / 2 * v)

        return z

    def get_weights(self):
        return {
            'AK': self._W_xi.weight.data[:, :self._nx].numpy(),
            'BK1': self._W_xi.weight.data[:,
                   self._nx:self._nx + self._nh].numpy(),
            'BK2': self._W_xi.weight.data[:, self._nx + self._nh:].numpy(),

            'CK1': self._W_u.weight.data[:, :self._nx].numpy(),
            'DK1': self._W_u.weight.data[:,
                   self._nx:self._nx + self._nh].numpy(),
            'DK2': self._W_u.weight.data[:, self._nx + self._nh:].numpy(),

            'CK2': self._W_v.weight.data[:, :self._nx].numpy(),
            'DK3': self._W_v.weight.data[:, self._nx:].numpy(),
        }

    def set_weights(
            self,
            AK, BK1, BK2,
            CK1, DK1, DK2,
            CK2, DK3,
    ):
        with torch.no_grad():
            self._W_xi.weight.data = torch.cat((AK, BK1, BK2), dim=-1)
            self._W_u.weight.data = torch.cat((CK1, DK1, DK2), dim=-1)
            self._W_v.weight.data = torch.cat((CK2, DK3), dim=-1)


class RobRNNTildeActor(RobustRNNCellTilde):

    def forward(
            self,
            y,        # ([B]atch, [I]nput), or just (I,)
            xi=None,  # ([B]atch, [S]tate), or just (S,)
            eps=1e-8,
    ):
        u, xi_, log_std = super(RobRNNTildeActor, self).forward(y, xi=xi)

        dist = Normal(u, (eps + log_std).exp())
        action = dist.sample()
        log_p = dist.log_prob(action)

        return u, xi_, dist, action, log_p
