"""Pluggable recurrent cores for DRQN.

Interface every core must implement (LSTM is the thesis setting; other cores can be added later):
    core = Core(input_dim, hidden_dim)
    state = core.init_state(batch_size, device)       # any pytree of tensors
    out, state = core(x, state)                        # x: [B, T, input_dim] -> out: [B, T, hidden_dim]
    core.output_dim                                    # int
To add a new core, subclass RecurrentCore and register it in CORES.
"""
import torch
import torch.nn as nn


class RecurrentCore(nn.Module):
    output_dim: int

    def init_state(self, batch, device):
        raise NotImplementedError

    def forward(self, x, state):
        raise NotImplementedError


class LSTMCore(RecurrentCore):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.output_dim, self.h = hidden_dim, hidden_dim

    def init_state(self, batch, device):
        z = torch.zeros(1, batch, self.h, device=device)
        return (z, z.clone())

    def forward(self, x, state):
        return self.rnn(x, state)


CORES = {"lstm": LSTMCore}
