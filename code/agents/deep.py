"""DQN (no memory) and DRQN (LSTM) with Double-DQN targets."""
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cores import CORES


class Encoder(nn.Module):
    """Flattened k x k window + (row, col) goal offset -> two fully-connected hidden layers."""

    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.out_dim = hidden

    def forward(self, obs):  # [..., obs_dim]
        return self.fc(obs)


class QNet(nn.Module):
    """DQN: input -> FC -> FC -> Q(4).   DRQN: input -> FC -> FC -> LSTM(hidden) -> Q(4)."""

    def __init__(self, obs_dim, n_actions=4, hidden=128, core=None):
        super().__init__()
        self.enc = Encoder(obs_dim, hidden)
        self.core = CORES[core](hidden, hidden) if core else None
        if core:
            out = self.core.output_dim
        else:
            # parameter-matched replacement for the LSTM: one FC layer of width w such that
            # (hidden+1)*w + (w+1)*A  ~=  LSTM params + (hidden+1)*A
            lstm_p = 4 * (hidden * hidden * 2 + 2 * hidden)
            out = round((lstm_p + (hidden + 1) * n_actions - n_actions) / (hidden + 1 + n_actions))
            self.match = nn.Sequential(nn.Linear(hidden, out), nn.ReLU())
        self.head = nn.Linear(out, n_actions)

    def forward(self, obs, state=None):
        z = self.enc(obs)
        if self.core is not None:
            z, state = self.core(z, state)
        else:
            z = self.match(z)
        return self.head(z), state


class _DeepBase:
    def __init__(self, obs_dim, k, n_actions=4, lr=5e-4, gamma=0.99, batch_size=32,
                 buffer_size=50_000, target_update=1000, train_every=4, learn_start=1000,
                 hidden=128, device="cpu", core=None, **_):
        self.nA, self.gamma, self.bs = n_actions, gamma, batch_size
        self.target_update, self.train_every, self.learn_start = target_update, train_every, learn_start
        self.device = torch.device(device)
        self.net = QNet(obs_dim, n_actions, hidden, core).to(self.device)
        self.tgt = QNet(obs_dim, n_actions, hidden, core).to(self.device)
        self.tgt.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.buffer_size = buffer_size
        self.n_steps = 0

    def n_params(self):
        return sum(p.numel() for p in self.net.parameters())

    def maybe_update(self, step):
        loss = None
        if self.n_steps >= self.learn_start and step % self.train_every == 0:
            loss = self.update()
        if step % self.target_update == 0:
            self.tgt.load_state_dict(self.net.state_dict())
        return loss

    def _opt_step(self, loss):
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.opt.step()
        return float(loss.detach())

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.tgt.load_state_dict(self.net.state_dict())


class DQNAgent(_DeepBase):
    recurrent = False

    def __init__(self, obs_dim, k, **kw):
        kw.pop("core", None)
        super().__init__(obs_dim, k, core=None, **kw)
        self.buf = deque(maxlen=self.buffer_size)

    def reset_state(self):
        pass

    @torch.no_grad()
    def act(self, obs, eps, rng):
        if rng.random() < eps:
            return int(rng.integers(self.nA))
        q, _ = self.net(torch.as_tensor(obs, device=self.device)[None])
        return int(q.argmax(-1))

    def observe(self, o, a, r, o2, terminated, truncated=False):
        # only `terminated` stops bootstrapping; `truncated` is stored for bookkeeping/analysis
        self.buf.append((o, a, r, o2, float(terminated), float(truncated)))
        self.n_steps += 1

    def end_episode(self):
        pass

    def update(self):
        batch = random.sample(self.buf, self.bs)
        o, a, r, o2, d, _tr = (torch.as_tensor(np.array(x), device=self.device) for x in zip(*batch))
        q = self.net(o)[0].gather(1, a.long()[:, None]).squeeze(1)
        with torch.no_grad():
            a2 = self.net(o2)[0].argmax(1, keepdim=True)
            y = r.float() + self.gamma * (1 - d) * self.tgt(o2)[0].gather(1, a2).squeeze(1)
        return self._opt_step(F.smooth_l1_loss(q, y))


class DRQNAgent(_DeepBase):
    """DRQN with whole-episode replay; sampled sub-sequences of length seq_len,
    hidden state zero-initialised at (start - burn_in); loss masked on the burn-in prefix
    (R2D2-style burn-in, without stored hidden states)."""
    recurrent = True

    def __init__(self, obs_dim, k, core="lstm", seq_len=16, burn_in=8, **kw):
        super().__init__(obs_dim, k, core=core, **kw)
        self.seq_len, self.burn_in = seq_len, burn_in
        self.episodes = deque()
        self.stored = 0
        self._cur = []
        self.state = None

    def reset_state(self):
        self.state = self.net.core.init_state(1, self.device)

    @torch.no_grad()
    def act(self, obs, eps, rng):
        q, self.state = self.net(torch.as_tensor(obs, device=self.device)[None, None], self.state)
        if rng.random() < eps:
            return int(rng.integers(self.nA))
        return int(q[0, -1].argmax())

    def observe(self, o, a, r, o2, terminated, truncated=False):
        self._cur.append((o, a, r, o2, float(terminated), float(truncated)))
        self.n_steps += 1

    def end_episode(self):
        if not self._cur:
            return
        o, a, r, o2, d, tr = zip(*self._cur)
        # episode tuple: obs[T+1], act[T], rew[T], terminated[T], truncated[T]
        ep = (np.array(o + (o2[-1],), np.float32), np.array(a), np.array(r, np.float32),
              np.array(d, np.float32), np.array(tr, np.float32))
        self.episodes.append(ep)
        self.stored += len(a)
        while self.stored > self.buffer_size:
            self.stored -= len(self.episodes.popleft()[1])
        self._cur = []

    def update(self):
        if not self.episodes:
            return None
        L, B, Bi = self.seq_len, self.burn_in, self.bs
        T = L + B
        obs = np.zeros((Bi, T + 1, self.episodes[0][0].shape[1]), np.float32)
        act = np.zeros((Bi, T), np.int64); rew = np.zeros((Bi, T), np.float32)
        done = np.ones((Bi, T), np.float32); mask = np.zeros((Bi, T), np.float32)  # padded steps masked; mask = np.zeros((Bi, T), np.float32)
        for i in range(Bi):
            eo, ea, er, ed, _etr = self.episodes[random.randrange(len(self.episodes))]
            n = len(ea)
            s = random.randrange(n)                 # first trained step
            b = max(0, s - B)                       # burn-in start (hidden state = 0 here)
            e = min(n, s + L)
            m = e - b
            obs[i, :m + 1] = eo[b:e + 1]
            act[i, :m], rew[i, :m], done[i, :m] = ea[b:e], er[b:e], ed[b:e]
            mask[i, s - b:m] = 1.0
        t = lambda x: torch.as_tensor(x, device=self.device)
        obs, act, rew, done, mask = map(t, (obs, act, rew, done, mask))
        st = self.net.core.init_state(Bi, self.device)
        q_all, _ = self.net(obs, st)                         # [B, T+1, A]
        q = q_all[:, :-1].gather(2, act[..., None]).squeeze(-1)
        with torch.no_grad():
            qt, _ = self.tgt(obs, self.tgt.core.init_state(Bi, self.device))
            a2 = q_all[:, 1:].argmax(-1, keepdim=True)
            y = rew + self.gamma * (1 - done) * qt[:, 1:].gather(2, a2).squeeze(-1)
        loss = (F.smooth_l1_loss(q, y, reduction="none") * mask).sum() / mask.sum().clamp(min=1)
        return self._opt_step(loss)
