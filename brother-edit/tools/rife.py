"""Standalone RIFE v4.26 frame interpolation on CPU (architecture from vs-rife, MIT).

Usage:
    r = Rife("/path/to/flownet_v4.26.pkl")
    mid = r.interp(frame_a, frame_b, 0.5)   # uint8 HxWx3 RGB in, uint8 out
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.set_grad_enabled(False)


def warp(x, flow, div, grid):
    flow = torch.cat([flow[:, 0:1] / div[0], flow[:, 1:2] / div[1]], 1)
    g = (grid + flow).permute(0, 2, 3, 1)
    return F.grid_sample(x, g, mode="bilinear", padding_mode="border", align_corners=True)


def conv(i, o, k=3, s=1, p=1, d=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, p, dilation=d, bias=True), nn.LeakyReLU(0.2, True))


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.relu(self.cnn0(x.clamp(0.0, 1.0)))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


class ResConv(nn.Module):
    def __init__(self, c, d=1):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, d, dilation=d)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)))
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, cin, c=64):
        super().__init__()
        self.conv0 = nn.Sequential(conv(cin, c // 2, 3, 2, 1), conv(c // 2, c, 3, 2, 1))
        self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1), nn.PixelShuffle(2))

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear")
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear") / scale
            x = torch.cat((x, flow), 1)
        tmp = self.lastconv(self.convblock(self.conv0(x)))
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear")
        return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]


class IFNet(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.block0 = IFBlock(7 + 8, c=192)
        self.block1 = IFBlock(8 + 4 + 8 + 8, c=128)
        self.block2 = IFBlock(8 + 4 + 8 + 8, c=96)
        self.block3 = IFBlock(8 + 4 + 8 + 8, c=64)
        self.block4 = IFBlock(8 + 4 + 8 + 8, c=32)
        self.encode = Head()
        self.scale_list = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]

    def forward(self, img0, img1, timestep, div, grid, f0, f1):
        blocks = [self.block0, self.block1, self.block2, self.block3, self.block4]
        flow = mask = feat = None
        w0, w1 = img0, img1
        for i in range(5):
            if flow is None:
                flow, mask, feat = blocks[i](torch.cat((img0, img1, f0, f1, timestep), 1), None, self.scale_list[i])
            else:
                wf0 = warp(f0, flow[:, :2], div, grid)
                wf1 = warp(f1, flow[:, 2:4], div, grid)
                fd, mask, feat = blocks[i](torch.cat((w0, w1, wf0, wf1, timestep, mask, feat), 1), flow, self.scale_list[i])
                flow = flow + fd
            w0 = warp(img0, flow[:, :2], div, grid)
            w1 = warp(img1, flow[:, 2:4], div, grid)
        m = torch.sigmoid(mask)
        return w0 * m + w1 * (1 - m)


class Rife:
    def __init__(self, weights, scale=1.0, threads=None):
        if threads:
            torch.set_num_threads(threads)
        sd = torch.load(weights, map_location="cpu")
        sd = {k.replace("module.", ""): v for k, v in sd.items() if "module." in k}
        self.net = IFNet(scale)
        self.net.load_state_dict(sd, strict=False)
        self.net.eval()
        self.mod = max(64, int(64 / scale))
        self._grid = {}
        self._enc = {}

    def _prep(self, key, img):
        if key is not None and key in self._enc:
            return self._enc[key]
        h, w = img.shape[:2]
        ph, pw = math.ceil(h / self.mod) * self.mod, math.ceil(w / self.mod) * self.mod
        t = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
        t = F.pad(t, (0, pw - w, 0, ph - h))
        out = (t, self.net.encode(t))
        if key is not None:
            if len(self._enc) > 24:
                self._enc.pop(next(iter(self._enc)))
            self._enc[key] = out
        return out

    def interp(self, a, b, t, ka=None, kb=None):
        h, w = a.shape[:2]
        ia, fa = self._prep(ka, a)
        ib, fb = self._prep(kb, b)
        ph, pw = ia.shape[2:]
        if (ph, pw) not in self._grid:
            gx = torch.linspace(-1.0, 1.0, pw).view(1, 1, 1, pw).expand(-1, -1, ph, -1)
            gy = torch.linspace(-1.0, 1.0, ph).view(1, 1, ph, 1).expand(-1, -1, -1, pw)
            div = torch.tensor([(pw - 1.0) / 2.0, (ph - 1.0) / 2.0])
            self._grid[(ph, pw)] = (div, torch.cat([gx, gy], 1))
        div, grid = self._grid[(ph, pw)]
        ts = torch.full([1, 1, ph, pw], float(t))
        out = self.net(ia, ib, ts, div, grid, fa, fb)[:, :, :h, :w]
        return (out[0].permute(1, 2, 0).clamp_(0, 1).mul_(255.0).add_(0.5)).byte().numpy()
