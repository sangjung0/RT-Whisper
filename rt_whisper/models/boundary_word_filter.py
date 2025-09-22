import torch

import torch.nn as nn
import numpy as np

from pathlib import Path


class BoundaryWordFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(6, 6)
        # self.hidden = nn.Linear(6, 6)
        self.output = nn.Linear(6, 1)
        self.act = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.input(x))
        # x = self.act(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: torch.device | str = torch.device("cpu"),
    ) -> "BoundaryWordFilter":
        # 일단 임의로 weights_only =False 함
        sd = torch.load(path, map_location=device, weights_only=False)
        if isinstance(sd, BoundaryWordFilter):
            return sd
        m = cls()
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        m.load_state_dict(sd)
        m.to(device)
        return m


class BoundaryWordFilterWrapper:
    def __init__(
        self,
        model: BoundaryWordFilter,
        boundary: int,
        device: torch.device | str = "cpu",
    ):
        if boundary < 0:
            raise ValueError("boundary should be non-negative")

        self.model = model
        self.boundary = boundary
        self.device = device

        self.model.to(device).eval()

    def __call__(self, start: np.ndarray, end: np.ndarray, length: int) -> np.ndarray:
        assert length > 0, "length should be positive"
        assert start.shape == end.shape, "start and end should have the same shape"
        assert np.all(start >= 0), "start should be non-negative"
        assert np.all(start <= end), "start should be less than or equal to end"
        assert np.all(end <= length), "end should be less than or equal to length"

        if self.boundary == 0:
            return np.ones_like(start, dtype=np.float32)

        mid = (start + end) * 0.5

        ss = np.clip(start / self.boundary, 0, 1.0)
        se = np.clip(end / self.boundary, 0, 1.0)
        es = np.clip((length - start) / self.boundary, 0, 1.0)
        ee = np.clip((length - end) / self.boundary, 0, 1.0)
        sm = np.clip(mid / self.boundary, 0, 1.0)
        em = np.clip((length - mid) / self.boundary, 0, 1.0)
        # sl = start / length
        # el = end / length
        # ml = mid / length
        # dl = (end - start) / length

        x = np.stack([ss, se, es, ee, sm, em], axis=1).astype(np.float32)
        x = torch.from_numpy(x).to(self.device)

        with torch.no_grad():
            y = self.model(x).squeeze(1).cpu().numpy()
        return y.astype(np.float32)

    @classmethod
    def load(
        cls, path: str | Path, boundary: int, device: torch.device | str = "cpu"
    ) -> "BoundaryWordFilterWrapper":
        model = BoundaryWordFilter.load(path, device=device)
        return cls(model=model, boundary=boundary, device=device)


__all__ = ["BoundaryWordFilter", "BoundaryWordFilterWrapper"]
