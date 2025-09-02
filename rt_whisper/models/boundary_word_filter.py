import torch
import torch.nn as nn

from pathlib import Path


class BoundaryWordFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(4, 8)
        # self.hidden = nn.Linear(8, 8)
        self.output = nn.Linear(8, 1)
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
        device: torch.device = torch.device("cpu"),
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


__all__ = ["BoundaryWordFilter"]
