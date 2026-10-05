"""Minimal compatibility layer for the PULSE import surface used by the demo."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch
from torch import nn


class PulseModel(nn.Module):
    """Compatibility stub matching the repo's expected API."""

    def __init__(
        self,
        modality_dims: Dict[str, int],
        latent_dim: int = 16,
        hidden_dim: int = 32,
        temporal_model: str = "recurrent",
        **_: Any,
    ) -> None:
        super().__init__()
        self.modality_dims = {str(name): int(value) for name, value in dict(modality_dims).items()}
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.temporal_model = temporal_model

        self.encoders = nn.ModuleDict(
            {name: nn.Linear(dim, hidden_dim) for name, dim in self.modality_dims.items()}
        )
        self.state_to_latent = nn.Linear(hidden_dim, latent_dim)
        self.decoders = nn.ModuleDict(
            {name: nn.Linear(latent_dim, dim) for name, dim in self.modality_dims.items()}
        )

    def _coerce(self, tensor: Optional[torch.Tensor], *, expected_dim: int) -> torch.Tensor:
        if tensor is None:
            return torch.zeros((1, expected_dim), dtype=torch.float32)
        tensor = torch.as_tensor(tensor, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[-1] != expected_dim:
            tensor = tensor[..., :expected_dim]
        return tensor

    def forward(
        self,
        visits: Iterable[Dict[str, Optional[torch.Tensor]]],
        masks: Optional[Iterable[Dict[str, int]]] = None,
        *,
        return_all_visit_states: bool = False,
    ):
        del masks
        states: List[torch.Tensor] = []
        outputs: List[Dict[str, torch.Tensor]] = []
        previous_state: Optional[torch.Tensor] = None

        for visit in visits:
            state = torch.zeros((1, self.latent_dim), dtype=torch.float32)
            for name, dim in self.modality_dims.items():
                if name in visit and visit[name] is not None:
                    x = self._coerce(visit[name], expected_dim=dim)
                    state = state + self.encoders[name](x.reshape(x.shape[0], -1)).mean(dim=0, keepdim=True)
            state = self.state_to_latent(state)
            if previous_state is not None:
                state = 0.5 * state + 0.5 * previous_state
            previous_state = state
            states.append(state)
            outputs.append({name: self.decoders[name](state) for name in self.modality_dims})

        if return_all_visit_states:
            return {"visit_states": states}
        return outputs

    def impute_missing(
        self,
        visits: Iterable[Dict[str, Optional[torch.Tensor]]],
        masks: Optional[Iterable[Dict[str, int]]] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        results: List[Dict[str, torch.Tensor]] = []
        decoded = self.forward(visits, masks=masks, return_all_visit_states=False)
        visits_list = list(visits)
        for visit, out in zip(visits_list, decoded):
            result: Dict[str, torch.Tensor] = {}
            for name, dim in self.modality_dims.items():
                if name in visit and visit[name] is not None:
                    result[name] = self._coerce(visit[name], expected_dim=dim)
                else:
                    result[name] = out.get(name, torch.zeros((1, dim), dtype=torch.float32))
            results.append(result)
        return results
