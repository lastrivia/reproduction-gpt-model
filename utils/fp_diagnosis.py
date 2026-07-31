from typing import Any

import torch
from torch import nn


def _nonfinite_summary(tensor: torch.Tensor) -> dict[str, Any]:
    tensor = tensor.detach()
    nan_mask = torch.isnan(tensor)
    posinf_mask = torch.isposinf(tensor)
    neginf_mask = torch.isneginf(tensor)
    good_mask = torch.isfinite(tensor)
    total = tensor.numel()

    return {
        "total": total,
        "shape": list(tensor.shape),
        "count": {
            "nan": int(nan_mask.sum().item()),
            "inf": int(posinf_mask.sum().item()),
            "ninf": int(neginf_mask.sum().item()),
            "good": int(good_mask.sum().item()),
        },
    }


def _bad_named_tensors(named_tensors) -> list[dict[str, Any]]:
    bad_tensors = []
    for name, tensor in named_tensors:
        summary = _nonfinite_summary(tensor)
        count = summary["count"]
        if count["nan"] + count["inf"] + count["ninf"] > 0:
            bad_tensors.append({
                "name": name,
                **summary,
            })
    return bad_tensors


def diagnose_loss(loss: torch.Tensor) -> dict[str, Any]:
    return _nonfinite_summary(loss)


def diagnose_parameters(model: nn.Module) -> dict[str, Any]:
    bad_parameters = _bad_named_tensors(model.named_parameters())
    return {
        "bad_parameter_count": len(bad_parameters),
        "bad_parameters": bad_parameters,
    }


def diagnose_gradients(model: nn.Module) -> dict[str, Any]:
    named_gradients = (
        (name, parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    )
    bad_gradients = _bad_named_tensors(named_gradients)
    return {
        "bad_gradient_count": len(bad_gradients),
        "bad_gradients": bad_gradients,
    }


def diagnose_optimizer_state(model: nn.Module, optimizer) -> dict[str, Any]:
    named_parameters = dict(model.named_parameters())
    named_state_tensors = []
    for name, parameter in named_parameters.items():
        state = optimizer.state.get(parameter)
        if not state:
            continue
        for state_name, value in state.items():
            if torch.is_tensor(value) and torch.is_floating_point(value):
                named_state_tensors.append((f"{name}.{state_name}", value))

    bad_state_tensors = _bad_named_tensors(named_state_tensors)
    return {
        "bad_state_tensor_count": len(bad_state_tensors),
        "bad_state_tensors": bad_state_tensors,
    }
