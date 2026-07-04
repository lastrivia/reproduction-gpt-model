from torch import nn

def get_module_class(name: str) -> type[nn.Module]:
    cls = getattr(nn, name, None)

    if cls is None:
        raise NotImplementedError

    if not isinstance(cls, type) or not issubclass(cls, nn.Module):
        raise TypeError

    return cls
