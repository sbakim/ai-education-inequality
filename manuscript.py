"""Primary manuscript configuration: a common fixed evaluation horizon."""
from model import Config

PRIMARY_HORIZON = 50

def PC(**kwargs):
    """Config evaluated at exactly PRIMARY_HORIZON periods (no early stopping)."""
    kwargs = dict(kwargs)
    kwargs["T_max"] = PRIMARY_HORIZON
    kwargs["T_post"] = PRIMARY_HORIZON
    kwargs["patience"] = PRIMARY_HORIZON + 1
    return Config(**kwargs)
