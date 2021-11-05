"""Probe importer and high level functions."""
from yuntu.soundscape.processors.probes.crosscorr import CrossCorrelationProbe

def probe(ptype="cross_correlation", **kwargs):
    """Create probe of type 'ptype'."""
    if ptype == "cross_correlation":
        return CrossCorrelationProbe(**kwargs)
    raise NotImplementedError(f"Probe type {ptype} not found.")
