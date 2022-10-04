"""Probe importer and high level functions."""
from yuntu.soundscape.processors.probes.crosscorr import CrossCorrelationProbe

def probe(ptype="cross_correlation", **kwargs):
    """Create probe of specified type.

    Parameters
    ----------
    ptype : str
        The type of probe to create (currently, only 'cross_correlation').

    Returns
    -------
    probe : Probe
        The constructed probe.

    Raises
    ------
    NotImplementedError
        When 'ptype' is not found.

    """
    if ptype == "cross_correlation":
        return CrossCorrelationProbe(**kwargs)
    raise NotImplementedError(f"Probe type {ptype} not found.")
