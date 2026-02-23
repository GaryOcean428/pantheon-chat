# kernel_lifecycle.py patch — apply manually if MCP push of full file fails

## Patch 1 — after `from pantheon_registry import` block

```python
# TCP v6.1 — VoterRegistry: live φ/κ for governance vote weighting
_VOTER_REGISTRY = None
_VOTER_REGISTRY_ATTEMPTED = False

def _vr():
    global _VOTER_REGISTRY, _VOTER_REGISTRY_ATTEMPTED
    if _VOTER_REGISTRY is not None:
        return _VOTER_REGISTRY
    if _VOTER_REGISTRY_ATTEMPTED:
        return None
    _VOTER_REGISTRY_ATTEMPTED = True
    try:
        from olympus.voter_registry import get_voter_registry
        _VOTER_REGISTRY = get_voter_registry()
    except ImportError:
        pass
    return _VOTER_REGISTRY
```

## Patch 2 — after `self._active_gods[kernel.god_name] = ...` block in spawn()

```python
            # TCP v6.1: register god with voter registry for live φ/κ weighting
            vr = _vr()
            if vr is not None:
                vr.register(
                    god_name=kernel.god_name.capitalize(),
                    kernel_id=kernel_id,
                    phi=kernel.phi,
                    kappa=kernel.kappa,
                )
```

## Patch 3 — after `vr.register()` call, add update hook in update_metrics() or tick()

Find wherever `kernel.phi` and `kernel.kappa` are updated after generation cycles,
then add:

```python
vr = _vr()
if vr and kernel.god_name:
    vr.update(kernel.god_name.capitalize(), kernel.phi, kernel.kappa)
```
