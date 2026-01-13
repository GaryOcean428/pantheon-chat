import os
import sys

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_observation_action_has_no_emergency_stop() -> None:
    from qig_core.self_observer import ObservationAction

    assert not hasattr(ObservationAction, 'EMERGENCY_STOP')
    assert 'emergency_stop' not in {a.value for a in ObservationAction}


def test_self_observer_never_emits_emergency_stop_action() -> None:
    from qig_core.self_observer import ObservationAction, SelfObserver
    from qigkernels.physics_constants import BASIN_DIM

    observer = SelfObserver(kernel_name='test', enable_course_correction=True)

    basin = np.ones(BASIN_DIM, dtype=np.float64)
    basin = basin / np.sum(basin)

    # Run past the minimum-history guard.
    actions = []
    for i in range(25):
        obs = observer.observe_token(
            token=f't{i}',
            basin=basin,
            phi=1.0,
            kappa=64.2,
            generated_text='test generation text that is long enough for metrics'
        )
        actions.append(obs.action)

    assert actions
    assert all(action in {ObservationAction.CONTINUE, ObservationAction.COURSE_CORRECT, ObservationAction.PAUSE_REFLECT} for action in actions)
