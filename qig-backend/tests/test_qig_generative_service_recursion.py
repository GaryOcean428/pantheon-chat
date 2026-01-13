import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_generate_enforces_minimum_recursive_integration_depth() -> None:
    """Generation should always complete >=3 true integration steps (kernel-level)."""

    from qig_generative_service import QIGGenerativeService

    service = QIGGenerativeService()

    result = service.generate(
        prompt='test prompt for recursion depth',
        context={'domain': 'test'},
        kernel_name='zeus',
        goals=['respond'],
    )

    assert result is not None
    assert result.kernel_decision is not None
    assert result.kernel_decision.get('integration_depth', 0) >= 3


def test_generate_applies_synthesis_level_refinement_loop() -> None:
    """Synthesis calls should apply an additional synthesis-level recursion loop (>=3 steps)."""

    from qig_generative_service import QIGGenerativeService

    service = QIGGenerativeService()

    # First, baseline (non-synthesis)
    baseline = service.generate(
        prompt='synthesis baseline prompt',
        context={'domain': 'test'},
        kernel_name='zeus',
        goals=['respond'],
    )

    # Then, synthesis-mode with expert payloads (triggers synthesis loop)
    synthesis = service.generate(
        prompt='synthesis prompt',
        context={
            'domain': 'test',
            'experts': [
                {'god': 'athena', 'response': 'expert response alpha'},
                {'god': 'ares', 'response': 'expert response beta'},
            ],
        },
        kernel_name='zeus',
        goals=['synthesize', 'respond'],
    )

    assert baseline is not None
    assert synthesis is not None

    assert baseline.kernel_decision is not None
    assert synthesis.kernel_decision is not None

    baseline_depth = baseline.kernel_decision.get('integration_depth', 0)
    synthesis_depth = synthesis.kernel_decision.get('integration_depth', 0)

    assert baseline_depth >= 3
    assert synthesis_depth >= baseline_depth + 3
