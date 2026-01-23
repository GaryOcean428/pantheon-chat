"""
Integration Test for WP5.4 Kernel Rest Scheduler
================================================

Tests the end-to-end integration:
1. BaseGod has KernelRestMixin
2. Gods can update fatigue
3. Rest scheduler coordinates coverage
4. API endpoints work correctly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
from kernel_rest_scheduler import get_rest_scheduler, reset_rest_scheduler


def test_base_god_has_rest_mixin():
    """Verify BaseGod includes KernelRestMixin."""
    from olympus.base_god import BaseGod, KERNEL_REST_MIXIN_AVAILABLE
    
    # Should be available
    assert KERNEL_REST_MIXIN_AVAILABLE, "KernelRestMixin should be available"
    
    # BaseGod should have rest methods
    assert hasattr(BaseGod, 'update_rest_fatigue'), "BaseGod should have update_rest_fatigue method"
    assert hasattr(BaseGod, 'check_rest_needed'), "BaseGod should have check_rest_needed method"


def test_apollo_updates_fatigue():
    """Test that Apollo can update fatigue metrics."""
    reset_rest_scheduler()
    
    from olympus.apollo import Apollo
    apollo = Apollo()
    
    # Apollo should have rest tracking
    assert hasattr(apollo, '_rest_scheduler'), "Apollo should have _rest_scheduler"
    assert hasattr(apollo, '_kernel_rest_id'), "Apollo should have _kernel_rest_id"
    
    # Should be registered with scheduler
    scheduler = get_rest_scheduler()
    status = scheduler.get_rest_status(apollo._kernel_rest_id)
    
    assert status is not None, "Apollo should be registered with scheduler"
    assert status['kernel_name'] == 'Apollo'
    assert status['rest_policy'] == 'coordinated_alternating'


def test_athena_updates_fatigue():
    """Test that Athena can update fatigue metrics."""
    reset_rest_scheduler()
    
    from olympus.athena import Athena
    athena = Athena()
    
    # Should be registered
    scheduler = get_rest_scheduler()
    status = scheduler.get_rest_status(athena._kernel_rest_id)
    
    assert status is not None, "Athena should be registered"
    assert status['kernel_name'] == 'Athena'
    assert status['rest_policy'] == 'coordinated_alternating'


def test_apollo_athena_coupling():
    """Test Apollo-Athena coupling and coverage."""
    reset_rest_scheduler()
    
    from olympus.apollo import Apollo
    from olympus.athena import Athena
    
    apollo = Apollo()
    athena = Athena()
    
    scheduler = get_rest_scheduler()
    
    # Simulate Apollo getting fatigued
    scheduler.update_fatigue(apollo._kernel_rest_id, phi=0.8, kappa=64.0, load=0.3)
    scheduler.update_fatigue(apollo._kernel_rest_id, phi=0.35, kappa=50.0, load=0.85)
    
    # Athena stays healthy
    scheduler.update_fatigue(athena._kernel_rest_id, phi=0.75, kappa=63.0, load=0.25)
    
    # Apollo should need rest
    should_rest, reason = scheduler.should_rest(apollo._kernel_rest_id)
    assert should_rest, f"Apollo should need rest: {reason}"
    
    # Check coupling partners
    partners = scheduler.get_coupling_partners(apollo._kernel_rest_id)
    assert athena._kernel_rest_id in partners, "Athena should be Apollo's coupling partner"
    
    # Request rest - should be approved with Athena covering
    approved, approval_reason, partner = scheduler.request_rest(apollo._kernel_rest_id)
    assert approved, f"Rest should be approved: {approval_reason}"
    assert partner == athena._kernel_rest_id, "Athena should cover for Apollo"
    
    # Check statuses
    apollo_status = scheduler.get_rest_status(apollo._kernel_rest_id)
    athena_status = scheduler.get_rest_status(athena._kernel_rest_id)
    
    assert apollo_status['status'] == 'resting'
    assert apollo_status['covered_by'] == athena._kernel_rest_id
    assert athena_status['covering_for'] == apollo._kernel_rest_id


def test_hermes_minimal_rotating():
    """Test Hermes essential tier with minimal_rotating."""
    reset_rest_scheduler()
    
    from olympus.hermes import Hermes
    hermes = Hermes()
    
    scheduler = get_rest_scheduler()
    status = scheduler.get_rest_status(hermes._kernel_rest_id)
    
    assert status is not None, "Hermes should be registered"
    assert status['kernel_name'] == 'Hermes'
    assert status['tier'] == 'essential'
    assert status['rest_policy'] == 'minimal_rotating'


def test_rest_api_health():
    """Test REST API health endpoint."""
    from api_rest_scheduler import rest_api
    from flask import Flask
    
    app = Flask(__name__)
    app.register_blueprint(rest_api)
    
    with app.test_client() as client:
        response = client.get('/api/rest/health')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['status'] == 'ok'
        assert data['scheduler_available'] is True


def test_rest_api_constellation_status():
    """Test constellation status API."""
    reset_rest_scheduler()
    
    # Register some kernels
    from olympus.apollo import Apollo
    from olympus.athena import Athena
    apollo = Apollo()
    athena = Athena()
    
    from api_rest_scheduler import rest_api
    from flask import Flask
    
    app = Flask(__name__)
    app.register_blueprint(rest_api)
    
    with app.test_client() as client:
        response = client.get('/api/rest/status')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['status'] == 'ok'
        assert 'constellation' in data
        assert data['constellation']['total_kernels'] >= 2


def test_rest_api_kernel_status():
    """Test individual kernel status API."""
    reset_rest_scheduler()
    
    from olympus.apollo import Apollo
    apollo = Apollo()
    
    from api_rest_scheduler import rest_api
    from flask import Flask
    
    app = Flask(__name__)
    app.register_blueprint(rest_api)
    
    with app.test_client() as client:
        response = client.get(f'/api/rest/status/{apollo._kernel_rest_id}')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['status'] == 'ok'
        assert data['kernel']['kernel_name'] == 'Apollo'


if __name__ == '__main__':
    print("Running WP5.4 Integration Tests...")
    pytest.main([__file__, '-v', '-s'])
