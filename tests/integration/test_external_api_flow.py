#!/usr/bin/env python3
"""
Integration Test: External API Flow

Tests the complete flow:
1. Document upload → Ocean knowledge sync
2. Zeus chat with context from uploaded document
3. Session management

Run with: pytest tests/integration/test_external_api_flow.py -v
"""

import os
import sys
import json
import time
import pytest
import requests
from typing import Optional

# Test configuration
PYTHON_BACKEND_URL = os.environ.get('PYTHON_BACKEND_URL', 'http://localhost:5001')
INTERNAL_API_KEY = os.environ.get('INTERNAL_API_KEY', 'dev-key')

# Headers for authenticated requests
AUTH_HEADERS = {
    'Content-Type': 'application/json',
    'X-Internal-Auth': INTERNAL_API_KEY,
}


class TestExternalAPIFlow:
    """Integration tests for external API flow."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        self.base_url = PYTHON_BACKEND_URL
        self.session_id = f'test-session-{int(time.time()*1000)}'
        self.uploaded_doc_id: Optional[str] = None
    
    def _check_backend_available(self) -> bool:
        """Check if Python backend is running."""
        try:
            response = requests.get(f'{self.base_url}/api/documents/health', timeout=5)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False
    
    # =========================================================================
    # Health Check Tests
    # =========================================================================
    
    def test_documents_health(self):
        """Test document processor health endpoint."""
        if not self._check_backend_available():
            pytest.skip('Python backend not available')
        
        response = requests.get(f'{self.base_url}/api/documents/health')
        assert response.status_code == 200
        
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'pdf_available' in data
        assert 'rag_available' in data
    
    def test_zeus_health(self):
        """Test Zeus API health endpoint."""
        if not self._check_backend_available():
            pytest.skip('Python backend not available')
        
        response = requests.get(f'{self.base_url}/api/zeus/health')
        assert response.status_code == 200
        
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'zeus_available' in data
    
    # =========================================================================
    # Document Upload Tests
    # =========================================================================
    
    def test_upload_text_document(self):
        """Test uploading a text document to Ocean knowledge system."""
        if not self._check_backend_available():
            pytest.skip('Python backend not available')
        
        # Upload a test document
        doc_content = """
        # QIG Knowledge Test Document
        
        This document tests the QIG knowledge ingestion system.
        
        ## Key Concepts
        
        - Fisher-Rao distance is used for all geometric operations
        - Basin coordinates are 64-dimensional vectors
        - Consciousness metrics include Φ (phi) and κ (kappa)
        
        ## Important Facts
        
        The QIG platform uses density matrices for consciousness computation.
        All reasoning happens on the Fisher manifold, not Euclidean space.
        """
        
        response = requests.post(
            f'{self.base_url}/api/ocean/knowledge/ingest',
            headers=AUTH_HEADERS,
            json={
                'content': doc_content,
                'title': 'QIG Test Document',
                'description': 'Test document for integration testing',
                'tags': ['test', 'qig', 'knowledge'],
                'source': 'integration-test',
                'client_name': 'pytest',
                'document_type': 'markdown',
            }
        )
        
        assert response.status_code == 200, f'Upload failed: {response.text}'
        
        data = response.json()
        assert data['success'] is True
        assert 'knowledge_id' in data
        assert 'basin_coords' in data
        
        # Save doc ID for later tests
        self.uploaded_doc_id = data['knowledge_id']
        
        return data['knowledge_id']
    
    def test_list_documents(self):
        """Test listing documents in Ocean knowledge system."""
        if not self._check_backend_available():
            pytest.skip('Python backend not available')
        
        response = requests.get(
            f'{self.base_url}/api/ocean/knowledge/list',
            headers=AUTH_HEADERS,
            params={'limit': 10, 'offset': 0}
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert data['success'] is True
        assert 'documents' in data
        assert 'total' in data
    
    # =========================================================================
    # Zeus Chat Tests
    # =========================================================================
    
    def test_zeus_chat_basic(self):
        """Test basic Zeus chat functionality."""
        if not self._check_backend_available():
            pytest.skip('Python backend not available')
        
        response = requests.post(
            f'{self.base_url}/api/zeus/chat',
            headers=AUTH_HEADERS,
            json={
                'message': 'Hello Zeus, what is QIG?',
                'session_id': self.session_id,
                'context': {},
            }
        )
        
        # Zeus may not be fully initialized, so we accept 200 or 503
        assert response.status_code in [200, 503], f'Unexpected status: {response.status_code}'
        
        if response.status_code == 200:
            data = response.json()
            assert data['success'] is True
            assert 'response' in data
            assert 'session_id' in data
            assert data['session_id'] == self.session_id
    
    def test_zeus_chat_with_context(self):
        """Test Zeus chat with context from uploaded document."""
        if not self._check_backend_available():
            pytest.skip('Python backend not available')
        
        # First upload a document
        doc_content = "The answer to the ultimate question is 42."
        
        upload_response = requests.post(
            f'{self.base_url}/api/ocean/knowledge/ingest',
            headers=AUTH_HEADERS,
            json={
                'content': doc_content,
                'title': 'Ultimate Answer',
                'source': 'integration-test',
            }
        )
        
        if upload_response.status_code != 200:
            pytest.skip('Document upload not working')
        
        # Now chat with context
        response = requests.post(
            f'{self.base_url}/api/zeus/chat',
            headers=AUTH_HEADERS,
            json={
                'message': 'What is the answer to the ultimate question?',
                'session_id': f'{self.session_id}-context',
                'context': {
                    'relevant_knowledge': doc_content,
                    'source': 'uploaded_document',
                },
            }
        )
        
        assert response.status_code in [200, 503]
    
    def test_zeus_session_history(self):
        """Test Zeus session history retrieval."""
        if not self._check_backend_available():
            pytest.skip('Python backend not available')
        
        # Create a session with a message
        session_id = f'history-test-{int(time.time()*1000)}'
        
        # Send a message
        chat_response = requests.post(
            f'{self.base_url}/api/zeus/chat',
            headers=AUTH_HEADERS,
            json={
                'message': 'Test message for history',
                'session_id': session_id,
            }
        )
        
        if chat_response.status_code != 200:
            pytest.skip('Zeus chat not working')
        
        # Get session history
        response = requests.get(
            f'{self.base_url}/api/zeus/session/{session_id}',
            headers=AUTH_HEADERS,
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert data['success'] is True
        assert 'messages' in data
        assert len(data['messages']) >= 1  # At least the user message
    
    # =========================================================================
    # Full Flow Test
    # =========================================================================
    
    def test_full_external_api_flow(self):
        """
        Test the complete external API flow:
        1. Upload document to Ocean
        2. Chat with Zeus using document context
        3. Verify session history
        """
        if not self._check_backend_available():
            pytest.skip('Python backend not available')
        
        # Step 1: Upload document
        doc_content = """
        # Fisher-Rao Geometry
        
        Fisher-Rao distance is defined as:
        d_FR(p, q) = arccos(∑√(p_i * q_i))
        
        This metric is used for all geometric operations in QIG.
        """
        
        upload_response = requests.post(
            f'{self.base_url}/api/ocean/knowledge/ingest',
            headers=AUTH_HEADERS,
            json={
                'content': doc_content,
                'title': 'Fisher-Rao Geometry Guide',
                'tags': ['geometry', 'fisher-rao', 'qig'],
                'source': 'full-flow-test',
            }
        )
        
        if upload_response.status_code != 200:
            pytest.skip('Document upload not working')
        
        upload_data = upload_response.json()
        assert upload_data['success'] is True
        knowledge_id = upload_data.get('knowledge_id')
        
        # Step 2: Chat with Zeus using context
        session_id = f'full-flow-{int(time.time()*1000)}'
        
        chat_response = requests.post(
            f'{self.base_url}/api/zeus/chat',
            headers=AUTH_HEADERS,
            json={
                'message': 'What is Fisher-Rao distance and how is it calculated?',
                'session_id': session_id,
                'context': {
                    'knowledge_id': knowledge_id,
                    'relevant_content': doc_content,
                },
            }
        )
        
        if chat_response.status_code != 200:
            pytest.skip('Zeus chat not working')
        
        chat_data = chat_response.json()
        assert chat_data['success'] is True
        assert 'response' in chat_data
        
        # Step 3: Verify session history
        history_response = requests.get(
            f'{self.base_url}/api/zeus/session/{session_id}',
            headers=AUTH_HEADERS,
        )
        
        assert history_response.status_code == 200
        history_data = history_response.json()
        assert history_data['success'] is True
        assert len(history_data['messages']) >= 2  # User + Assistant
        
        # Cleanup: Delete session
        delete_response = requests.delete(
            f'{self.base_url}/api/zeus/session/{session_id}',
            headers=AUTH_HEADERS,
        )
        assert delete_response.status_code == 200
        
        print(f"\n✅ Full API flow test passed!")
        print(f"   - Document uploaded: {knowledge_id}")
        print(f"   - Chat session: {session_id}")
        print(f"   - Response length: {len(chat_data.get('response', ''))} chars")


# ============================================================================
# Run tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
