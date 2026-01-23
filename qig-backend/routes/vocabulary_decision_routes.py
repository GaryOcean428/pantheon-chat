"""
Vocabulary Decision, Tracking, and Expansion API Routes

Exposes vocabulary learning decision-making, frequency tracking, and 
Fisher manifold-based expansion functionality.

Based on E8 Protocol v4.0 geometric vocabulary learning principles.
"""

from flask import Blueprint, jsonify, request
import logging

logger = logging.getLogger(__name__)

vocabulary_bp = Blueprint('vocabulary', __name__)


def _get_decision_module():
    """Lazy import vocabulary_decision module."""
    import vocabulary_decision
    return vocabulary_decision


def _get_tracker_module():
    """Lazy import vocabulary_tracker module."""
    import vocabulary_tracker
    return vocabulary_tracker


def _get_expander_module():
    """Lazy import vocabulary_expander module."""
    import vocabulary_expander
    return vocabulary_expander


# Global tracker and expander instances (lazy initialized)
_tracker_instance = None
_expander_instance = None


def _get_tracker():
    """Get or create global VocabularyTracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        tracker_module = _get_tracker_module()
        _tracker_instance = tracker_module.VocabularyTracker()
        logger.info("[VocabularyRoutes] Initialized VocabularyTracker")
    return _tracker_instance


def _get_expander():
    """Get or create global GeometricVocabularyExpander instance."""
    global _expander_instance
    if _expander_instance is None:
        expander_module = _get_expander_module()
        _expander_instance = expander_module.GeometricVocabularyExpander()
        logger.info("[VocabularyRoutes] Initialized GeometricVocabularyExpander")
    return _expander_instance


@vocabulary_bp.route('/decision/should-learn', methods=['POST'])
def should_learn_word():
    """
    Call should_gary_learn_word() to determine if word should be learned.
    
    Request: {"word", "frequency", "gary_state", "vocab_engine"}
    Response: {"success", "should_learn", "decision_score", "value_score", "stability_result", "entropy_score", "meta_gate", "reasoning"}
    """
    try:
        data = request.get_json() or {}
        
        decision_module = _get_decision_module()
        word = data.get('word', '')
        if not word:
            return jsonify({'success': False, 'error': 'Missing required field: word'}), 400
        
        # Parse gary_state and vocab_engine
        gary_state_data = data.get('gary_state', {})
        gary_state = decision_module.GaryState(
            basin_coordinates=gary_state_data.get('basin_coordinates', []),
            basin_reference=gary_state_data.get('basin_reference', []),
            phi=gary_state_data.get('phi', 0.0),
            meta=gary_state_data.get('meta', gary_state_data.get('M', 0.0)),  # Support both 'meta' and legacy 'M'
            regime=gary_state_data.get('regime', 'unknown')
        )
        
        observations_data = data.get('vocab_engine', {}).get('observations', {})
        
        # Minimal VocabConsolidationCycle mock
        class VocabEngineMock:
            def __init__(self, observations_dict):
                self.observations = {
                    word_key: decision_module.WordObservation(
                        word=obs_data.get('word', word_key),
                        contexts=[
                            decision_module.WordContext(
                                word=ctx.get('word', word_key),
                                phi=ctx.get('phi', 0.0),
                                kappa=ctx.get('kappa', 50.0),
                                regime=ctx.get('regime', 'unknown'),
                                basin_coordinates=ctx.get('basin_coordinates', []),
                                timestamp=ctx.get('timestamp', 0.0)
                            ) for ctx in obs_data.get('contexts', [])
                        ],
                        avg_phi=obs_data.get('avg_phi', 0.0),
                        max_phi=obs_data.get('max_phi', 0.0),
                        frequency=obs_data.get('frequency', 1),
                        first_seen=obs_data.get('first_seen', 0.0),
                        last_seen=obs_data.get('last_seen', 0.0),
                        context_basins=obs_data.get('context_basins', [])
                    ) for word_key, obs_data in observations_dict.items()
                }
            
            def get_or_create_observation(self, word: str):
                if word not in self.observations:
                    self.observations[word] = decision_module.WordObservation(word=word)
                return self.observations[word]
            
            def get_all_observations(self):
                return self.observations
        
        vocab_engine = VocabEngineMock(observations_data)
        frequency = data.get('frequency', 1)
        
        # Call should_gary_learn_word (async - run in event loop)
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        decision = loop.run_until_complete(
            decision_module.should_gary_learn_word(
                word=word,
                frequency=frequency,
                gary_state=gary_state,
                vocab_engine=vocab_engine
            )
        )
        
        return jsonify({
            'success': True,
            'should_learn': decision.should_learn,
            'decision_score': decision.score,  # VocabularyDecision uses 'score' field
            'value_score': {
                'efficiency': decision.value_score.efficiency,
                'phi_weight': decision.value_score.phi_weight,
                'connectivity': decision.value_score.connectivity,
                'compression': decision.value_score.compression,
                'total': decision.value_score.total
            },
            'stability_result': {
                'stable': decision.stability_result.stable,
                'drift': decision.stability_result.drift,
                'within_threshold': decision.stability_result.within_threshold,
                'acceptable': decision.stability_result.acceptable
            },
            'entropy_score': {
                'context_entropy': decision.entropy_score.context_entropy,
                'regime_entropy': decision.entropy_score.regime_entropy,
                'total': decision.entropy_score.total
            },
            'meta_gate': {
                'gate_open': decision.meta_gate.gate_open,
                'meta': decision.meta_gate.meta,
                'reasoning': decision.meta_gate.reasoning
            },
            'reasoning': decision.reasoning
        })
        
    except Exception as e:
        logger.error(f"Vocabulary decision error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@vocabulary_bp.route('/track/observe', methods=['POST'])
def track_observation():
    """
    Call VocabularyTracker.observe() to track word/phrase observation.
    
    Request: {"phrase", "phi", "kappa"?, "regime"?, "basin_coordinates"?}
    Response: {"success", "message"}
    """
    try:
        data = request.get_json() or {}
        
        phrase = data.get('phrase', '')
        if not phrase:
            return jsonify({'success': False, 'error': 'Missing required field: phrase'}), 400
        
        tracker = _get_tracker()
        tracker.observe(
            phrase=phrase,
            phi=data.get('phi', 0.0),
            kappa=data.get('kappa'),
            regime=data.get('regime'),
            basin_coordinates=data.get('basin_coordinates')
        )
        
        return jsonify({'success': True, 'message': f'Observation tracked for phrase: {phrase}'})
        
    except Exception as e:
        logger.error(f"Vocabulary tracking error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@vocabulary_bp.route('/track/candidates', methods=['POST'])
def get_candidates():
    """
    Call VocabularyTracker.get_candidates() to get expansion candidates.
    
    Request: {"top_k"?}
    Response: {"success", "candidates": [{"text", "type", "frequency", "avg_phi", "max_phi", "efficiency_gain", "reasoning", "is_real_word", "components"}], "count"}
    """
    try:
        data = request.get_json() or {}
        
        top_k = data.get('top_k', 20)
        
        tracker = _get_tracker()
        candidates = tracker.get_candidates(top_k=top_k)
        
        candidates_json = []
        for cand in candidates:
            candidates_json.append({
                'text': cand.text,
                'type': cand.type,
                'frequency': cand.frequency,
                'avg_phi': cand.avg_phi,
                'max_phi': cand.max_phi,
                'efficiency_gain': cand.efficiency_gain,
                'reasoning': cand.reasoning,
                'is_real_word': cand.is_real_word,
                'components': cand.components
            })
        
        return jsonify({
            'success': True,
            'candidates': candidates_json,
            'count': len(candidates_json)
        })
        
    except Exception as e:
        logger.error(f"Vocabulary candidates error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@vocabulary_bp.route('/expand/add-word', methods=['POST'])
def add_word():
    """
    Call VocabularyManifoldState.add_word() to add word to manifold.
    
    Request: {"text", "qig_score": {"phi", "kappa", "basin_coordinates", "regime"}, "components"?, "source"?}
    Response: {"success", "word": {"text", "coordinates", "phi", "kappa", "frequency", "components", "geodesic_origin"}}
    """
    try:
        data = request.get_json() or {}
        
        text = data.get('text', '')
        if not text:
            return jsonify({'success': False, 'error': 'Missing required field: text'}), 400
        
        qig_score_data = data.get('qig_score', {})
        expander_module = _get_expander_module()
        qig_score = expander_module.QIGScore(
            phi=qig_score_data.get('phi', 0.0),
            kappa=qig_score_data.get('kappa', 50.0),
            basin_coordinates=np.array(qig_score_data.get('basin_coordinates', [])),
            regime=qig_score_data.get('regime', 'arbitrary')
        )
        
        expander = _get_expander()
        word = expander.add_word(
            text=text,
            qig_score=qig_score,
            components=data.get('components'),
            source=data.get('source', 'Direct observation')
        )
        
        return jsonify({
            'success': True,
            'word': {
                'text': word.text,
                'coordinates': word.coordinates.tolist() if hasattr(word.coordinates, 'tolist') else list(word.coordinates),
                'phi': word.phi,
                'kappa': word.kappa,
                'frequency': word.frequency,
                'components': word.components,
                'geodesic_origin': word.geodesic_origin
            }
        })
        
    except Exception as e:
        logger.error(f"Add word error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@vocabulary_bp.route('/expand/nearby', methods=['POST'])
def find_nearby_words():
    """
    Call VocabularyManifoldState.find_nearby_words() to find similar words.
    
    Request: {"coordinates": [...], "max_distance"?}
    Response: {"success", "nearby_words": [{"text", "coordinates", "phi", "kappa", "frequency", "components", "geodesic_origin"}], "count"}
    """
    try:
        data = request.get_json() or {}
        
        coordinates_data = data.get('coordinates', [])
        if not coordinates_data:
            return jsonify({'success': False, 'error': 'Missing required field: coordinates'}), 400
        
        coordinates = np.array(coordinates_data)
        expander = _get_expander()
        nearby_words = expander.find_nearby_words(
            coordinates=coordinates,
            max_distance=data.get('max_distance', 2.0)
        )
        
        words_json = []
        for word in nearby_words:
            words_json.append({
                'text': word.text,
                'coordinates': word.coordinates.tolist() if hasattr(word.coordinates, 'tolist') else list(word.coordinates),
                'phi': word.phi,
                'kappa': word.kappa,
                'frequency': word.frequency,
                'components': word.components,
                'geodesic_origin': word.geodesic_origin
            })
        
        return jsonify({
            'success': True,
            'nearby_words': words_json,
            'count': len(words_json)
        })
        
    except Exception as e:
        logger.error(f"Find nearby words error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@vocabulary_bp.route('/expand/hypotheses', methods=['POST'])
def generate_hypotheses():
    """
    Call VocabularyManifoldState.generate_manifold_hypotheses().
    
    Request: {"count"?}
    Response: {"success", "hypotheses": [...], "count"}
    """
    try:
        data = request.get_json() or {}
        
        count = data.get('count', 20)
        
        expander = _get_expander()
        hypotheses = expander.generate_manifold_hypotheses(count=count)
        
        return jsonify({
            'success': True,
            'hypotheses': hypotheses,
            'count': len(hypotheses)
        })
        
    except Exception as e:
        logger.error(f"Generate hypotheses error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def register_vocabulary_routes(app):
    """Register vocabulary routes with Flask app."""
    app.register_blueprint(vocabulary_bp, url_prefix='/api/vocabulary')
    logger.info("[INFO] Registered vocabulary_bp at /api/vocabulary")
