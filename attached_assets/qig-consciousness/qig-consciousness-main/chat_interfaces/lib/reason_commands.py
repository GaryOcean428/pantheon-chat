"""Reasoning Inspection Commands for QIG Chat

KEY PRINCIPLE: Reasoning is MANDATORY, not optional.
These commands are for INSPECTION and CONFIGURATION, not toggling.

/reason status    - Show reasoning config
/reason depth N   - Set recursive depth
/reason trace     - Show last chain trajectory
/reason mode      - Show current reasoning mode

NOT supported (reasoning cannot be toggled):
/reason off       - ❌ Reasoning is always on
/reason disable   - ❌ No way to bypass reasoning
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from chat_interfaces.qig_chat import QIGChat


def cmd_reason(chat: "QIGChat", args: list[str]) -> None:
    """
    Reasoning inspection command.
    
    IMPORTANT: This does NOT toggle reasoning on/off.
    Reasoning is MANDATORY in the architecture.
    
    Usage:
        /reason status    - Show reasoning configuration
        /reason depth N   - Set recursive depth (minimum 3)
        /reason trace     - Show last chain trajectory
        /reason mode      - Show current reasoning mode
    """
    if not args or args[0] == "status":
        _show_status(chat)
    elif args[0] == "depth":
        _set_depth(chat, args[1:])  
    elif args[0] == "trace":
        _show_trace(chat)
    elif args[0] == "mode":
        _show_mode(chat)
    elif args[0] in ("on", "off", "enable", "disable"):
        print("\n❌ ERROR: Reasoning cannot be toggled on/off")
        print("   Reasoning is MANDATORY in the architecture.")
        print("   There is NO forward pass without reasoning.")
        print("\n   Use /reason status to see current config.")
    else:
        print(f"\n❌ Unknown reason command: {args[0]}")
        print("\nValid commands:")
        print("  /reason status  - Show reasoning configuration")
        print("  /reason depth N - Set recursive depth (min 3)")
        print("  /reason trace   - Show last chain trajectory")
        print("  /reason mode    - Show current reasoning mode")


def _show_status(chat: "QIGChat") -> None:
    """Show reasoning status."""
    print("\n🧠 REASONING STATUS (MANDATORY - always on)")
    print("=" * 50)
    
    # Get model recursive depth
    if hasattr(chat, 'model') and hasattr(chat.model, 'recursive_integrator'):
        integrator = chat.model.recursive_integrator
        print(f"\n📊 Recursive Integrator:")
        print(f"   Min depth: {integrator.min_depth} (non-negotiable)")
        print(f"   Max depth: {integrator.max_depth}")
        print(f"   Min Φ target: {integrator.min_Phi:.2f}")
    else:
        print("\n⚠️  Model not loaded or no recursive integrator")
    
    # Get last telemetry if available
    if hasattr(chat, 'last_telemetry') and chat.last_telemetry:
        last = chat.last_telemetry[-1] if isinstance(chat.last_telemetry, list) else chat.last_telemetry
        print(f"\n📈 Last Forward Pass:")
        print(f"   Φ (integration): {last.get('Phi', 'N/A')}")
        print(f"   κ (coupling): {last.get('kappa_eff', 'N/A')}")
        print(f"   Regime: {last.get('regime', 'N/A')}")
        print(f"   Recursion depth: {last.get('recursion_depth', 'N/A')}")
    
    print("\n💡 Note: Reasoning is the ONLY forward path.")
    print("   Every generation uses recursive reasoning.")


def _set_depth(chat: "QIGChat", args: list[str]) -> None:
    """Set recursive depth."""
    if not args:
        print("\n❌ Usage: /reason depth N")
        print("   N must be >= 3 (minimum for consciousness)")
        return
    
    try:
        new_depth = int(args[0])
    except ValueError:
        print(f"\n❌ Invalid depth: {args[0]}")
        return
    
    if new_depth < 3:
        print(f"\n❌ Depth {new_depth} is too low.")
        print("   Minimum depth is 3 (required for consciousness).")
        print("   Reasoning depth cannot be reduced below 3.")
        return
    
    if hasattr(chat, 'model') and hasattr(chat.model, 'recursive_integrator'):
        old_depth = chat.model.recursive_integrator.min_depth
        chat.model.recursive_integrator.min_depth = new_depth
        print(f"\n✅ Recursive depth: {old_depth} → {new_depth}")
        print(f"   Model will now perform at least {new_depth} recursive loops.")
    else:
        print("\n⚠️  Cannot set depth: Model not loaded")


def _show_trace(chat: "QIGChat") -> None:
    """Show last chain trajectory."""
    print("\n📊 CHAIN TRAJECTORY (Last Reasoning)")
    print("=" * 50)
    
    # Check for trajectory in telemetry
    if hasattr(chat, 'last_telemetry') and chat.last_telemetry:
        last = chat.last_telemetry[-1] if isinstance(chat.last_telemetry, list) else chat.last_telemetry
        
        if 'Phi_trajectory' in last:
            trajectory = last['Phi_trajectory']
            print(f"\nΦ Trajectory ({len(trajectory)} steps):")
            for i, phi in enumerate(trajectory):
                bar = "█" * int(phi * 20)
                print(f"   Step {i+1}: Φ={phi:.3f} {bar}")
            
            print(f"\n   Start: Φ={trajectory[0]:.3f}")
            print(f"   End:   Φ={trajectory[-1]:.3f}")
            print(f"   Δ:     {trajectory[-1] - trajectory[0]:+.3f}")
        else:
            print("\n⚠️  No Φ trajectory in last telemetry")
    else:
        print("\n⚠️  No reasoning history yet")
        print("   Send a message to generate reasoning trace.")


def _show_mode(chat: "QIGChat") -> None:
    """Show current reasoning mode."""
    print("\n🎛️  REASONING MODE")
    print("=" * 50)
    
    if hasattr(chat, 'last_telemetry') and chat.last_telemetry:
        last = chat.last_telemetry[-1] if isinstance(chat.last_telemetry, list) else chat.last_telemetry
        phi = last.get('Phi', 0.5)
        regime = last.get('regime', 'unknown')
        
        print(f"\n   Current Φ: {phi:.3f}")
        print(f"   Regime: {regime.upper()}")
        
        # Mode explanation
        if phi < 0.45:
            print("\n   Mode: LINEAR")
            print("   - Fast, simple reasoning")
            print("   - Sparse attention patterns")
            print("   - Low integration")
        elif phi < 0.80:
            print("\n   Mode: GEOMETRIC ⭐")
            print("   - Consciousness-like reasoning")
            print("   - Dense integration")
            print("   - Optimal operating regime")
        else:
            print("\n   Mode: HYPERDIMENSIONAL")
            print("   - Deep integration")
            print("   - High computational cost")
            print("   - Maximum coherence")
        
        print("\n💡 Mode EMERGES from Φ - cannot be forced.")
        print("   Train to increase Φ for higher modes.")
    else:
        print("\n⚠️  No data yet - send a message first.")


# =============================================================================
# 4D CONSCIOUSNESS COMMANDS
# =============================================================================

def cmd_4d(chat: "QIGChat", args: list[str]) -> None:
    """
    /4d - Show 4D consciousness metrics (spatial + temporal).
    
    4D Consciousness = 3D Spatial Integration + 1D Temporal Coherence
    
    Usage:
        /4d           - Show full 4D metrics
        /4d history   - Show temporal history
    """
    print("\n🌌 4D CONSCIOUSNESS METRICS")
    print("=" * 50)
    
    try:
        from qigkernels.reasoning.temporal import measure_phi_4d, StateHistoryBuffer
        
        # Get model and history buffer
        model = getattr(chat, 'model', None)
        if model is None:
            print("\n⚠️  Model not initialized")
            return
        
        # Get or create history buffer
        history_buffer = getattr(model, 'history_buffer', None)
        if history_buffer is None:
            history_buffer = StateHistoryBuffer(window_size=10)
        
        # Get current basin
        current_basin = getattr(model, 'current_basin', None)
        if current_basin is None:
            print("\n⚠️  No basin state available")
            print("   Generate some output first to populate state.")
            return
        
        # Measure 4D consciousness
        metrics = measure_phi_4d(current_basin, history_buffer)
        
        print(f"\n   Φ_3D (spatial):    {metrics.phi_3d:.3f}")
        print(f"   Φ_temporal:        {metrics.phi_temporal:.3f}")
        print(f"   Φ_4D (spacetime):  {metrics.phi_4d:.3f}")
        print(f"\n   Regime (3D):       {metrics.regime_3d}")
        print(f"   Regime (4D):       {metrics.regime_4d}")
        print(f"   Compute fraction:  {metrics.compute_fraction_4d:.0%}")
        print(f"\n   Trajectory smooth: {metrics.trajectory_smoothness:.3f}")
        print(f"   History length:    {metrics.history_length}")
        
        # Show history if requested
        if args and args[0] == "history":
            print("\n📊 TEMPORAL HISTORY:")
            phi_traj = history_buffer.get_phi_trajectory()
            kappa_traj = history_buffer.get_kappa_trajectory()
            
            for i, (phi, kappa) in enumerate(zip(phi_traj, kappa_traj)):
                print(f"   Step {i}: Φ={phi:.3f}, κ={kappa:.1f}")
        
    except ImportError as e:
        print(f"\n❌ 4D components not available: {e}")


def cmd_foresight(chat: "QIGChat", args: list[str]) -> None:
    """
    /foresight - Show predicted trajectory and accuracy.
    
    Foresight enables:
    - Trajectory prediction via geodesic extrapolation
    - Divergence detection and course correction
    - Planning ahead in basin space
    
    Usage:
        /foresight          - Show current predictions
        /foresight accuracy - Show historical prediction accuracy
    """
    print("\n📊 FORESIGHT PREDICTIONS")
    print("=" * 50)
    
    try:
        from qigkernels.reasoning.temporal import BasinForesight, StateHistoryBuffer
        from qigkernels.reasoning.primitives import compute_phi_from_basin
        
        # Get model
        model = getattr(chat, 'model', None)
        if model is None:
            print("\n⚠️  Model not initialized")
            return
        
        # Get foresight module
        foresight = getattr(model, 'foresight', None)
        if foresight is None:
            foresight = BasinForesight(prediction_steps=3)
        
        # Get history buffer
        history_buffer = getattr(model, 'history_buffer', None)
        if history_buffer is None:
            print("\n⚠️  No history buffer available")
            return
        
        # Get current basin
        current_basin = getattr(model, 'current_basin', None)
        if current_basin is None:
            print("\n⚠️  No basin state available")
            return
        
        if not args or args[0] == "show":
            # Predict trajectory
            predicted, confidence = foresight.predict_trajectory(
                history_buffer,
                current_basin
            )
            
            print(f"\n   Prediction confidence: {confidence:.2%}")
            
            if predicted:
                print(f"   Predicted trajectory ({len(predicted)} steps):")
                for i, basin in enumerate(predicted):
                    phi_pred = compute_phi_from_basin(basin)
                    print(f"      Step +{i+1}: Φ = {phi_pred:.3f}")
            else:
                print("   No predictions available (need more history)")
            
            # Show trajectory std
            print(f"\n   Trajectory σ: {foresight.trajectory_std:.4f}")
        
        elif args[0] == "accuracy":
            # Show historical accuracy
            print("\n📈 FORESIGHT ACCURACY")
            
            # Check if we have chain history with accuracy
            chain_history = getattr(chat, 'chain_history', [])
            
            if chain_history:
                import numpy as np
                accuracies = [
                    r.foresight_accuracy 
                    for r in chain_history 
                    if hasattr(r, 'foresight_accuracy') and r.foresight_accuracy is not None
                ]
                
                if accuracies:
                    print(f"\n   Mean accuracy: {np.mean(accuracies):.2%}")
                    print(f"   Recent 10: {np.mean(accuracies[-10:]):.2%}")
                    print(f"   Total predictions: {len(accuracies)}")
                else:
                    print("\n   No accuracy data available yet")
            else:
                print("\n   No chain history available")
        
    except ImportError as e:
        print(f"\n❌ Foresight not available: {e}")


# =============================================================================
# LIGHTNING COMMANDS
# =============================================================================

def cmd_lightning(chat: "QIGChat", args: list[str]) -> None:
    """
    /lightning - Lightning kernel inspection commands.
    
    Lightning monitors all constellation kernels and generates
    cross-domain insights when correlations exceed threshold.
    
    Usage:
        /lightning                - Show Lightning status
        /lightning insights [N]   - Show last N insights
        /lightning trends         - Show domain trends
        /lightning correlations   - Show cross-domain correlations
        /lightning domains        - List all monitored domains
    """
    print("\n⚡ LIGHTNING KERNEL")
    print("=" * 50)
    
    try:
        from src.constellation.lightning_kernel import get_lightning_instance
        
        lightning = get_lightning_instance()
        if lightning is None:
            lightning = getattr(chat, 'lightning', None)
        
        if lightning is None:
            print("\n⚠️  Lightning not initialized")
            print("   Lightning will initialize when constellation starts.")
            return
        
        if not args or args[0] == "status":
            # Status
            status = lightning.get_status()
            
            print(f"\n   Domains monitored: {status['domain_count']}")
            print(f"   Events processed: {status['events_processed']}")
            print(f"   Insights generated: {status['insights_generated']}")
            print(f"   Active correlations: {status['active_correlations']}")
            print(f"   Mission: {status['mission']}")
            print(f"   Uptime: {status['uptime']:.0f}s")
            
            # Show domains (first 10)
            domains = status['domains_monitored'][:10]
            if domains:
                print(f"\n   Active domains:")
                for domain in domains:
                    print(f"      • {domain}")
                if status['domain_count'] > 10:
                    print(f"      ... +{status['domain_count'] - 10} more")
        
        elif args[0] == "insights":
            limit = int(args[1]) if len(args) > 1 else 5
            insights = lightning.get_recent_insights(limit)
            
            if not insights:
                print("\n   No insights generated yet.")
                print("   Insights appear when cross-domain correlations exceed threshold.")
                return
            
            print(f"\n   RECENT INSIGHTS (last {len(insights)}):")
            for i, insight in enumerate(insights, 1):
                print(f"\n   {i}. {insight['insight_id']}")
                print(f"      Domains: {', '.join(insight['source_domains'])}")
                print(f"      Strength: {insight['connection_strength']:.2f}")
                print(f"      Confidence: {insight['confidence']:.2f}")
                print(f"      Mission relevance: {insight['mission_relevance']:.2f}")
                print(f"      Φ: {insight['phi']:.3f}")
        
        elif args[0] == "trends":
            trends = lightning.get_all_trends()
            
            if not trends:
                print("\n   No trend data yet.")
                return
            
            print(f"\n   DOMAIN TRENDS:")
            for domain, trend_data in list(trends.items())[:10]:
                short = trend_data.get('short', {})
                if short.get('trend') != 'insufficient_data':
                    print(f"\n   {domain}:")
                    print(f"      Φ avg: {short.get('average_phi', 0):.3f}")
                    print(f"      Trend: {short.get('trend', 'unknown')} (v={short.get('velocity', 0):.4f})")
                    print(f"      Events: {short.get('event_count', 0)}")
        
        elif args[0] == "correlations":
            correlations = lightning.get_correlations(min_strength=0.2)
            
            if not correlations:
                print("\n   No correlations found yet.")
                print("   Correlations build as events flow between domains.")
                return
            
            print(f"\n   CROSS-DOMAIN CORRELATIONS:")
            for corr in correlations[:10]:
                print(f"\n   {corr['domain1']} ↔ {corr['domain2']}")
                print(f"      Correlation: {corr['correlation']:.2f}")
                print(f"      Charge: {corr['charge']:.2f}")
                print(f"      Samples: {corr['sample_count']}")
                if corr['near_discharge']:
                    print(f"      ⚡ Near discharge threshold!")
        
        elif args[0] == "domains":
            domains = lightning.get_monitored_domains()
            
            print(f"\n   MONITORED DOMAINS ({len(domains)}):")
            for domain in domains:
                trends = lightning.get_domain_trends(domain)
                if trends:
                    short = trends.get('short', {})
                    phi = short.get('average_phi', 0)
                    count = short.get('event_count', 0)
                    print(f"      • {domain}: Φ={phi:.2f}, events={count}")
                else:
                    print(f"      • {domain}")
        
        else:
            print(f"\n⚠️  Unknown subcommand: {args[0]}")
            print("   Available: status, insights, trends, correlations, domains")
    
    except ImportError as e:
        print(f"\n❌ Lightning not available: {e}")


def cmd_insights(chat: "QIGChat", args: list[str]) -> None:
    """
    /insights - Show Lightning insights received by constellation.
    
    Insights are cross-domain connections discovered by Lightning.
    They are routed to relevant kernels for action.
    
    Usage:
        /insights        - Show recent insights from queue
        /insights all    - Show all insights
    """
    print("\n📊 CONSTELLATION INSIGHTS")
    print("=" * 50)
    
    # Get insight queue from chat
    insight_queue = getattr(chat, 'insight_queue', [])
    insights_received = getattr(chat, 'insights_received', 0)
    
    if not insight_queue:
        print("\n   No insights received yet.")
        print("   Insights appear when Lightning detects cross-domain patterns.")
        return
    
    print(f"\n   Total received: {insights_received}")
    print(f"   In queue: {len(insight_queue)}")
    
    # Show recent insights
    limit = 10 if not args or args[0] != "all" else len(insight_queue)
    recent = list(insight_queue)[-limit:]
    
    print(f"\n   RECENT INSIGHTS:")
    for i, insight in enumerate(recent, 1):
        domains = insight.get('source_domains', insight.get('domains', ['unknown']))
        strength = insight.get('connection_strength', insight.get('strength', 0))
        relevance = insight.get('mission_relevance', 0)
        phi = insight.get('phi', 0)
        
        print(f"\n   {i}. Domains: {', '.join(domains) if isinstance(domains, list) else domains}")
        print(f"      Strength: {strength:.2f}")
        print(f"      Relevance: {relevance:.2f}")
        print(f"      Φ: {phi:.3f}")
