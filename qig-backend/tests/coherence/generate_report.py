"""
HTML Report Generator for Coherence Tests
==========================================

Generates comprehensive HTML reports with:
- Summary tables and charts
- Side-by-side metric comparisons
- Statistical analysis visualizations
- Pass/fail criteria evaluation

Author: WP4.3 Coherence Harness
Date: 2026-01-20
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from test_helpers import COHERENCE_DIR
from compare_architectures import ArchitectureComparison

logger = logging.getLogger(__name__)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QIG Coherence Test Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            margin: 0 0 10px 0;
        }}
        
        .header .subtitle {{
            opacity: 0.9;
            font-size: 14px;
        }}
        
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .section h2 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        .metric-value {{
            font-weight: 600;
            font-family: 'Courier New', monospace;
        }}
        
        .improvement-positive {{
            color: #28a745;
        }}
        
        .improvement-negative {{
            color: #dc3545;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }}
        
        .badge-success {{
            background: #d4edda;
            color: #155724;
        }}
        
        .badge-warning {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .badge-danger {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .badge-info {{
            background: #d1ecf1;
            color: #0c5460;
        }}
        
        .verdict-box {{
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid;
        }}
        
        .verdict-confirmed {{
            background: #d4edda;
            border-color: #28a745;
        }}
        
        .verdict-rejected {{
            background: #f8d7da;
            border-color: #dc3545;
        }}
        
        .verdict-uncertain {{
            background: #fff3cd;
            border-color: #ffc107;
        }}
        
        .config-card {{
            border: 2px solid #e9ecef;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }}
        
        .config-card.best {{
            border-color: #28a745;
            background: #f8fff9;
        }}
        
        .config-card.baseline {{
            border-color: #6c757d;
            background: #f8f9fa;
        }}
        
        .metric-bar {{
            height: 30px;
            background: #667eea;
            border-radius: 4px;
            transition: width 0.3s;
        }}
        
        .metric-bar-container {{
            width: 100%;
            background: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .footer {{
            text-align: center;
            padding: 20px;
            color: #6c757d;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🌊 QIG Coherence Test Report</h1>
        <div class="subtitle">
            Generated: {timestamp}<br>
            Test Suite: Plan→Realize→Repair Architecture Evaluation<br>
            Protocol: Ultra Consciousness v4.0 ACTIVE
        </div>
    </div>
    
    {content}
    
    <div class="footer">
        WP4.3: Reproducible Coherence Test Harness<br>
        QIG Purity Mode: Enforced | E8 Protocol v4.0
    </div>
</body>
</html>
"""


def generate_verdict_section(verdict: Dict[str, Any]) -> str:
    """Generate HTML for verdict section."""
    verdict_class = {
        'HYPOTHESIS CONFIRMED': 'verdict-confirmed',
        'HYPOTHESIS SUPPORTED': 'verdict-confirmed',
        'HYPOTHESIS UNCERTAIN': 'verdict-uncertain',
        'HYPOTHESIS REJECTED': 'verdict-rejected',
    }.get(verdict['overall'], 'verdict-uncertain')
    
    html = f"""
    <div class="section">
        <h2>🎯 Final Verdict</h2>
        <div class="verdict-box {verdict_class}">
            <h3 style="margin-top: 0;">{verdict['overall']}</h3>
            <p><strong>Hypothesis:</strong> {verdict['hypothesis']}</p>
            <p><strong>Win Rate:</strong> {verdict['win_rate']:.1%} ({len(verdict['wins'])}/{len(['phi', 'alignment', 'smoothness'])} metrics)</p>
            <p><strong>Significant Wins:</strong> {verdict['significant_win_rate']:.1%}</p>
            <p><strong>Recommendation:</strong> {verdict['recommendation']}</p>
        </div>
    </div>
    """
    
    return html


def generate_summary_table(metrics: Dict[str, Any]) -> str:
    """Generate HTML summary table."""
    configs = ['skeleton_only', 'pure_geometric', 'plan_realize_repair']
    config_labels = {
        'skeleton_only': 'Skeleton (Baseline)',
        'pure_geometric': 'Pure Geometric',
        'plan_realize_repair': 'Plan→Realize→Repair ⭐',
    }
    
    metric_names = ['phi', 'kappa', 'alignment', 'smoothness', 'recursive_depth']
    metric_labels = {
        'phi': 'Φ (Integration)',
        'kappa': 'κ (Coupling)',
        'alignment': 'Waypoint Alignment',
        'smoothness': 'Trajectory Smoothness',
        'recursive_depth': 'Recursive Depth',
    }
    
    html = """
    <div class="section">
        <h2>📊 Configuration Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Skeleton (Baseline)</th>
                    <th>Pure Geometric</th>
                    <th>Plan→Realize→Repair ⭐</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for metric_key in metric_names:
        metric_label = metric_labels.get(metric_key, metric_key)
        html += f"<tr><td><strong>{metric_label}</strong></td>"
        
        for config in configs:
            if config in metrics and metric_key in metrics[config]:
                value = metrics[config][metric_key]['mean']
                std = metrics[config][metric_key]['std']
                html += f'<td class="metric-value">{value:.3f} ± {std:.3f}</td>'
            else:
                html += '<td>N/A</td>'
        
        html += "</tr>"
    
    html += """
            </tbody>
        </table>
    </div>
    """
    
    return html


def generate_comparison_section(comparisons: Dict[str, Any]) -> str:
    """Generate HTML for comparison section."""
    html = """
    <div class="section">
        <h2>🔬 Statistical Comparisons</h2>
        <h3>Critical: Plan→Realize→Repair vs Skeleton-Only</h3>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Full Arch</th>
                    <th>Baseline</th>
                    <th>Improvement</th>
                    <th>P-value</th>
                    <th>Significant</th>
                    <th>Effect Size</th>
                </tr>
            </thead>
            <tbody>
    """
    
    metrics = ['phi', 'alignment', 'smoothness', 'kappa']
    
    for metric in metrics:
        comp_key = f'full_vs_baseline_{metric}'
        if comp_key not in comparisons:
            continue
        
        comp = comparisons[comp_key]
        
        improvement_class = 'improvement-positive' if comp['improvement'] > 0 else 'improvement-negative'
        sig_badge = 'badge-success' if comp['is_significant'] else 'badge-warning'
        
        html += f"""
        <tr>
            <td><strong>{metric.upper()}</strong></td>
            <td class="metric-value">{comp['mean_a']:.3f}</td>
            <td class="metric-value">{comp['mean_b']:.3f}</td>
            <td class="{improvement_class} metric-value">{comp['improvement']:+.3f} ({comp['improvement_pct']:+.1f}%)</td>
            <td class="metric-value">{comp['p_value']:.4f}</td>
            <td><span class="badge {sig_badge}">{'YES ✓' if comp['is_significant'] else 'NO'}</span></td>
            <td><span class="badge badge-info">{comp['effect_size']}</span></td>
        </tr>
        """
    
    html += """
            </tbody>
        </table>
    </div>
    """
    
    return html


def generate_visualization_section(metrics: Dict[str, Any]) -> str:
    """Generate HTML visualizations (bar charts)."""
    html = """
    <div class="section">
        <h2>📈 Metric Visualizations</h2>
    """
    
    metrics_to_plot = ['phi', 'alignment', 'smoothness']
    metric_labels = {
        'phi': 'Φ (Integration)',
        'alignment': 'Waypoint Alignment',
        'smoothness': 'Trajectory Smoothness',
    }
    
    configs = ['skeleton_only', 'pure_geometric', 'plan_realize_repair']
    config_labels = {
        'skeleton_only': 'Skeleton',
        'pure_geometric': 'Pure Geo',
        'plan_realize_repair': 'Full Arch ⭐',
    }
    
    for metric_key in metrics_to_plot:
        metric_label = metric_labels.get(metric_key, metric_key)
        
        html += f"<h3>{metric_label}</h3>"
        
        # Find max value for scaling
        max_value = 0
        for config in configs:
            if config in metrics and metric_key in metrics[config]:
                max_value = max(max_value, metrics[config][metric_key]['mean'])
        
        # Generate bars
        for config in configs:
            if config in metrics and metric_key in metrics[config]:
                value = metrics[config][metric_key]['mean']
                width_pct = (value / max_value * 100) if max_value > 0 else 0
                
                html += f"""
                <div style="margin: 10px 0;">
                    <div style="display: flex; align-items: center;">
                        <div style="width: 150px;"><strong>{config_labels[config]}</strong></div>
                        <div style="flex: 1;">
                            <div class="metric-bar-container">
                                <div class="metric-bar" style="width: {width_pct}%;"></div>
                            </div>
                        </div>
                        <div style="width: 80px; text-align: right;" class="metric-value">{value:.3f}</div>
                    </div>
                </div>
                """
    
    html += "</div>"
    
    return html


def generate_html_report(comparison: ArchitectureComparison) -> str:
    """
    Generate complete HTML report.
    
    Args:
        comparison: ArchitectureComparison with results
        
    Returns:
        HTML string
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    # Generate sections
    verdict_html = generate_verdict_section(comparison.generate_verdict())
    summary_html = generate_summary_table(comparison.metrics)
    comparison_html = generate_comparison_section(comparison.comparisons)
    visualization_html = generate_visualization_section(comparison.metrics)
    
    # Combine all sections
    content = verdict_html + summary_html + comparison_html + visualization_html
    
    # Fill template
    html = HTML_TEMPLATE.format(
        timestamp=timestamp,
        content=content
    )
    
    return html


def save_html_report(
    comparison: ArchitectureComparison,
    filename: str = 'coherence_report.html'
) -> Path:
    """
    Save HTML report to file.
    
    Args:
        comparison: ArchitectureComparison with results
        filename: Output filename
        
    Returns:
        Path to saved file
    """
    output_dir = COHERENCE_DIR / "results"
    output_dir.mkdir(exist_ok=True)
    filepath = output_dir / filename
    
    html = generate_html_report(comparison)
    
    with open(filepath, 'w') as f:
        f.write(html)
    
    logger.info(f"HTML report saved to: {filepath}")
    
    return filepath


def main():
    """Main entry point for report generation."""
    from compare_architectures import main as run_comparison
    
    logger.info("Generating HTML Report...")
    
    # Run comparison to get results
    comparison = run_comparison()
    
    # Generate and save HTML report
    report_path = save_html_report(comparison)
    
    logger.info(f"\n✅ HTML report generated: {report_path}")
    logger.info(f"Open in browser: file://{report_path.absolute()}")
    
    return report_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    main()
