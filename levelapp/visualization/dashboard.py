"""levelapp/visualization/dashboard.py: Dashboard generation for evaluation results."""

from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape

from levelapp.simulator.schemas import SimulationResults
from levelapp.visualization.charts import ChartGenerator
from levelapp.aspects import logger


class DashboardGenerator:
    """Generate comprehensive HTML dashboards for evaluation results."""

    def __init__(self, template_dir: str | None = None):
        """
        Initialize DashboardGenerator.

        Args:
            template_dir: Optional custom template directory path
        """
        if template_dir is None:
            # Use default templates directory
            template_dir = Path(__file__).parent / "templates"

        self.template_dir = Path(template_dir)
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )
        self.chart_gen = ChartGenerator()

    def generate_simulator_dashboard(
        self,
        results: SimulationResults,
        output_path: str,
        title: str = "Evaluation Dashboard",
    ) -> str:
        """
        Generate a complete HTML dashboard for simulator results.

        Args:
            results: SimulationResults object
            output_path: Path to save the HTML file
            title: Dashboard title

        Returns:
            Path to the generated HTML file
        """
        logger.info(
            f"[DashboardGenerator] Generating simulator dashboard: {output_path}"
        )

        # Generate all charts
        charts = {
            "score_trend": self.chart_gen.create_score_trend(results),
            "provider_comparison": self.chart_gen.create_provider_comparison(results),
            "summary_metrics": self.chart_gen.create_summary_metrics(results),
        }

        # Add distribution charts for each provider
        if results.average_scores:
            for provider in results.average_scores.keys():
                if provider not in ["processing_time", "guardrail", "metadata"]:
                    try:
                        charts[f"distribution_{provider}"] = (
                            self.chart_gen.create_score_distribution(results, provider)
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to create distribution chart for {provider}: {e}"
                        )

        # Add timeline chart
        try:
            charts["timeline"] = self.chart_gen.create_interaction_timeline(results)
        except Exception as e:
            logger.warning(f"Failed to create timeline chart: {e}")

        # Convert charts to HTML
        chart_htmls = {}
        for name, fig in charts.items():
            chart_htmls[name] = fig.to_html(
                include_plotlyjs="cdn",
                div_id=f"chart_{name}",
                config={"responsive": True},
            )

        # Calculate summary statistics
        summary_stats = self._create_summary_stats(results)

        # Prepare context for template
        context = {
            "title": title,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary_stats": summary_stats,
            "charts": chart_htmls,
            "results": results,
            "has_evaluation_summary": bool(results.evaluation_summary),
        }

        # Render template
        html_content = self._render_template("simulator_dashboard.html", context)

        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(html_content, encoding="utf-8")

        logger.info(f"[DashboardGenerator] Dashboard saved to: {output_path}")
        return str(output_file.absolute())

    def generate_comparator_dashboard(
        self,
        results: Dict[str, Any],
        output_path: str,
        title: str = "Comparison Dashboard",
    ) -> str:
        """
        Generate a complete HTML dashboard for comparator results.

        Args:
            results: Comparator results dictionary
            output_path: Path to save the HTML file
            title: Dashboard title

        Returns:
            Path to the generated HTML file
        """
        logger.info(
            f"[DashboardGenerator] Generating comparator dashboard: {output_path}"
        )

        # Generate heatmap
        heatmap = self.chart_gen.create_metadata_heatmap(results)

        # Convert to HTML
        chart_html = heatmap.to_html(
            include_plotlyjs="cdn", div_id="chart_heatmap", config={"responsive": True}
        )

        # Prepare context
        context = {
            "title": title,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "chart_html": chart_html,
            "results": results,
        }

        # Render template
        html_content = self._render_template("comparator_dashboard.html", context)

        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(html_content, encoding="utf-8")

        logger.info(f"[DashboardGenerator] Dashboard saved to: {output_path}")
        return str(output_file.absolute())

    def _create_summary_stats(self, results: SimulationResults) -> Dict[str, Any]:
        """
        Extract key summary statistics from results.

        Args:
            results: SimulationResults object

        Returns:
            Dictionary of summary statistics
        """
        stats = {
            "total_scripts": len(results.interaction_results or []),
            "total_time": results.elapsed_time,
            "started_at": results.started_at.strftime("%Y-%m-%d %H:%M:%S"),
            "finished_at": results.finished_at.strftime("%Y-%m-%d %H:%M:%S"),
            "average_scores": results.average_scores or {},
            "providers": list(results.average_scores.keys())
            if results.average_scores
            else [],
        }

        # Calculate overall average (excluding non-score metrics)
        score_values = [
            v
            for k, v in (results.average_scores or {}).items()
            if k not in ["processing_time", "guardrail", "metadata"]
        ]
        stats["overall_average"] = (
            sum(score_values) / len(score_values) if score_values else 0.0
        )

        return stats

    def _render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """
        Render a Jinja2 template with the given context.

        Args:
            template_name: Name of the template file
            context: Template context dictionary

        Returns:
            Rendered HTML string
        """
        try:
            template = self.env.get_template(template_name)
            return template.render(**context)
        except Exception as e:
            logger.error(f"[DashboardGenerator] Template rendering failed: {e}")
            # Return a basic HTML fallback
            return self._create_fallback_html(context)

    def _create_fallback_html(self, context: Dict[str, Any]) -> str:
        """Create a basic HTML fallback if template rendering fails."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>{context.get("title", "Evaluation Dashboard")}</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #333; }}
        .chart {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{context.get("title", "Evaluation Dashboard")}</h1>
        <p>Generated at: {context.get("generated_at", "N/A")}</p>
        <div class="charts">
            {"".join(context.get("charts", {}).values())}
        </div>
    </div>
</body>
</html>
"""
