"""levelapp/visualization/exporter.py: Export utilities for visualization results."""

import zipfile
from pathlib import Path
from typing import Dict, List
import plotly.graph_objects as go

from levelapp.simulator.schemas import SimulationResults
from levelapp.visualization.dashboard import DashboardGenerator
from levelapp.aspects import logger


class ResultsExporter:
    """Export evaluation results and visualizations to multiple formats."""

    def __init__(self, output_dir: str):
        """
        Initialize ResultsExporter.

        Args:
            output_dir: Directory to save exported files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_chart(
        self, figure: go.Figure, filename: str, formats: List[str] = ["html", "png"]
    ) -> Dict[str, str]:
        """
        Export a single chart to multiple formats.

        Args:
            figure: Plotly Figure object
            filename: Base filename (without extension)
            formats: List of formats to export (html, png, pdf, json)

        Returns:
            Dictionary mapping format to file path
        """
        logger.info(
            f"[ResultsExporter] Exporting chart '{filename}' to formats: {formats}"
        )

        exported_files = {}

        for fmt in formats:
            output_path = self.output_dir / f"{filename}.{fmt}"

            try:
                if fmt == "html":
                    figure.write_html(
                        str(output_path),
                        include_plotlyjs="cdn",
                        config={"responsive": True},
                    )
                elif fmt == "png":
                    figure.write_image(str(output_path), width=1200, height=800)
                elif fmt == "pdf":
                    figure.write_image(str(output_path))
                elif fmt == "json":
                    figure.write_json(str(output_path))
                else:
                    logger.warning(f"Unsupported format: {fmt}")
                    continue

                exported_files[fmt] = str(output_path.absolute())
                logger.info(f"[ResultsExporter] Exported to: {output_path}")

            except Exception as e:
                logger.error(
                    f"[ResultsExporter] Failed to export {filename} as {fmt}: {e}"
                )

        return exported_files

    def export_dashboard(
        self, results: SimulationResults, formats: List[str] = ["html"]
    ) -> Dict[str, str]:
        """
        Export a complete dashboard with all visualizations.

        Args:
            results: SimulationResults object
            formats: List of formats to export

        Returns:
            Dictionary mapping format to file path
        """
        logger.info(f"[ResultsExporter] Exporting dashboard to formats: {formats}")

        exported_files = {}
        dashboard_gen = DashboardGenerator()

        # Generate HTML dashboard
        if "html" in formats:
            html_path = self.output_dir / "dashboard.html"
            dashboard_path = dashboard_gen.generate_simulator_dashboard(
                results=results,
                output_path=str(html_path),
                title="Evaluation Dashboard",
            )
            exported_files["html"] = dashboard_path

        # Export individual charts if PNG/PDF requested
        if any(fmt in formats for fmt in ["png", "pdf"]):
            from levelapp.visualization.charts import ChartGenerator

            chart_gen = ChartGenerator()

            charts_to_export = {
                "score_trend": chart_gen.create_score_trend(results),
                "provider_comparison": chart_gen.create_provider_comparison(results),
            }

            for chart_name, figure in charts_to_export.items():
                chart_formats = [fmt for fmt in formats if fmt in ["png", "pdf"]]
                self.export_chart(figure, chart_name, chart_formats)

        return exported_files

    def create_archive(
        self, files: List[str], archive_name: str = "evaluation_results.zip"
    ) -> str:
        """
        Create a ZIP archive of exported files.

        Args:
            files: List of file paths to include
            archive_name: Name of the archive file

        Returns:
            Path to the created archive
        """
        logger.info(f"[ResultsExporter] Creating archive: {archive_name}")

        archive_path = self.output_dir / archive_name

        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in files:
                file_path = Path(file_path)
                if file_path.exists():
                    zipf.write(file_path, file_path.name)
                    logger.info(f"[ResultsExporter] Added to archive: {file_path.name}")

        logger.info(f"[ResultsExporter] Archive created: {archive_path}")
        return str(archive_path.absolute())

    def export_results_json(
        self, results: SimulationResults, filename: str = "results.json"
    ) -> str:
        """
        Export raw results to JSON file.

        Args:
            results: SimulationResults object
            filename: Output filename

        Returns:
            Path to the JSON file
        """
        logger.info(f"[ResultsExporter] Exporting results to JSON: {filename}")

        output_path = self.output_dir / filename
        output_path.write_text(results.model_dump_json(indent=2), encoding="utf-8")

        logger.info(f"[ResultsExporter] JSON exported to: {output_path}")
        return str(output_path.absolute())
