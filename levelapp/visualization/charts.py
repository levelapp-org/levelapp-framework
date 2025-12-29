"""levelapp/visualization/charts.py: Chart generation for evaluation results."""

import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any
from collections import defaultdict

from levelapp.simulator.schemas import SimulationResults
from levelapp.aspects import logger


class ChartGenerator:
    """Generate interactive charts for evaluation results."""

    def __init__(self, theme: str = "plotly_white"):
        """
        Initialize ChartGenerator with a theme.

        Args:
            theme: Plotly theme name (plotly, plotly_white, plotly_dark, ggplot2, seaborn, etc.)
        """
        self.theme = theme
        self._color_palette = px.colors.qualitative.Plotly

    def create_score_trend(
        self, results: SimulationResults, metric: str = "average"
    ) -> go.Figure:
        """
        Create a line chart showing score trends across scripts/attempts.

        Args:
            results: SimulationResults object
            metric: Which metric to plot (default: "average")

        Returns:
            Plotly Figure object
        """
        logger.info(f"[ChartGenerator] Creating score trend chart for metric: {metric}")

        fig = go.Figure()

        # Extract scores by provider
        provider_scores = defaultdict(list)
        script_ids = []

        if results.script_results:
            for idx, script_result in enumerate(results.script_results):
                script_id = script_result.script_id
                script_ids.append(script_id)

                avg_scores = script_result.average_scores
                for provider, score in avg_scores.items():
                    if (
                        provider != "processing_time"
                        and provider != "guardrail"
                        and provider != "metadata"
                    ):
                        provider_scores[provider].append(score)

        # Create line for each provider
        for idx, (provider, scores) in enumerate(provider_scores.items()):
            fig.add_trace(
                go.Scatter(
                    x=script_ids[: len(scores)],
                    y=scores,
                    mode="lines+markers",
                    name=provider.upper(),
                    line=dict(
                        width=2,
                        color=self._color_palette[idx % len(self._color_palette)],
                    ),
                    marker=dict(size=8),
                )
            )

        fig.update_layout(
            title="Evaluation Score Trends Across Scripts",
            xaxis_title="Script ID",
            yaxis_title="Score",
            template=self.theme,
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        return fig

    def create_provider_comparison(self, results: SimulationResults) -> go.Figure:
        """
        Create a bar chart comparing average scores across providers.

        Args:
            results: SimulationResults object

        Returns:
            Plotly Figure object
        """
        logger.info("[ChartGenerator] Creating provider comparison chart")

        providers = []
        scores = []

        if results.average_scores:
            for provider, score in results.average_scores.items():
                if provider not in ["processing_time", "guardrail", "metadata"]:
                    providers.append(provider.upper())
                    scores.append(score)

        fig = go.Figure(
            data=[
                go.Bar(
                    x=providers,
                    y=scores,
                    marker_color=self._color_palette[: len(providers)],
                    text=[f"{s:.3f}" for s in scores],
                    textposition="auto",
                )
            ]
        )

        fig.update_layout(
            title="Average Scores by Provider",
            xaxis_title="Provider",
            yaxis_title="Average Score",
            template=self.theme,
            yaxis=dict(range=[0, 1]),
        )

        return fig

    def create_score_distribution(
        self, results: SimulationResults, provider: str
    ) -> go.Figure:
        """
        Create a histogram and box plot showing score distribution for a provider.

        Args:
            results: SimulationResults object
            provider: Provider name to analyze

        Returns:
            Plotly Figure object with subplots
        """
        logger.info(
            f"[ChartGenerator] Creating score distribution for provider: {provider}"
        )

        from plotly.subplots import make_subplots

        # Collect all scores for the provider
        scores = []
        if results.interaction_results:
            for script_result in results.interaction_results:
                avg_scores = script_result.get("average_scores", {})
                if provider in avg_scores:
                    scores.append(avg_scores[provider])

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Score Distribution", "Box Plot"),
            vertical_spacing=0.15,
        )

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=scores,
                nbinsx=20,
                name="Distribution",
                marker_color=self._color_palette[0],
            ),
            row=1,
            col=1,
        )

        # Box plot
        fig.add_trace(
            go.Box(
                x=scores,
                name=provider.upper(),
                marker_color=self._color_palette[1],
                boxmean="sd",
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            title=f"Score Distribution for {provider.upper()}",
            template=self.theme,
            showlegend=False,
            height=600,
        )

        fig.update_xaxes(title_text="Score", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)

        return fig

    def create_metadata_heatmap(self, comparator_results: Dict[str, Any]) -> go.Figure:
        """
        Create a heatmap showing field-level accuracy from comparator results.

        Args:
            comparator_results: Dictionary of comparator evaluation results

        Returns:
            Plotly Figure object
        """
        logger.info("[ChartGenerator] Creating metadata accuracy heatmap")

        fields = []
        scores = []

        for idx, result in comparator_results.items():
            field_name = result.get("field_name", f"Field {idx}")
            set_scores = result.get("set_scores")

            if set_scores is not None:
                score = set_scores[0] if isinstance(set_scores, list) else set_scores
                fields.append(field_name)
                scores.append(float(score) if score is not None else 0.0)

        fig = go.Figure(
            data=go.Heatmap(
                z=[scores],
                x=fields,
                y=["Accuracy"],
                colorscale="RdYlGn",
                text=[[f"{s:.2f}" for s in scores]],
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Score"),
            )
        )

        fig.update_layout(
            title="Metadata Field Accuracy Heatmap",
            xaxis_title="Field Name",
            template=self.theme,
            height=300,
        )

        return fig

    def create_interaction_timeline(self, results: SimulationResults) -> go.Figure:
        """
        Create a timeline visualization of interaction performance.

        Args:
            results: SimulationResults object

        Returns:
            Plotly Figure object
        """
        logger.info("[ChartGenerator] Creating interaction timeline")

        fig = go.Figure()

        if results.interaction_results:
            for script_idx, script_result in enumerate(results.interaction_results):
                script_id = script_result.get("script_id", f"Script {script_idx + 1}")
                attempts = script_result.get("attempts", [])

                for attempt in attempts:
                    attempt_num = attempt.get("attempt", 1)
                    duration = attempt.get("total_duration", 0)

                    fig.add_trace(
                        go.Bar(
                            x=[duration],
                            y=[f"{script_id} - Attempt {attempt_num}"],
                            orientation="h",
                            name=script_id,
                            showlegend=False,
                            marker_color=self._color_palette[
                                script_idx % len(self._color_palette)
                            ],
                        )
                    )

        fig.update_layout(
            title="Interaction Processing Timeline",
            xaxis_title="Duration (seconds)",
            yaxis_title="Script - Attempt",
            template=self.theme,
            height=max(400, len(results.interaction_results or []) * 60),
        )

        return fig

    def create_summary_metrics(self, results: SimulationResults) -> go.Figure:
        """
        Create a summary metrics card visualization.

        Args:
            results: SimulationResults object

        Returns:
            Plotly Figure object
        """
        logger.info("[ChartGenerator] Creating summary metrics visualization")

        from plotly.subplots import make_subplots

        # Calculate summary metrics
        total_scripts = len(results.interaction_results or [])
        avg_score = (
            results.average_scores.get("openai", 0) if results.average_scores else 0
        )
        total_time = results.elapsed_time

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=("Total Scripts", "Avg Score", "Total Time (s)"),
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]
            ],
        )

        fig.add_trace(
            go.Indicator(
                mode="number",
                value=total_scripts,
                number={"font": {"size": 40}},
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=avg_score,
                number={"font": {"size": 40}},
                delta={"reference": 0.8, "relative": False},
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Indicator(
                mode="number",
                value=total_time,
                number={"font": {"size": 40}, "suffix": "s"},
            ),
            row=1,
            col=3,
        )

        fig.update_layout(template=self.theme, height=200, showlegend=False)

        return fig
