#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report Builder Module

Builds consolidated PDF reports while preserving individual figures
for publication extraction.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class ReportBuilder:
    """
    Builds consolidated PDF reports while preserving individual figures.
    
    Usage:
        report = ReportBuilder(output_dir)
        report.add_text_page("Population summary...", title="Configuration")
        report.add_figure(fig, "price_series")
        report.add_figure(fig2, "success_boxplot")
        report_path = report.build()
    
    Output structure:
        output_dir/
        ├── experiment_report.pdf    (consolidated)
        └── figures/
            ├── price_series.pdf     (individual, publication-ready)
            └── success_boxplot.pdf
    """

    def __init__(self, output_dir: str, report_name: str = "experiment_report.pdf"):
        """
        Initialize ReportBuilder.
        
        Parameters
        ----------
        output_dir : str
            Base directory for all outputs
        report_name : str
            Name of consolidated PDF report
        """
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = self.output_dir / report_name
        self._pages: List[Tuple[plt.Figure, str, bool]] = []  # (fig, name, close_after)

    def add_figure(self, fig: plt.Figure, name: str, close_after: bool = False) -> str:
        """
        Save figure individually and queue for report.
        
        Parameters
        ----------
        fig : plt.Figure
            Matplotlib figure to add
        name : str
            Filename stem (without extension)
        close_after : bool
            If True, close figure after building report
            
        Returns
        -------
        str
            Path to individual figure PDF
        """
        # Save individual PDF for publication extraction
        individual_path = self.figures_dir / f"{name}.pdf"
        fig.savefig(individual_path, format="pdf", dpi=300, bbox_inches="tight")

        # Queue for consolidated report
        self._pages.append((fig, name, close_after))

        return str(individual_path)

    def add_text_page(
        self,
        text: str,
        title: Optional[str] = None,
        fontsize: int = 10,
        fontfamily: str = "monospace"
    ) -> None:
        """
        Add a text-only page (for summaries, population descriptions).
        
        Parameters
        ----------
        text : str
            Text content for the page
        title : str, optional
            Page title
        fontsize : int
            Font size for body text
        fontfamily : str
            Font family for body text
        """
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait
        ax.axis("off")

        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

        ax.text(
            0.05, 0.95, text,
            transform=ax.transAxes,
            fontsize=fontsize,
            verticalalignment="top",
            fontfamily=fontfamily,
            wrap=True
        )

        # Text pages go to report only (no individual PDF needed)
        self._pages.append((fig, f"_text_{title or 'page'}", True))

    def add_summary_table(
        self,
        data: dict,
        title: str = "Summary Statistics"
    ) -> None:
        """
        Add a formatted key-value summary page.
        
        Parameters
        ----------
        data : dict
            Key-value pairs to display
        title : str
            Page title
        """
        lines = [f"{k}: {v}" for k, v in data.items()]
        text = "\n".join(lines)
        self.add_text_page(text, title=title)

    def build(self) -> str:
        """
        Assemble all pages into consolidated report.
        
        Returns
        -------
        str
            Path to consolidated report PDF
        """
        with PdfPages(self.report_path) as pdf:
            for fig, name, close_after in self._pages:
                pdf.savefig(fig, bbox_inches="tight")
                if close_after:
                    plt.close(fig)

        # Close any remaining figures
        for fig, name, close_after in self._pages:
            if not close_after:
                plt.close(fig)

        self._pages = []
        return str(self.report_path)

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Build report on context exit."""
        if self._pages:
            self.build()
        return False
