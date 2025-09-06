#!/usr/bin/env python3
"""
Clarke Error Grid Analysis Utilities

This module provides functions for performing Clarke Error Grid analysis
on glucose prediction results, including visualization and statistics calculation.
"""

import numpy as np
import matplotlib.pyplot as plt


def clarke_error_grid_analysis(ref_values, pred_values, title_string, save_path=None):
    """
    Perform Clarke Error Grid analysis and return zone statistics

    Parameters:
    -----------
    ref_values : array-like
        Reference glucose values (mg/dl)
    pred_values : array-like
        Predicted glucose values (mg/dl)
    title_string : str
        Title for the plot
    save_path : str, optional
        Path to save the plot. If None, plot is not saved

    Returns:
    --------
    dict
        Dictionary containing zone statistics:
        - zone_counts: Count of points in each zone (A, B, C, D, E)
        - zone_percentages: Percentage of points in each zone
        - total_points: Total number of valid points
        - clinically_acceptable: Percentage in zones A+B
        - clinically_dangerous: Percentage in zones D+E
    """
    # Convert to numpy arrays
    ref_values = np.array(ref_values)
    pred_values = np.array(pred_values)

    # Filter out invalid values
    valid_mask = (
        ~np.isnan(ref_values)
        & ~np.isnan(pred_values)
        & (ref_values >= 0)
        & (pred_values >= 0)
        & (ref_values <= 400)
        & (pred_values <= 400)
    )

    ref_values = ref_values[valid_mask]
    pred_values = pred_values[valid_mask]

    # Calculate zone statistics
    zone_counts = _calculate_clarke_zones(ref_values, pred_values)

    total_points = sum(zone_counts)
    zone_percentages = [count / total_points * 100 for count in zone_counts]

    stats = {
        "zone_counts": dict(zip(["A", "B", "C", "D", "E"], zone_counts)),
        "zone_percentages": dict(zip(["A", "B", "C", "D", "E"], zone_percentages)),
        "total_points": total_points,
        "clinically_acceptable": (zone_counts[0] + zone_counts[1]) / total_points * 100,
        "clinically_dangerous": (zone_counts[3] + zone_counts[4]) / total_points * 100,
    }

    # Create plot if save_path is provided
    if save_path:
        _create_clarke_error_grid_plot(ref_values, pred_values, title_string, save_path)
        print(f"✓ Clarke Error Grid saved to: {save_path}")

    return stats


def _calculate_clarke_zones(ref_values, pred_values):
    """
    Calculate Clarke Error Grid zone assignments for each point

    Parameters:
    -----------
    ref_values : numpy.ndarray
        Reference glucose values
    pred_values : numpy.ndarray
        Predicted glucose values

    Returns:
    --------
    list
        List of 5 integers representing counts in zones A, B, C, D, E
    """
    zone = [0] * 5

    for i in range(len(ref_values)):
        ref_val = ref_values[i]
        pred_val = pred_values[i]

        # Zone A: Clinically accurate values
        if (ref_val <= 70 and pred_val <= 70) or (
            pred_val <= 1.2 * ref_val and pred_val >= 0.8 * ref_val
        ):
            zone[0] += 1  # Zone A

        # Zone E: Erroneous values (most dangerous)
        elif (ref_val >= 180 and pred_val <= 70) or (ref_val <= 70 and pred_val >= 180):
            zone[4] += 1  # Zone E

        # Zone C: Overcorrection values
        elif ((ref_val >= 70 and ref_val <= 290) and pred_val >= ref_val + 110) or (
            (ref_val >= 130 and ref_val <= 180)
            and (pred_val <= (7 / 5) * ref_val - 182)
        ):
            zone[2] += 1  # Zone C

        # Zone D: Dangerous failure to detect values
        elif (
            (ref_val >= 240 and (pred_val >= 70 and pred_val <= 180))
            or (ref_val <= 175 / 3 and pred_val <= 180 and pred_val >= 70)
            or (
                (ref_val >= 175 / 3 and ref_val <= 70) and pred_val >= (6 / 5) * ref_val
            )
        ):
            zone[3] += 1  # Zone D

        # Zone B: Benign errors
        else:
            zone[1] += 1  # Zone B

    return zone


def _create_clarke_error_grid_plot(ref_values, pred_values, title_string, save_path):
    """
    Create and save Clarke Error Grid visualization

    Parameters:
    -----------
    ref_values : numpy.ndarray
        Reference glucose values
    pred_values : numpy.ndarray
        Predicted glucose values
    title_string : str
        Title for the plot
    save_path : str
        Path to save the plot
    """
    plt.figure(figsize=(8, 8), dpi=300)

    # Plot data points
    plt.scatter(
        ref_values,
        pred_values,
        marker="o",
        color="navy",
        s=12,
        alpha=0.6,
        edgecolors="black",
        linewidth=0.1,
    )

    # Set labels and title
    plt.title(title_string + " Clarke Error Grid", fontsize=16, fontweight="bold")
    plt.xlabel("Reference Concentration (mg/dl)", fontsize=14)
    plt.ylabel("Prediction Concentration (mg/dl)", fontsize=14)

    # Add perfect prediction line
    plt.plot(
        [0, 400],
        [0, 400],
        ":",
        c="gray",
        linewidth=2,
        alpha=0.7,
        label="Perfect prediction",
    )

    # Add zone boundaries
    _add_clarke_zone_boundaries()

    # Add zone labels
    _add_clarke_zone_labels()

    # Configure plot appearance
    plt.xlim([0, 400])
    plt.ylim([0, 400])
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.gca().set_aspect("equal")
    plt.tight_layout()

    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def _add_clarke_zone_boundaries():
    """Add Clarke Error Grid zone boundary lines to the current plot"""
    zone_line_color = "black"
    zone_line_width = 1.5

    # Zone boundary lines according to Clarke Error Grid specification
    plt.plot([0, 175 / 3], [70, 70], "-", c=zone_line_color, linewidth=zone_line_width)
    plt.plot(
        [175 / 3, 400 / 1.2],
        [70, 400],
        "-",
        c=zone_line_color,
        linewidth=zone_line_width,
    )
    plt.plot([70, 70], [84, 400], "-", c=zone_line_color, linewidth=zone_line_width)
    plt.plot([0, 70], [180, 180], "-", c=zone_line_color, linewidth=zone_line_width)
    plt.plot([70, 290], [180, 400], "-", c=zone_line_color, linewidth=zone_line_width)
    plt.plot([70, 70], [0, 56], "-", c=zone_line_color, linewidth=zone_line_width)
    plt.plot([70, 400], [56, 320], "-", c=zone_line_color, linewidth=zone_line_width)
    plt.plot([180, 180], [0, 70], "-", c=zone_line_color, linewidth=zone_line_width)
    plt.plot([180, 400], [70, 70], "-", c=zone_line_color, linewidth=zone_line_width)
    plt.plot([240, 240], [70, 180], "-", c=zone_line_color, linewidth=zone_line_width)
    plt.plot([240, 400], [180, 180], "-", c=zone_line_color, linewidth=zone_line_width)
    plt.plot([130, 180], [0, 70], "-", c=zone_line_color, linewidth=zone_line_width)


def _add_clarke_zone_labels():
    """Add zone labels (A, B, C, D, E) to the current plot"""
    zone_font_size = 15
    zone_font_weight = "bold"

    # Zone A
    plt.text(
        30, 15, "A", fontsize=zone_font_size, fontweight=zone_font_weight, ha="center"
    )

    # Zone B (appears in two regions)
    plt.text(
        370, 220, "B", fontsize=zone_font_size, fontweight=zone_font_weight, ha="center"
    )
    plt.text(
        290, 370, "B", fontsize=zone_font_size, fontweight=zone_font_weight, ha="center"
    )

    # Zone C (appears in two regions)
    plt.text(
        160, 370, "C", fontsize=zone_font_size, fontweight=zone_font_weight, ha="center"
    )
    plt.text(
        160, 15, "C", fontsize=zone_font_size, fontweight=zone_font_weight, ha="center"
    )

    # Zone D (appears in two regions)
    plt.text(
        30, 140, "D", fontsize=zone_font_size, fontweight=zone_font_weight, ha="center"
    )
    plt.text(
        370, 90, "D", fontsize=zone_font_size, fontweight=zone_font_weight, ha="center"
    )

    # Zone E (appears in two regions)
    plt.text(
        30, 370, "E", fontsize=zone_font_size, fontweight=zone_font_weight, ha="center"
    )
    plt.text(
        370, 15, "E", fontsize=zone_font_size, fontweight=zone_font_weight, ha="center"
    )


def get_clarke_error_grid_stats(
    ref_values, pred_values, title_string="", show_plot=True, save_path=None
):
    """
    Enhanced Clarke Error Grid analysis with detailed statistics

    This function is a wrapper around clarke_error_grid_analysis for backward compatibility
    with existing notebooks and scripts.

    Parameters:
    -----------
    ref_values : array-like
        Reference glucose values
    pred_values : array-like
        Predicted glucose values
    title_string : str, optional
        Title for the plot
    show_plot : bool, optional
        Whether to show the plot (deprecated, use save_path instead)
    save_path : str, optional
        Path to save the plot

    Returns:
    --------
    dict
        Dictionary containing Clarke Error Grid statistics
    """
    return clarke_error_grid_analysis(ref_values, pred_values, title_string, save_path)
