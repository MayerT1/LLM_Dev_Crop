#!/usr/bin/env python3
"""
Visualization script for the EconomicEvaluatorAgent LangGraph workflow.
This script creates a visual representation of the workflow including LLM calls as black boxes.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from typing import Dict, List, Tuple


class WorkflowVisualizer:
    """Visualizes the EconomicEvaluatorAgent workflow"""

    def __init__(self):
        self.fig, self.ax = plt.subplots(1, 1, figsize=(18, 14))  # Larger canvas
        self.node_positions = {}
        self.node_colors = {
            'entry': '#4CAF50',  # Green for entry point
            'process': '#2196F3',  # Blue for processing nodes
            'llm': '#FF9800',  # Orange for LLM black boxes
            'decision': '#9C27B0',  # Purple for conditional nodes
            'end': '#F44336'  # Red for end nodes
        }

    def create_visualization(self):
        """Create the complete workflow visualization"""
        self._define_node_positions()
        self._draw_nodes()
        self._draw_edges()
        self._add_llm_blackboxes()
        self._add_title_and_legend()
        self._configure_plot()

    def _define_node_positions(self):
        """Define positions for all nodes in the workflow"""
        # Main workflow nodes - spread out more to reduce overlaps
        self.node_positions = {
            'improve_query': (3, 11),
            'detect_intent': (3, 9.5),
            'provide_capabilities': (0.5, 7.5),
            'parse_input': (5.5, 7.5),
            'check_completeness': (5.5, 6),
            'ask_clarification': (1.5, 4.5),
            'run_experiment': (8, 4.5),
            'generate_output': (8, 3),
            'end_conversation': (3, 3),
            'END': (5, 1)
        }

        # LLM black box positions (further offset from main nodes to avoid overlap)
        self.llm_positions = {
            'improve_query_llm': (4.5, 11.5),
            'detect_intent_llm': (4.5, 10),
            'provide_capabilities_llm': (-0.8, 8),
            'ask_clarification_llm': (0.2, 5),
            'generate_output_llm': (9.5, 3.5)
        }

    def _draw_nodes(self):
        """Draw all workflow nodes with appropriate styling"""
        # Node categories and their types
        node_types = {
            'improve_query': 'entry',
            'detect_intent': 'process',
            'provide_capabilities': 'process',
            'parse_input': 'process',
            'check_completeness': 'decision',
            'ask_clarification': 'process',
            'run_experiment': 'process',
            'generate_output': 'process',
            'end_conversation': 'process',
            'END': 'end'
        }

        for node_name, (x, y) in self.node_positions.items():
            node_type = node_types.get(node_name, 'process')
            color = self.node_colors[node_type]

            if node_name == 'END':
                # Special styling for END node
                circle = plt.Circle((x, y), 0.3, color=color, alpha=0.8)
                self.ax.add_patch(circle)
                self.ax.text(x, y, 'END', ha='center', va='center',
                             fontsize=10, fontweight='bold', color='white')
            else:
                # Regular nodes as rounded rectangles
                width = 1.2 if len(node_name) < 12 else 1.5
                height = 0.4

                fancy_box = FancyBboxPatch(
                    (x - width / 2, y - height / 2), width, height,
                    boxstyle="round,pad=0.05",
                    facecolor=color,
                    edgecolor='black',
                    linewidth=1.5,
                    alpha=0.8
                )
                self.ax.add_patch(fancy_box)

                # Node text
                display_name = node_name.replace('_', '\n')
                self.ax.text(x, y, display_name, ha='center', va='center',
                             fontsize=9, fontweight='bold', color='white')

    def _add_llm_blackboxes(self):
        """Add LLM black boxes connected to relevant nodes"""
        llm_node_mapping = {
            'improve_query_llm': 'improve_query',
            'detect_intent_llm': 'detect_intent',
            'provide_capabilities_llm': 'provide_capabilities',
            'ask_clarification_llm': 'ask_clarification',
            'generate_output_llm': 'generate_output'
        }

        for llm_name, (x, y) in self.llm_positions.items():
            # Draw LLM black box
            width, height = 0.8, 0.3

            black_box = FancyBboxPatch(
                (x - width / 2, y - height / 2), width, height,
                boxstyle="round,pad=0.03",
                facecolor=self.node_colors['llm'],
                edgecolor='black',
                linewidth=2,
                alpha=0.9
            )
            self.ax.add_patch(black_box)

            self.ax.text(x, y, 'LLM\nCall', ha='center', va='center',
                         fontsize=8, fontweight='bold', color='white')

            # Connect to parent node with better line routing
            parent_node = llm_node_mapping[llm_name]
            parent_x, parent_y = self.node_positions[parent_node]

            # Use different line styles to distinguish from main workflow
            line_style = ':'  # Dotted line for LLM connections
            line_color = 'orange'

            # Draw connection line with offset to avoid main arrows
            self.ax.plot([parent_x, x], [parent_y, y],
                         linestyle=line_style, color=line_color,
                         alpha=0.8, linewidth=2)

            # Add small arrow with better positioning
            dx, dy = x - parent_x, y - parent_y
            norm = np.sqrt(dx ** 2 + dy ** 2)
            if norm > 0:
                dx_norm, dy_norm = dx / norm * 0.2, dy / norm * 0.2

                self.ax.arrow(x - dx_norm, y - dy_norm, dx_norm * 0.6, dy_norm * 0.6,
                              head_width=0.1, head_length=0.08,
                              fc='orange', ec='orange', alpha=0.8)

    def _draw_edges(self):
        """Draw edges between workflow nodes"""
        # Define all edges in the workflow
        edges = [
            ('improve_query', 'detect_intent'),
            ('parse_input', 'check_completeness'),
            ('run_experiment', 'generate_output'),
            ('ask_clarification', 'END'),
            ('generate_output', 'END'),
            ('end_conversation', 'END'),
            ('provide_capabilities', 'END')
        ]

        # Conditional edges
        conditional_edges = [
            ('detect_intent', 'provide_capabilities', 'capabilities'),
            ('detect_intent', 'parse_input', 'experiment'),
            ('check_completeness', 'ask_clarification', 'clarify'),
            ('check_completeness', 'run_experiment', 'proceed'),
            ('check_completeness', 'end_conversation', 'end')
        ]

        # Draw regular edges
        for start, end in edges:
            start_pos = self.node_positions[start]
            end_pos = self.node_positions[end]
            self._draw_arrow(start_pos, end_pos)

        # Draw conditional edges with labels
        for start, end, condition in conditional_edges:
            start_pos = self.node_positions[start]
            end_pos = self.node_positions[end]
            self._draw_arrow(start_pos, end_pos, condition)

    def _draw_arrow(self, start_pos: Tuple[float, float],
                    end_pos: Tuple[float, float], label: str = None):
        """Draw an arrow between two positions with optional label"""
        x1, y1 = start_pos
        x2, y2 = end_pos

        # Calculate arrow direction and adjust for node boundaries
        dx, dy = x2 - x1, y2 - y1
        norm = np.sqrt(dx ** 2 + dy ** 2)

        if norm == 0:
            return

        # Increased offset from node edges to avoid overlap
        offset = 0.8  # Increased from 0.25
        dx_norm, dy_norm = dx / norm, dy / norm

        start_x = x1 + dx_norm * offset
        start_y = y1 + dy_norm * offset
        end_x = x2 - dx_norm * offset
        end_y = y2 - dy_norm * offset

        # For curved arrows to avoid overlapping with nodes
        if abs(dx) > abs(dy) and abs(dx) > 2:  # Horizontal-ish long arrows
            # Add curve for long horizontal arrows
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            curve_offset = 0.5 if dy >= 0 else -0.5

            # Create curved path
            self.ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                             arrowprops=dict(
                                 arrowstyle='->',
                                 lw=2,
                                 color='black',
                                 connectionstyle=f"arc3,rad={curve_offset * 0.3}"
                             ))
        else:
            # Straight arrow for shorter connections
            self.ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                             arrowprops=dict(arrowstyle='->', lw=2, color='black'))

        # Add label if provided - position it better to avoid overlaps
        if label:
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2

            # Better label positioning based on arrow direction
            if abs(dx) > abs(dy):  # Horizontal-ish arrow
                offset_x = 0
                offset_y = 0.4 if dy >= 0 else -0.4
            else:  # Vertical-ish arrow
                offset_x = 0.4 if dx >= 0 else -0.4
                offset_y = 0

            self.ax.text(mid_x + offset_x, mid_y + offset_y, label,
                         ha='center', va='center', fontsize=8,
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow',
                                   alpha=0.9, edgecolor='black', linewidth=1))

    def _add_title_and_legend(self):
        """Add title and legend to the visualization"""
        self.ax.set_title('EconomicEvaluatorAgent Workflow\nwith LLM Black Boxes',
                          fontsize=16, fontweight='bold', pad=20)

        # Create legend
        legend_elements = [
            patches.Patch(color=self.node_colors['entry'], label='Entry Point'),
            patches.Patch(color=self.node_colors['process'], label='Processing Node'),
            patches.Patch(color=self.node_colors['llm'], label='LLM Black Box'),
            patches.Patch(color=self.node_colors['decision'], label='Decision Node'),
            patches.Patch(color=self.node_colors['end'], label='End Node')
        ]

        self.ax.legend(handles=legend_elements, loc='upper right',
                       bbox_to_anchor=(1.15, 1))

    def _configure_plot(self):
        """Configure plot appearance and save"""
        self.ax.set_xlim(-2, 11)  # Expanded to accommodate wider spread
        self.ax.set_ylim(0, 13)  # Expanded for better spacing
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # Add subtle grid for better readability
        self.ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)

        plt.tight_layout()

    def save_and_show(self, filename: str = 'economic_evaluator_workflow.png'):
        """Save the visualization and show it"""
        plt.savefig(filename, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.show()

    def print_workflow_summary(self):
        """Print a text summary of the workflow"""
        print("=" * 60)
        print("ECONOMIC EVALUATOR AGENT WORKFLOW SUMMARY")
        print("=" * 60)

        print("\n🔄 MAIN WORKFLOW NODES:")
        print("  1. improve_query     - Enhance query with conversation context")
        print("  2. detect_intent     - Determine if user wants experiment or info")
        print("  3. provide_capabilities - Explain system capabilities")
        print("  4. parse_input       - Extract experiment parameters")
        print("  5. check_completeness - Verify all required info present")
        print("  6. ask_clarification - Request missing information")
        print("  7. run_experiment    - Execute agricultural simulation")
        print("  8. generate_output   - Create natural language results")
        print("  9. end_conversation  - Handle invalid locations")

        print("\n🤖 LLM BLACK BOXES:")
        print("  • improve_query_llm     - Context-aware query enhancement")
        print("  • detect_intent_llm     - Intent classification")
        print("  • provide_capabilities_llm - Capability explanation")
        print("  • ask_clarification_llm - Natural clarification requests")
        print("  • generate_output_llm   - Results to natural language")

        print("\n🔀 DECISION POINTS:")
        print("  • detect_intent → {capabilities, experiment}")
        print("  • check_completeness → {clarify, proceed, end}")

        print("\n📊 WORKFLOW FEATURES:")
        print("  ✓ Conversational context tracking")
        print("  ✓ Intent-based routing")
        print("  ✓ Parameter validation")
        print("  ✓ Location-specific constraints (Alabama only)")
        print("  ✓ Natural language interaction")
        print("  ✓ Agricultural simulation integration")

        print("\n" + "=" * 60)


def main():
    """Main function to run the visualization"""
    print("Creating EconomicEvaluatorAgent Workflow Visualization...")

    visualizer = WorkflowVisualizer()

    # Print text summary
    visualizer.print_workflow_summary()

    # Create and display visualization
    visualizer.create_visualization()
    visualizer.save_and_show()

    print("\n✅ Visualization complete!")
    print("📁 Saved as: economic_evaluator_workflow.png")


if __name__ == "__main__":
    main()