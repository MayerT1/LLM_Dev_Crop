#!/usr/bin/env python3
"""
Graph Visualization Script for Economic Evaluator Agent
Shows different ways to visualize the workflow graph
"""

from agent import EconomicEvaluatorAgent
import os


def main():
    """Main function to demonstrate graph visualization"""
    print("Economic Evaluator Agent - Graph Visualization")
    print("=" * 50)

    # Initialize the agent
    agent = EconomicEvaluatorAgent()

    print("\n1. Text-based Graph Structure")
    print("-" * 30)
    agent.print_graph_structure()

    print("\n2. Creating Mermaid Diagram")
    print("-" * 30)
    mermaid_file = agent.create_mermaid_diagram("workflow_diagram.md")
    if mermaid_file:
        print(f"✓ Mermaid diagram created: {mermaid_file}")
        print("  You can view this in any Markdown viewer or GitHub")

    print("\n3. Attempting PNG Visualization")
    print("-" * 30)
    try:
        png_file = agent.visualize_graph("graph_visualization.png")
        if png_file and os.path.exists(png_file):
            print(f"✓ PNG visualization created: {png_file}")
        else:
            print("! PNG visualization not available (fallback to text)")
    except Exception as e:
        print(f"! PNG visualization failed: {e}")

    print("\n4. Graph Statistics")
    print("-" * 30)
    print(f"Total nodes: 7")
    print(f"Entry point: analyze_input")
    print(f"End point: present_results")
    print(f"Conditional routing: analyze_input → workflow handlers")
    print(f"Supported workflows: 3 (yield_data, fertilization_plan, generate_fertilization)")

    print("\n5. Workflow Decision Tree")
    print("-" * 30)
    print_decision_tree()

    print("\nVisualization complete!")
    print("Files created:")
    if os.path.exists("workflow_diagram.md"):
        print("  - workflow_diagram.md (Mermaid diagram)")
    if os.path.exists("graph_visualization.png"):
        print("  - graph_visualization.png (PNG image)")


def print_decision_tree():
    """Print a decision tree representation"""
    tree = """
    User Input
    │
    ├─ Has yield data?
    │  ├─ Yes → handle_yield_data
    │  │      ├─ Has crop type? → get market price
    │  │      └─ No crop type? → ask for crop type
    │  │
    │  └─ No → Continue checking...
    │
    ├─ Has fertilization plan?
    │  ├─ Yes → handle_fertilization_plan
    │  │      ├─ Has all info? → estimate yield → get market price
    │  │      └─ Missing info? → ask for missing data
    │  │
    │  └─ No → Continue checking...
    │
    ├─ Has planting date but no fertilization?
    │  ├─ Yes → generate_fertilization_options
    │  │      ├─ Create 3 scenarios (low/med/high)
    │  │      ├─ Estimate yields for each
    │  │      └─ Get market price
    │  │
    │  └─ No → get_crop_info
    │
    └─ All paths lead to:
       calculate_economics → present_results
    """
    print(tree)


if __name__ == "__main__":
    main()