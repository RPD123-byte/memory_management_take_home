#!/usr/bin/env python3
"""
Demo script comparing compression effectiveness between original and optimized services
"""

import json
import os
from datetime import datetime
from termcolor import colored

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from knowledge_graph_service import KnowledgeGraphService
from optimized_knowledge_graph_service import OptimizedKnowledgeGraphService

def load_sample_tools():
    """Load sample tool execution data from examples/tool_execution_trace.json"""
    try:
        with open('examples/tool_execution_trace.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(colored("Warning: examples/tool_execution_trace.json not found, using minimal fallback data", "yellow"))
        return [
        ]

def main():
    print(colored("=== COMPRESSION COMPARISON DEMO ===", "cyan", attrs=["bold"]))
    print(colored("Comparing original vs optimized knowledge graph services", "cyan"))
    print()
    
    workflow_id = f"demo_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print(colored("OPENAI_API_KEY not set. Demo will use fallback compression methods.", "yellow"))
    
    print(colored("Initializing services with shared data...", "blue"))
    # Use same workflow_id so they share the same tool data
    original_service = KnowledgeGraphService(workflow_id, api_key)
    optimized_service = OptimizedKnowledgeGraphService(workflow_id, api_key)
    
    sample_tools = load_sample_tools()
    print(colored(f"Adding {len(sample_tools)} tool results to both services...", "blue"))
    
    for i, tool_data in enumerate(sample_tools, 1):
        print(f"  Adding tool {i}/{len(sample_tools)}: {tool_data['action_type']}")
        
        # Add to original service only (optimized service will see the same data)
        original_service.add_tool_result(tool_data)
    
    print()
    
    print(colored("Generating summaries...", "blue"))
    
    original_tools = original_service.get_all_tool_results()
    optimized_tools = optimized_service.get_all_tool_results()
    
    for tool in original_tools:
        original_service.generate_summary(tool.tool_id)
    
    for tool in optimized_tools:
        optimized_service.generate_summary(tool.tool_id)
    
    print()
    
    print(colored("=== GENERATING DASHBOARDS ===", "cyan", attrs=["bold"]))
    print()
    
    print(colored("1. ORIGINAL SERVICE DASHBOARD:", "green", attrs=["bold"]))
    all_tools = {
        "all" : {
            "tool_ids" : [tool.tool_id for tool in original_tools], 
            "summary" : "All tools",
            "timestamp": datetime.now().isoformat()
        }
    }
    original_dashboard = original_service.generate_dashboard(compressed_tool_groups=all_tools)
    print(original_dashboard)
    
    print("\n" + "="*80 + "\n")
    
    print(colored("2. OPTIMIZED SERVICE DASHBOARD:", "green", attrs=["bold"]))
    optimized_dashboard = optimized_service.generate_dashboard(compressed_tool_groups=all_tools)
    print(optimized_dashboard)
    
    print("\n" + "="*80 + "\n")
    
    # Calculate final comparison
    original_tokens = original_service.token_counter.count_tokens(original_dashboard)
    optimized_tokens = optimized_service.token_counter.count_tokens(optimized_dashboard)
    
    print(colored("=== FINAL COMPARISON ===", "cyan", attrs=["bold"]))
    print(f"📊 Original Dashboard: {original_tokens:,} tokens")
    print(f"🔧 Optimized Dashboard: {optimized_tokens:,} tokens")
    print(f"💾 Space Saved: {original_tokens - optimized_tokens:,} tokens")
    print(f"📈 Compression Ratio: {optimized_tokens/original_tokens*100:.1f}%")
    
    if optimized_tokens < original_tokens:
        print(colored(f"✅ Optimized service saved {original_tokens - optimized_tokens:,} tokens!", "green"))
    else:
        print(colored("⚠️  Optimized service used more tokens (may need tuning)", "yellow"))
    
    print()
    print(colored("Demo completed! Check the output above to see compression effectiveness.", "cyan"))

if __name__ == "__main__":
    main() 