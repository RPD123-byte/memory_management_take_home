    #!/usr/bin/env python3
"""
Demo script showing the memory management and compression system in action.

This script demonstrates:
1. Loading tool execution results from a trace file
2. Adding them to the knowledge graph
3. Generating summaries (simulating parallel processing)
4. Compressing tool results to save memory
5. Expanding compressed results when needed
6. Showing the dashboard with different compression states

REQUIREMENTS:
- Neo4j database running with credentials in .env
- OpenAI API key in .env
- All dependencies installed via pip install -r requirements.txt
"""

import json
import os
import time
from typing import Dict, List, Set, Any
from termcolor import colored
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from knowledge_graph_service import KnowledgeGraphService


def load_tool_execution_trace(file_path: str) -> List[Dict[str, Any]]:
    """Load tool execution trace from JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(colored(f"Loaded {len(data)} tool executions from {file_path}", "green"))
        return data
    except Exception as e:
        print(colored(f"Error loading trace file: {str(e)}", "red"))
        return []


def check_environment():
    """Check if all required environment variables are set"""
    required_vars = {
        "NEO4J_URI": "Neo4j database URI",
        "NEO4J_USERNAME": "Neo4j username", 
        "NEO4J_PASSWORD": "Neo4j password",
        "OPENAI_API_KEY": "OpenAI API key"
    }
    
    missing = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing.append(f"{var} ({description})")
    
    if missing:
        print(colored("❌ Missing required environment variables:", "red"))
        for var in missing:
            print(colored(f"  - {var}", "red"))
        print(colored("\nPlease create a .env file with these variables:", "yellow"))
        print(colored("NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io", "yellow"))
        print(colored("NEO4J_USERNAME=neo4j", "yellow"))
        print(colored("NEO4J_PASSWORD=your-password-here", "yellow"))
        print(colored("OPENAI_API_KEY=sk-your-openai-api-key-here", "yellow"))
        return False
    
    return True


def test_services():
    """Test connections to Neo4j and OpenAI services"""
    print(colored("Testing service connections...", "blue"))
    
    # Test Neo4j connection
    try:
        from neo4j_service import Neo4jService
        neo4j_service = Neo4jService()
        neo4j_service.test_connection()
        print(colored("✓ Neo4j connection successful", "green"))
        neo4j_service.close()
    except Exception as e:
        print(colored(f"✗ Neo4j connection failed: {str(e)}", "red"))
        return False
    
    # Test OpenAI connection
    try:
        from llm_service import LLMService, Message
        llm_service = LLMService()
        test_messages = [Message(role="user", content="Say 'test successful'")]
        response = llm_service.generate(test_messages)
        print(colored("✓ OpenAI connection successful", "green"))
        print(colored(f"  Response: {response[:50]}...", "cyan"))
    except Exception as e:
        print(colored(f"✗ OpenAI connection failed: {str(e)}", "red"))
        return False
    
    return True

def run_entry_with_cache(kg_service: KnowledgeGraphService, tool_entry: Dict[str, Any], token_counter) -> Dict[str, Any]:
    """
    Try to reuse a cached result for READ-like actions.
    If cache hit -> return {"reused": True, "tool_id": <TR-id>, "avoided_tokens": <int>}
    If miss -> add a new tool result and return {"reused": False, "tool_id": "TR-x", "avoided_tokens": 0}
    """
    action_type = tool_entry.get("action_type", "unknown")
    action = tool_entry.get("action", {}) or {}

    # only try reuse for reads
    op_type = kg_service._classify_op(action_type, action)
    if op_type == "read":
        cached = kg_service.preflight(action_type, action)
        if cached:
            # estimate tokens we would have added if we *did* store this tool result
            would_store = json.dumps(tool_entry, indent=2)
            avoided = token_counter.count_tokens(would_store)
            print(colored(kg_service.render_reused_result(cached), "magenta"))
            return {"reused": True, "tool_id": cached["tool_id"], "avoided_tokens": avoided}

    # miss (or write): store a new node
    tool_id = kg_service.add_tool_result(tool_entry)
    return {"reused": False, "tool_id": tool_id, "avoided_tokens": 0}

def simulate_agent_workflow():
    """Simulate an agent workflow with memory management"""
    print(colored("=== MEMORY MANAGEMENT DEMO ===", "cyan", attrs=['bold']))
    print(colored("Using REAL Neo4j and OpenAI services", "green", attrs=['bold']))
    print()
    
    # Check environment
    if not check_environment():
        print(colored("Please fix environment setup and try again.", "red"))
        return
    
    # Test services
    if not test_services():
        print(colored("Please fix service connections and try again.", "red"))
        return
    
    # Initialize the knowledge graph service
    workflow_id = f"demo_workflow_{int(time.time())}"
    print(colored(f"Initializing workflow: {workflow_id}", "blue"))
    
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        kg_service = KnowledgeGraphService(workflow_id, api_key)
        print(colored("✓ Knowledge graph service initialized", "green"))
    except Exception as e:
        print(colored(f"✗ Failed to initialize knowledge graph service: {str(e)}", "red"))
        return
    
    try:
        # Load tool execution data
        trace_data = load_tool_execution_trace("examples/tool_execution_trace.json")
        if not trace_data:
            print(colored("No trace data to process", "red"))
            return
        
        print()
        print(colored("=== PHASE 1: ADDING TOOL RESULTS ===", "cyan", attrs=['bold']))
        
        # Add all tool results to the knowledge graph
        tool_ids = []
        cache_hits = 0
        cache_avoided_tokens = 0

        # a local token counter for avoided-tokens estimate
        from token_counter import TokenCounter
        _demo_token_counter = TokenCounter()

        for i, tool_entry in enumerate(trace_data):
            print(f"Processing tool {i+1}/{len(trace_data)}...")
            res = run_entry_with_cache(kg_service, tool_entry, _demo_token_counter)
            if res["reused"]:
                cache_hits += 1
                cache_avoided_tokens += res["avoided_tokens"]
            else:
                tool_ids.append(res["tool_id"])
            time.sleep(0.05)
        
        print()
        print(colored("=== INITIAL DASHBOARD (before summaries) ===", "cyan"))
        dashboard = kg_service.generate_dashboard()
        print(dashboard)
        
        # Right before PHASE 2
        tool_ids = [tr.tool_id for tr in kg_service.get_all_tool_results()]

        print()
        print(colored("=== PHASE 2: GENERATING SUMMARIES ===", "cyan", attrs=['bold']))
        print(colored("# NOTE: In the actual agent, this summarization happens in PARALLEL", "yellow"))
        print(colored("# with determining the next action, saving time since they don't depend on each other.", "yellow"))
        print(colored("# This is using REAL OpenAI API calls to generate intelligent summaries!", "yellow"))
        print()
        
        # Generate summaries for all tool results
        summaries = []
        for i, tool_id in enumerate(tool_ids):
            print(f"Generating summary for {tool_id} ({i+1}/{len(tool_ids)})...")
            summary = kg_service.generate_summary(tool_id)
            if summary:
                summaries.append(summary)
                print(colored(f"  Summary: {summary.summary_content[:80]}...", "green"))
            else:
                print(colored(f"  Failed to generate summary for {tool_id}", "red"))
            time.sleep(0.5)  # Small delay for API rate limiting
        
        print()
        print(colored("=== DASHBOARD WITH SUMMARIES ===", "cyan"))
        dashboard = kg_service.generate_dashboard()
        print(dashboard)
        
        print()
        print(colored("=== PHASE 3: DEMONSTRATING COMPRESSION ===", "cyan", attrs=['bold']))
        
        # Demonstrate compression of related tools
        # Group AWS-related commands
        aws_tools = [tid for tid in tool_ids[:6]]  # First 6 are AWS commands
        print(colored(f"Compressing AWS-related tools: {', '.join(aws_tools)}", "yellow"))
        
        if kg_service.compress_tool_results(aws_tools):
            print(colored("✓ AWS tools compressed successfully", "green"))
        else:
            print(colored("✗ Failed to compress AWS tools", "red"))
        
        # Group file operation tools
        file_tools = [tid for tid in tool_ids[6:]]  # Rest are file operations
        if file_tools:
            print(colored(f"Compressing file operation tools: {', '.join(file_tools)}", "yellow"))
            
            if kg_service.compress_tool_results(file_tools):
                print(colored("✓ File operation tools compressed successfully", "green"))
            else:
                print(colored("✗ Failed to compress file operation tools", "red"))
        
        print()
        print(colored("=== DASHBOARD WITH COMPRESSION ===", "cyan"))
        
        # Create compression groups for dashboard
        compressed_groups = {
            "aws_group": {
                "tool_ids": aws_tools,
                "summary": "AWS Infrastructure Operations",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        if file_tools:
            compressed_groups["file_group"] = {
                "tool_ids": file_tools,
                "summary": "File System Operations", 
                "timestamp": datetime.now().isoformat()
            }
        
        dashboard = kg_service.generate_dashboard(compressed_tool_groups=compressed_groups)
        print(dashboard)
        
        print()
        print(colored("=== PHASE 4: DEMONSTRATING EXPANSION ===", "cyan", attrs=['bold']))
        
        # Demonstrate expansion of specific tools
        expanded_tools = {aws_tools[0]}  # Expand first AWS tool
        if file_tools:
            expanded_tools.add(file_tools[0])  # Expand first file tool
            
        print(colored(f"Expanding tools for detailed view: {', '.join(expanded_tools)}", "yellow"))
        
        dashboard = kg_service.generate_dashboard(
            compressed_tool_groups=compressed_groups,
            expanded_tools=expanded_tools
        )
        print(dashboard)
        
        print()
        print(colored("=== PHASE 5: RETRIEVING INDIVIDUAL SUMMARIES ===", "cyan", attrs=['bold']))
        
        # Show individual tool summaries with salient data
        for tool_id in tool_ids[:3]:  # Show first 3 as examples
            print(f"\n--- {tool_id} ---")
            summary_with_data = kg_service.retrieve_tool_result_with_salient_data(tool_id)
            if summary_with_data:
                print(colored(f"Summary with salient data: {summary_with_data}", "green"))
            else:
                print(colored("No summary available", "red"))
        
        print()
        print(colored("=== FINAL STATISTICS ===", "cyan", attrs=['bold']))
        
        # Show final statistics
        all_tools = kg_service.get_all_tool_results()
        total_tokens = sum(tool.token_count for tool in all_tools)
        
        print(f"Total tools processed: {len(all_tools)}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Summaries generated: {len(summaries)}")
        print(f"Compression groups created: {len(compressed_groups)}")

        print(f"Cache hits (initial pass): {cache_hits}")
        print(f"Cache avoided tokens (initial): {cache_avoided_tokens:,}")
        
        # Calculate compression ratio
        if summaries:
            summary_tokens = sum(s.token_count for s in summaries)
            compression_ratio = (total_tokens - summary_tokens) / total_tokens * 100
            print(f"Compression ratio: {compression_ratio:.1f}%")
            print(f"Token savings: {total_tokens - summary_tokens:,} tokens")
        
        print()
        print(colored("=== DEMO COMPLETE ===", "cyan", attrs=['bold']))
        print(colored("This demonstrates how the agent manages memory by:", "white"))
        print(colored("1. Adding tool results as they're executed", "white"))
        print(colored("2. Generating summaries in parallel with planning", "white"))
        print(colored("3. Compressing related tools to save memory", "white"))
        print(colored("4. Expanding specific tools when details are needed", "white"))
        print(colored("5. Maintaining a dashboard view of all operations", "white"))
        print()
        print(colored("✨ All operations used REAL Neo4j and OpenAI services!", "green", attrs=['bold']))
        
    except Exception as e:
        print(colored(f"Error during demo: {str(e)}", "red"))
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        try:
            print(colored(f"\nCleaning up workflow {workflow_id}...", "yellow"))
            kg_service.reset_workflow()
            kg_service.close()
            print(colored("✓ Cleanup complete", "green"))
        except Exception as e:
            print(colored(f"Warning: Cleanup failed: {str(e)}", "yellow"))


if __name__ == "__main__":
    simulate_agent_workflow() 