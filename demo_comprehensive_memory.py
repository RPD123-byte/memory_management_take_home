"""
Comprehensive Memory Management Demo.

Demonstrates the full hierarchical memory system:
1. Static Memory - Project context
2. Working Memory - Current plan and active tools
3. Fact Extraction - Automatic fact extraction from executions
4. Vector Memory - Semantic search
5. Memory-aware decision making
"""

import json
import os
from termcolor import colored
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from comprehensive_memory_service import ComprehensiveMemoryService


def print_section(title: str):
    """Print colored section header"""
    print("\n" + colored("="*80, "magenta", attrs=["bold"]))
    print(colored(title, "cyan", attrs=["bold"]))
    print(colored("="*80, "magenta", attrs=["bold"]) + "\n")


def test_connections(service):
    """Test connections"""
    print("Testing service connections...")
    
    try:
        service.neo4j_service.test_connection()
        print(colored(" Neo4j connection successful", "green"))
    except Exception as e:
        print(colored(f"✗ Neo4j failed: {e}", "red"))
        return False
    
    try:
        from llm_service import Message
        response = service.llm_service.generate(
            [Message(role="user", content="Say 'Test successful.'")],
            json_mode=False
        )
        print(colored(" OpenAI connection successful", "green"))
    except Exception as e:
        print(colored(f"✗ OpenAI failed: {e}", "red"))
        return False
    
    return True


def main():
    print(colored("\n" + "="*80, "magenta", attrs=["bold"]))
    print(colored("COMPREHENSIVE HIERARCHICAL MEMORY SYSTEM DEMO", "magenta", attrs=["bold"]))
    print(colored("="*80 + "\n", "magenta", attrs=["bold"]))
    
    print("This demo showcases a memory management system with:")
    print("  1.  Static Memory - Project context and configuration")
    print("  2.  Working Memory - Current plan and active tasks")
    print("  3.  Fact Extraction - Automatic knowledge extraction")
    print("  4.  Vector Memory - Semantic search and retrieval")
    print("  5.  Memory-Aware Decisions - Context-informed actions")
    print()
    
    # Initialize service
    workflow_id = f"comprehensive_demo_{int(__import__('time').time())}"
    print(f"Initializing workflow: {workflow_id}\n")
    
    service = ComprehensiveMemoryService(
        workflow_id=workflow_id,
        enable_semantic_memory=True,
        enable_fact_extraction=True
    )
    
    # Test connections
    if not test_connections(service):
        print(colored("\n❌ Connection tests failed. Exiting.", "red"))
        return
    
    print(colored("\n All systems operational\n", "green", attrs=["bold"]))
    
    # ========== PHASE 1: Initialize Static Memory ==========
    print_section("PHASE 1: INITIALIZING STATIC MEMORY")
    
    print("Setting up project context...")
    service.initialize_static_memory(
        workspace_path="/workspace/devops-project",
        detected_languages={
            "infrastructure": {"terraform": "v1.5.0", "ansible": "v2.14.0"},
            "backend": {"terraform_cdk": "v0.17.0"}
        },
        configured_integrations=["AWS", "GCP", "Splunk", "Grafana"]
    )
    
    # Show static memory
    static_context = service.memory_manager.blocks["static"].get()
    print(colored("\n" + static_context, "cyan"))
    
    # ========== PHASE 2: Set Working Memory (Plan) ==========
    print_section("PHASE 2: SETTING CURRENT PLAN IN WORKING MEMORY")
    
    plan_goal = "Configure AWS IAM permissions for DevOps team"
    plan_steps = [
        {"name": "List existing IAM users", "type": "aws_command"},
        {"name": "Check current user groups", "type": "aws_command"},
        {"name": "Create/update IAM policies", "type": "aws_command"},
        {"name": "Verify permissions", "type": "aws_command"}
    ]
    
    service.set_current_plan(plan_goal, plan_steps)
    print(f"Set plan: {plan_goal}")
    print(f"Steps: {len(plan_steps)}")
    
    # Add context notes
    service.add_context_note("Target: DevOps team members need S3 and IAM access")
    service.add_context_note("Constraint: Use least-privilege principle")
    
    # Show working memory
    working_context = service.memory_manager.blocks["working"].get()
    print(colored("\n" + working_context, "yellow"))
    
    # ========== PHASE 3: Execute Tools with Memory ==========
    print_section("PHASE 3: EXECUTING TOOLS WITH MEMORY AWARENESS")
    
    # Load test data
    with open("examples/tool_execution_trace.json", 'r') as f:
        tool_executions = json.load(f)
    
    print(f"Processing {len(tool_executions)} tool executions...\n")
    
    for i, tool_entry in enumerate(tool_executions[:8], 1):
        print(colored(f"\n--- Tool Execution {i}/8 ---", "yellow"))
        
        action_type = tool_entry.get("action_type", "unknown")
        action = tool_entry.get("action", {})
        
        print(f"Action: {action_type}")
        if "command" in action:
            print(f"Command: {action['command'][:70]}...")
        
        # Check duplicates FIRST
        dup_check = service.check_for_duplicates(action_type, action)
        
        if dup_check["is_exact_duplicate"]:
            print(colored(f"   EXACT DUPLICATE - Skipping execution", "red", attrs=["bold"]))
            print(colored(f"     Previous: {dup_check['previous_tool_id']}", "red"))
            continue
        
        if dup_check["similar_tools"]:
            print(colored(f"   Found {len(dup_check['similar_tools'])} similar tools", "cyan"))
            for sim in dup_check["similar_tools"][:2]:
                # sim is now a dict, not an object
                print(colored(f"     - {sim['tool_id']} ({sim['similarity_score']:.1%} similar)", "cyan"))
        
        if dup_check["relevant_facts"]:
            print(colored(f"   Found {len(dup_check['relevant_facts'])} relevant facts", "blue"))
        
        # Execute tool
        tool_id, _ = service.add_tool_result(tool_entry, check_duplicates=False)
        
        # Generate structured summary (triggers fact extraction)
        summary = service.generate_structured_summary(tool_id)
        
        if summary:
            print(colored(f"   Summary generated", "green"))
            if summary.get('reusability', {}).get('can_be_reused'):
                print(colored(f"    Result is reusable!", "green", attrs=["bold"]))
    
    # ========== PHASE 4: Show Extracted Facts ==========
    print_section("PHASE 4: SHOWING EXTRACTED FACTS")
    
    fact_block = service.memory_manager.blocks.get("facts")
    if fact_block:
        facts_display = fact_block.get()
        if facts_display:
            print(colored(facts_display, "blue"))
        else:
            print(colored("(Fact extraction in progress...)", "yellow"))
    
    # ========== PHASE 5: Semantic Search Demonstration ==========
    print_section("PHASE 5: SEMANTIC SEARCH ACROSS MEMORY")
    
    print("Testing natural language queries...\n")
    
    test_queries = [
        "AWS IAM user permissions and groups",
        "database configuration files",
        "S3 bucket operations"
    ]
    
    for query in test_queries:
        print(colored(f"Query: \"{query}\"", "yellow", attrs=["bold"]))
        
        results = service.search_memory(query)
        
        # Vector matches
        vector_matches = results.get("vector_matches", [])
        if vector_matches:
            print(colored(f"  📊 Vector matches: {len(vector_matches)}", "green"))
            for match in vector_matches[:2]:
                print(f"     - {match.tool_id}: {match.tool_name} ({match.similarity_score:.1%})")
        
        # Fact matches
        fact_matches = results.get("fact_matches", [])
        if fact_matches:
            print(colored(f"   Fact matches: {len(fact_matches)}", "blue"))
            for fact in fact_matches[:2]:
                print(f"     - {fact.content[:80]}...")
        
        if not vector_matches and not fact_matches:
            print(colored("  (No matches found)", "yellow"))
        
        print()
    
    # ========== PHASE 6: Full Memory Context ==========
    print_section("PHASE 6: GENERATING FULL MEMORY CONTEXT")
    
    print("This is what would be provided to the agent in its prompt:\n")
    
    full_context = service.get_full_memory_context(max_tokens=5000)
    
    # Show abbreviated version
    lines = full_context.split('\n')
    if len(lines) > 40:
        print('\n'.join(lines[:20]))
        print(colored(f"\n... ({len(lines) - 40} lines omitted) ...\n", "yellow"))
        print('\n'.join(lines[-20:]))
    else:
        print(full_context)
    
    # ========== PHASE 7: Memory Summary ==========
    print_section("PHASE 7: MEMORY USAGE SUMMARY")
    
    memory_summary = service.get_memory_summary()
    
    print("Memory Block Status:")
    for block_name, info in memory_summary.items():
        if block_name == "metrics":
            continue
        
        print(f"\n{block_name.upper()}:")
        print(f"  Priority: {info['priority']}")
        print(f"  Token usage: {info['token_count']:,} / {info['max_tokens']:,}")
        print(f"  Within limit: {'' if info['within_limit'] else '✗'}")
        
        if 'fact_count' in info:
            print(f"  Facts extracted: {info['fact_count']}")
        elif 'tool_count' in info:
            print(f"  Tools in memory: {info['tool_count']}")
        elif 'info_count' in info:
            print(f"  Static entries: {info['info_count']}")
    
    # ========== PHASE 8: Performance Metrics ==========
    print_section("PHASE 8: PERFORMANCE METRICS")
    
    metrics = service.get_metrics()
    
    print("System Performance:")
    print(f"  Total tools processed: {metrics['total_tools']}")
    print(f"  Duplicates detected: {metrics['duplicates_detected']}")
    print(f"  Duplicate detection rate: {metrics['duplicate_rate']:.1f}%")
    print(f"  Semantic matches found: {metrics['semantic_matches_found']}")
    print(f"  Semantic hit rate: {metrics['semantic_hit_rate']:.1f}%")
    print(f"  Facts extracted: {metrics['facts_extracted']}")
    print(f"  Tokens saved: {metrics['tokens_saved']:,}")
    
    # ========== SUMMARY ==========
    print_section("DEMO COMPLETE - SUMMARY")
    
    print("This comprehensive memory system provides:")
    print()
    print(colored(" STATIC MEMORY", "green", attrs=["bold"]))
    print("  → Persistent project context")
    print("  → Configuration and integration info")
    print()
    print(colored(" WORKING MEMORY", "green", attrs=["bold"]))
    print("  → Current plan and goals")
    print("  → Active tasks and context notes")
    print()
    print(colored(" FACT EXTRACTION", "green", attrs=["bold"]))
    print("  → Automatic knowledge extraction")
    print("  → Structured, searchable facts")
    print()
    print(colored(" VECTOR MEMORY", "green", attrs=["bold"]))
    print("  → Semantic similarity search")
    print("  → Natural language queries")
    print()
    print(colored(" DEDUPLICATION", "green", attrs=["bold"]))
    print("  → Prevents redundant executions")
    print("  → Proactive warnings")
    print()
    
    # Cleanup
    print(f"Cleaning up workflow {workflow_id}...")
    service.reset_workflow()
    service.close()
    print(colored(" Cleanup complete\n", "green"))


if __name__ == "__main__":
    main()
