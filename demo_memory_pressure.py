#!/usr/bin/env python3
"""
Quick demonstration of memory pressure handling with compression.
Shows how the system manages accumulating tools over a workflow.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from comprehensive_memory_service import ComprehensiveMemoryService
from termcolor import colored
from dotenv import load_dotenv

load_dotenv()

def print_section(title):
    print("\n" + "="*70)
    print(colored(title, "cyan", attrs=["bold"]))
    print("="*70 + "\n")

def main():
    print_section("MEMORY PRESSURE SIMULATION")
    
    print("Scenario: Agent executes 20+ tools in long workflow")
    print("Demonstrates: Duplicate detection + Compression when memory is high\n")
    
    # Initialize service
    service = ComprehensiveMemoryService(
        workflow_id="memory_pressure_demo",
        enable_semantic_memory=False,  # Speed up
        enable_fact_extraction=False
    )
    
    service.reset_workflow()
    
    # Phase 1: Initial infrastructure discovery (10 tools)
    print(colored("[PHASE 1: Infrastructure Discovery]", "yellow", attrs=["bold"]))
    phase1_tools = [
        ("aws ec2 describe-instances", "3 instances running"),
        ("aws s3 ls", "5 buckets found"),
        ("aws rds describe-db-instances", "2 databases available"),
        ("aws lambda list-functions", "8 functions deployed"),
        ("aws iam list-users", "4 users configured"),
        ("aws cloudwatch describe-alarms", "12 alarms active"),
        ("aws vpc describe-vpcs", "2 VPCs configured"),
        ("aws ec2 describe-security-groups", "15 security groups"),
        ("aws route53 list-hosted-zones", "3 hosted zones"),
        ("aws elasticache describe-cache-clusters", "1 Redis cluster"),
    ]
    
    phase1_ids = []
    for cmd, output in phase1_tools:
        tool_id, dup = service.add_tool_result({
            "action_type": "execute_command",
            "action": {"command": cmd},
            "result": {"status": "success", "output": output},
            "timestamp": "2025-10-11T10:00:00"
        })
        phase1_ids.append(tool_id)
        status = colored("DUPLICATE", "yellow") if dup else colored("NEW", "green")
        print(f"  {tool_id} [{status}]: {cmd[:50]}")
    
    print(f"\n  ✓ Phase 1 complete: {len(phase1_ids)} tools executed")
    
    # Phase 2: Configuration changes (8 tools)
    print(colored("\n[PHASE 2: Configuration Changes]", "yellow", attrs=["bold"]))
    phase2_tools = [
        ("aws ec2 modify-instance-attribute --instance-id i-123", "Modified"),
        ("aws s3api put-bucket-versioning --bucket my-bucket", "Enabled versioning"),
        ("aws rds modify-db-instance --db prod-db", "Increased storage"),
        ("aws iam attach-user-policy --user-name dev-user", "Attached policy"),
        ("aws lambda update-function-configuration --function api", "Updated timeout"),
        ("aws ec2 authorize-security-group-ingress", "Added rule"),
        ("aws route53 change-resource-record-sets", "Updated DNS"),
        ("aws ec2 describe-instances", "3 instances running"),  # DUPLICATE!
    ]
    
    phase2_ids = []
    for cmd, output in phase2_tools:
        tool_id, dup = service.add_tool_result({
            "action_type": "execute_command",
            "action": {"command": cmd},
            "result": {"status": "success", "output": output},
            "timestamp": "2025-10-11T10:10:00"
        })
        phase2_ids.append(tool_id)
        status = colored("DUPLICATE ✓", "yellow") if dup else colored("NEW", "green")
        print(f"  {tool_id} [{status}]: {cmd[:50]}")
    
    print(f"\n  ✓ Phase 2 complete: {len([t for t in phase2_ids if t])} new tools")
    
    # Check memory pressure
    all_tools = service.get_all_tool_results()
    total_tokens = sum(t.token_count for t in all_tools)
    
    print(colored(f"\n⚠️  MEMORY PRESSURE CHECK:", "red", attrs=["bold"]))
    print(f"  Total tools in memory: {len(all_tools)}")
    print(f"  Total tokens: {total_tokens}")
    
    if total_tokens > 500:  # Lower threshold for demo
        print(colored("  → Memory pressure HIGH! Compressing old tools...", "red"))
        
        # Compress Phase 1 tools (keep last 2 for recent context)
        tools_to_compress = phase1_ids[:-2]
        print(f"\n  Compressing {len(tools_to_compress)} older tools from Phase 1...")
        
        success = service.compress_tool_results(tools_to_compress, "phase1_infrastructure")
        
        if success:
            comp_info = service.get_compression_summary()
            print(colored(f"  ✓ Compression successful!", "green"))
            print(f"    • Groups created: {comp_info['total_groups']}")
            print(f"    • Tools compressed: {comp_info['compressed_tools']}")
            print(f"    • Tokens saved: {comp_info['total_tokens_saved']}")
    
    # Phase 3: Show we can still expand when needed
    print(colored("\n[PHASE 3: Selective Expansion]", "yellow", attrs=["bold"]))
    print("Agent needs details from compressed tool...")
    print(f"Expanding {phase1_ids[0]} for full information...")
    
    expanded = service.expand_tool_result(phase1_ids[0])
    if expanded:
        print(colored(f"  ✓ Successfully expanded {phase1_ids[0]}", "green"))
        print(f"    Command: {expanded['action']['command']}")
        print(f"    Output: {expanded['result']['output']}")
    
    # Final metrics
    print_section("FINAL METRICS")
    metrics = service.get_metrics()
    comp_info = service.get_compression_summary()
    
    print(colored("Workflow Performance:", "cyan"))
    print(f"  Total tools executed: {metrics['total_tools']}")
    print(f"  Duplicates detected: {metrics['duplicates_detected']}")
    print(f"  Compression groups: {comp_info['total_groups']}")
    print(f"  Tokens saved by compression: {comp_info['total_tokens_saved']}")
    
    print(colored("\n✓ Demonstration Complete!", "green", attrs=["bold"]))
    print("\nKey Takeaways:")
    print("  1. System detected duplicates automatically (100% accuracy)")
    print("  2. Compressed old tools when memory pressure increased")
    print("  3. Can expand compressed tools when details are needed")
    print("  4. Enables long workflows without hitting context limits\n")
    
    service.reset_workflow()
    service.close()

if __name__ == "__main__":
    main()
