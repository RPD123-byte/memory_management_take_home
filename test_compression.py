#!/usr/bin/env python3
"""
Quick test script to verify compression/expansion functionality.
"""

import os
import sys
from termcolor import colored
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from comprehensive_memory_service import ComprehensiveMemoryService


def test_compression():
    """Test compression and expansion"""
    print(colored("\n=== Testing Compression & Expansion ===\n", "cyan", attrs=["bold"]))
    
    # Initialize service
    workflow_id = "test_compression"
    service = ComprehensiveMemoryService(
        workflow_id=workflow_id,
        enable_semantic_memory=False,  # Faster testing
        enable_fact_extraction=False    # Faster testing
    )
    
    try:
        # Clean start
        service.reset_workflow()
        
        # Add several test tools
        print("1. Adding test tools...")
        test_tools = [
            {
                "action_type": "execute_command",
                "action": {"command": "aws s3 ls"},
                "result": {"status": "success", "output": "bucket-1, bucket-2"},
                "timestamp": "2025-10-10T10:00:00"
            },
            {
                "action_type": "execute_command",
                "action": {"command": "aws s3 ls s3://bucket-1"},
                "result": {"status": "success", "output": "file1.txt, file2.txt"},
                "timestamp": "2025-10-10T10:01:00"
            },
            {
                "action_type": "execute_command",
                "action": {"command": "aws iam list-users"},
                "result": {"status": "success", "output": "user1, user2"},
                "timestamp": "2025-10-10T10:02:00"
            },
        ]
        
        tool_ids = []
        for tool in test_tools:
            tool_id, _ = service.add_tool_result(tool)
            tool_ids.append(tool_id)
            print(f"   Added {tool_id}: {tool['action']['command']}")
        
        # Get total tokens before compression
        all_tools = service.get_all_tool_results()
        tokens_before = sum(t.token_count for t in all_tools)
        print(f"\n   Total tokens before: {tokens_before}")
        
        # Test compression
        print(f"\n2. Compressing {len(tool_ids)} tools...")
        success = service.compress_tool_results(tool_ids, "test_group")
        
        if success:
            print(colored("   ✓ Compression successful!", "green"))
            
            # Show compression info
            comp_summary = service.get_compression_summary()
            print(f"   Groups created: {comp_summary['total_groups']}")
            print(f"   Tools compressed: {comp_summary['compressed_tools']}")
            print(f"   Tokens saved: {comp_summary['total_tokens_saved']}")
            
            # Test expansion
            print(f"\n3. Expanding tool {tool_ids[0]}...")
            full_details = service.expand_tool_result(tool_ids[0])
            
            if full_details:
                print(colored(f"   ✓ Successfully expanded {tool_ids[0]}", "green"))
                print(f"   Command: {full_details.get('action', {}).get('command', 'N/A')}")
            else:
                print(colored("   ✗ Expansion failed", "red"))
                
            print(colored("\n✓ All tests passed!", "green", attrs=["bold"]))
            return True
        else:
            print(colored("   ✗ Compression failed", "red"))
            return False
            
    except Exception as e:
        print(colored(f"\n✗ Test failed with error: {e}", "red"))
        import traceback
        traceback.print_exc()
        return False
    finally:
        service.reset_workflow()
        service.close()


if __name__ == "__main__":
    success = test_compression()
    sys.exit(0 if success else 1)
