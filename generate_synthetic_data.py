import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
import random
import argparse
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from llm_service import LLMService, Message

# tool types from agent_prompt
TOOL_TYPES = [
    "execute_command",
    "modify_code", 
    "create_file",
    "read_file_contents",
    "query_codebase",
    "search_documentation",
    "search_internet",
    "retrieve_integration_methods",
    "call_integration_method",
    "compress_tool_results",
    "get_tool_result",
    "ask_human_question",
    "request_human_intervention"
]

DEVOPS_SCENARIOS = [
    "AWS infrastructure setup",
    "Kubernetes deployment configuration",
    "CI/CD pipeline setup",
    "Database migration",
    "Terraform module creation",
    "Security policy implementation",
    "Monitoring and alerting setup",
    "Container orchestration",
    "API deployment",
    "Load balancer configuration"
]

class SyntheticDataGenerator:
    def __init__(self, api_key: str):
        self.llm_service = LLMService(api_key=api_key)
        self.start_time = datetime.now()
        
    def generate_tool_execution(self, action_type: str, scenario: str, index: int) -> Dict[str, Any]:
        """Generate a single tool execution entry using LLM."""
        
        # Create a prompt for the LLM to generate realistic data
        prompt = f"""Generate a realistic tool execution entry for a DevOps workflow.

Scenario: {scenario}
Action Type: {action_type}
Index in workflow: {index}

The entry should follow this exact JSON structure:
{{
    "timestamp": "ISO format timestamp",
    "action_type": "{action_type}",
    "action": {{
        // Action-specific fields based on action_type
    }},
    "result": {{
        "status": "success" or "error",
        "output": "Realistic output string",
        "error": null or "Error message if status is error"
    }},
    "context": {{
        "reasoning": "Why this action was taken",
        "description": "Brief description of what happened"
    }}
}}

Action-specific fields:
- execute_command: {{"command": "shell command"}}
- modify_code: {{"code": "code snippet", "instructions": "what to do", "files": ["file1.py"]}}
- create_file: {{"file_path": "path/to/file", "content": "file content"}}
- read_file_contents: {{"file_path": "path/to/file"}}
- query_codebase: {{"query": "search query"}}
- search_documentation: {{"query": "search query", "integration": "terraform", "provider_version": "aws v5.0.0"}}
- call_integration_method: {{"integration_name": "aws", "method": "describe_instances", "parameters": {{}}}}

Make the output realistic and relevant to the scenario. Include actual command outputs, error messages, or file contents as appropriate.
Only return the JSON object, no other text."""

        messages = [Message(role="user", content=prompt)]
        
        try:
            response = self.llm_service.generate(messages, json_mode=True)
            entry = json.loads(response)
            
            entry["timestamp"] = (self.start_time + timedelta(minutes=index)).isoformat() + "Z"

            print(f"Generated entry: {entry}")
            
            return entry
        except Exception as e:
            print(f"Error generating entry: {e}")
            sys.exit(1)
    
    def generate_workflow(self, num_commands: int, scenario: str = "") -> List[Dict[str, Any]]:
        """Generate a complete workflow with the specified number of commands."""
        if not scenario:
            scenario = random.choice(DEVOPS_SCENARIOS)
        
        print(f"Generating workflow for scenario: {scenario}")
        
        workflow = []
        
        # percents for each tool
        tool_distribution = {
            "execute_command": 0.4,
            "read_file_contents": 0.15,
            "modify_code": 0.15,
            "query_codebase": 0.1,
            "create_file": 0.1,
            "search_documentation": 0.05,
            "other": 0.05
        }
        
        def choose_tool_from_dist():
            rand = random.random()
            cumulative = 0
            for tool_type, prob in tool_distribution.items():
                cumulative += prob
                if rand <= cumulative:
                    if tool_type == "other":
                        return random.choice(TOOL_TYPES[6:])  # Other tools
                    else:
                        return tool_type
            return "execute_command"
        
        for i in range(num_commands):
            chosen_type = choose_tool_from_dist()
            print(f"Generating entry {i+1}/{num_commands}: {chosen_type}")
            entry = self.generate_tool_execution(chosen_type, scenario, i)
            workflow.append(entry)
        
        return workflow


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic tool execution traces")
    parser.add_argument("--num-commands", type=int, required=True, 
                        help="Number of commands to generate")
    parser.add_argument("--scenario", type=str, default="",
                        help="Specific scenario (optional, will be randomly chosen if not provided)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: examples/synthetic_trace_<timestamp>.json)")
    
    args = parser.parse_args()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return 1
    
    generator = SyntheticDataGenerator(api_key)
    
    workflow = generator.generate_workflow(args.num_commands, args.scenario)
    
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"examples/synthetic_trace_{timestamp}.json")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(workflow, f, indent=2)
    
    print(f"\nGenerated {len(workflow)} tool executions")
    print(f"Saved to: {output_path}")
    
    action_types = {}
    for entry in workflow:
        action_type = entry["action_type"]
        action_types[action_type] = action_types.get(action_type, 0) + 1
    
    print("\nSummary by action type:")
    for action_type, count in sorted(action_types.items()):
        print(f"  {action_type}: {count}")


if __name__ == "__main__":
    main() 