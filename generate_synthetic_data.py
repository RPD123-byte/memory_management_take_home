"""
Comprehensive Synthetic Test Data Generator for Memory Management System.

Generates diverse tool execution scenarios including:
- Multiple action types (commands, files, API calls)
- Exact duplicates
- Similar tools (semantic matches)
- Fact extraction opportunities
- Edge cases
"""

import json
import random
from typing import List, Dict, Any
from datetime import datetime, timedelta


class SyntheticDataGenerator:
    """Generate synthetic tool execution data for testing"""
    
    def __init__(self, seed: int = 42):
        """Initialize generator with random seed"""
        random.seed(seed)
        
    def generate_dataset(self, num_tools: int = 100, 
                        duplicate_ratio: float = 0.2,
                        similar_ratio: float = 0.15) -> Dict[str, Any]:
        """
        Generate comprehensive test dataset.
        
        Args:
            num_tools: Total number of tool executions
            duplicate_ratio: Ratio of exact duplicates (0.2 = 20%)
            similar_ratio: Ratio of similar tools (0.15 = 15%)
            
        Returns:
            Dict with tools, duplicate_indices, similar_groups
        """
        tools = []
        duplicate_indices = []
        similar_groups = []
        
        # Calculate counts
        num_unique = int(num_tools * (1 - duplicate_ratio - similar_ratio))
        num_duplicates = int(num_tools * duplicate_ratio)
        num_similar = num_tools - num_unique - num_duplicates
        
        # Generate unique tools
        unique_tools = self._generate_unique_tools(num_unique)
        tools.extend(unique_tools)
        
        # Generate duplicates
        for _ in range(num_duplicates):
            source_idx = random.randint(0, len(tools) - 1)
            duplicate = self._create_exact_duplicate(tools[source_idx])
            duplicate_indices.append(len(tools))
            tools.append(duplicate)
        
        # Generate similar tools
        num_similar_groups = max(1, num_similar // 3)
        for _ in range(num_similar_groups):
            base_idx = random.randint(0, len(unique_tools) - 1)
            base_tool = unique_tools[base_idx]
            similar_group = [base_idx]
            
            # Generate 2-3 similar variants
            num_variants = min(3, num_similar - len([t for g in similar_groups for t in g]))
            for _ in range(num_variants):
                variant = self._create_similar_tool(base_tool)
                similar_group.append(len(tools))
                tools.append(variant)
            
            similar_groups.append(similar_group)
        
        # Shuffle to mix duplicates and similar tools
        indices = list(range(len(tools)))
        random.shuffle(indices)
        
        shuffled_tools = [tools[i] for i in indices]
        
        # Update duplicate/similar indices
        index_map = {old: new for new, old in enumerate(indices)}
        duplicate_indices = [index_map[idx] for idx in duplicate_indices if idx in index_map]
        similar_groups = [[index_map[idx] for idx in group if idx in index_map] for group in similar_groups]
        
        return {
            "tools": shuffled_tools,
            "metadata": {
                "total_tools": len(shuffled_tools),
                "unique_tools": num_unique,
                "exact_duplicates": len(duplicate_indices),
                "similar_groups": len(similar_groups),
                "duplicate_indices": duplicate_indices,
                "similar_groups": similar_groups,
                "generated_at": datetime.now().isoformat()
            }
        }
    
    def _generate_unique_tools(self, count: int) -> List[Dict[str, Any]]:
        """Generate unique tool executions"""
        tools = []
        
        # Distribution of tool types
        execute_command_count = int(count * 0.5)
        read_file_count = int(count * 0.2)
        write_file_count = int(count * 0.15)
        api_call_count = int(count * 0.15)
        
        # Generate execute_command tools
        tools.extend([self._generate_execute_command() for _ in range(execute_command_count)])
        
        # Generate read_file tools
        tools.extend([self._generate_read_file() for _ in range(read_file_count)])
        
        # Generate write_file tools
        tools.extend([self._generate_write_file() for _ in range(write_file_count)])
        
        # Generate api_call tools
        tools.extend([self._generate_api_call() for _ in range(api_call_count)])
        
        # Pad if needed
        while len(tools) < count:
            tools.append(random.choice([
                self._generate_execute_command,
                self._generate_read_file,
                self._generate_write_file,
                self._generate_api_call
            ])())
        
        return tools[:count]
    
    def _generate_execute_command(self) -> Dict[str, Any]:
        """Generate execute_command tool"""
        command_templates = [
            # AWS commands
            ("aws s3 ls s3://{bucket}", {"bucket": self._random_bucket()}),
            ("aws s3 ls s3://{bucket} --recursive", {"bucket": self._random_bucket()}),
            ("aws iam list-users", {}),
            ("aws iam list-groups-for-user --user-name {user}", {"user": self._random_user()}),
            ("aws iam get-user --user-name {user}", {"user": self._random_user()}),
            ("aws iam list-attached-user-policies --user-name {user}", {"user": self._random_user()}),
            ("aws ec2 describe-instances", {}),
            ("aws ec2 describe-instances --instance-ids {id}", {"id": self._random_instance_id()}),
            ("aws rds describe-db-instances", {}),
            ("aws lambda list-functions", {}),
            
            # Kubernetes commands
            ("kubectl get pods", {}),
            ("kubectl get pods -n {namespace}", {"namespace": self._random_namespace()}),
            ("kubectl describe pod {pod}", {"pod": self._random_pod()}),
            ("kubectl get services", {}),
            ("kubectl get deployments", {}),
            
            # Git commands
            ("git status", {}),
            ("git log --oneline -n 10", {}),
            ("git branch", {}),
            ("git diff", {}),
            
            # Database commands
            ("psql -c 'SELECT * FROM {table} LIMIT 10'", {"table": self._random_table()}),
            ("mysql -e 'SHOW TABLES'", {}),
            
            # File system commands
            ("ls -la {path}", {"path": self._random_path()}),
            ("find {path} -name '*.py'", {"path": self._random_path()}),
        ]
        
        template, params = random.choice(command_templates)
        command = template.format(**params)
        
        return {
            "action_type": "execute_command",
            "action": {"command": command},
            "result": self._generate_command_result(command),
            "timestamp": self._random_timestamp()
        }
    
    def _generate_read_file(self) -> Dict[str, Any]:
        """Generate read_file tool"""
        file_paths = [
            "app/config.yaml",
            "terraform/main.tf",
            "src/database/schema.sql",
            "docker-compose.yml",
            "requirements.txt",
            "package.json",
            ".env",
            "README.md",
            "Dockerfile",
            "k8s/deployment.yaml"
        ]
        
        file_path = random.choice(file_paths)
        
        return {
            "action_type": "read_file",
            "action": {"file_path": file_path},
            "result": {
                "status": "success",
                "content": self._generate_file_content(file_path),
                "lines": random.randint(10, 200)
            },
            "timestamp": self._random_timestamp()
        }
    
    def _generate_write_file(self) -> Dict[str, Any]:
        """Generate write_file tool"""
        file_paths = [
            "app/models.py",
            "terraform/variables.tf",
            "src/api/routes.py",
            "config/settings.json",
            "scripts/deploy.sh",
            "tests/test_api.py"
        ]
        
        file_path = random.choice(file_paths)
        
        return {
            "action_type": "write_file",
            "action": {
                "file_path": file_path,
                "content": f"# Generated content for {file_path}\n..."
            },
            "result": {
                "status": "success",
                "bytes_written": random.randint(100, 5000)
            },
            "timestamp": self._random_timestamp()
        }
    
    def _generate_api_call(self) -> Dict[str, Any]:
        """Generate api_call tool"""
        api_endpoints = [
            ("/api/users", "GET", {}),
            ("/api/users/{user_id}", "GET", {"user_id": random.randint(1, 100)}),
            ("/api/posts", "GET", {}),
            ("/api/posts", "POST", {"title": "New Post", "body": "Content"}),
            ("/api/auth/login", "POST", {"username": "admin", "password": "***"}),
            ("/health", "GET", {}),
            ("/metrics", "GET", {}),
        ]
        
        endpoint, method, params = random.choice(api_endpoints)
        endpoint = endpoint.format(**params)
        
        return {
            "action_type": "api_call",
            "action": {
                "endpoint": endpoint,
                "method": method,
                "params": params
            },
            "result": {
                "status": "success",
                "status_code": 200,
                "response": self._generate_api_response(endpoint, method)
            },
            "timestamp": self._random_timestamp()
        }
    
    def _create_exact_duplicate(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """Create exact duplicate with new timestamp"""
        duplicate = tool.copy()
        duplicate["timestamp"] = self._random_timestamp()
        return duplicate
    
    def _create_similar_tool(self, base_tool: Dict[str, Any]) -> Dict[str, Any]:
        """Create semantically similar tool"""
        similar = base_tool.copy()
        action_type = similar["action_type"]
        
        if action_type == "execute_command":
            command = similar["action"]["command"]
            
            # Similar command variations
            if "aws s3 ls" in command:
                similar["action"]["command"] = command + " --region us-east-1"
            elif "kubectl get pods" in command:
                similar["action"]["command"] = command + " --all-namespaces"
            elif "git" in command:
                similar["action"]["command"] = command + " --verbose"
            else:
                similar["action"]["command"] = command + " --output json"
                
        elif action_type == "read_file":
            # Similar file path
            path = similar["action"]["file_path"]
            base_name = path.rsplit('.', 1)[0]
            extension = path.rsplit('.', 1)[1] if '.' in path else ''
            similar["action"]["file_path"] = f"{base_name}_v2.{extension}" if extension else f"{base_name}_v2"
        
        similar["timestamp"] = self._random_timestamp()
        similar["result"] = self._generate_similar_result(base_tool["result"])
        
        return similar
    
    def _generate_command_result(self, command: str) -> Dict[str, Any]:
        """Generate realistic command result"""
        if "aws s3 ls" in command:
            return {
                "status": "success",
                "output": f"2025-01-01 12:00:00    bucket-{random.randint(1, 100)}\n" * random.randint(1, 5),
                "exit_code": 0
            }
        elif "aws iam list-users" in command:
            users = [f"user-{i}" for i in range(random.randint(1, 5))]
            return {
                "status": "success",
                "output": json.dumps({"Users": [{"UserName": u, "Arn": f"arn:aws:iam::123456:user/{u}"} for u in users]}),
                "exit_code": 0
            }
        elif "kubectl get pods" in command:
            return {
                "status": "success",
                "output": f"pod-{random.randint(1, 10)}    Running    1/1\n" * random.randint(1, 5),
                "exit_code": 0
            }
        else:
            return {
                "status": "success",
                "output": f"Command output for: {command}",
                "exit_code": 0
            }
    
    def _generate_file_content(self, file_path: str) -> str:
        """Generate realistic file content"""
        if file_path.endswith(".yaml") or file_path.endswith(".yml"):
            return "version: '3.8'\nservices:\n  app:\n    image: myapp:latest\n    ports:\n      - 8000:8000"
        elif file_path.endswith(".json"):
            return json.dumps({"key": "value", "config": {"setting": True}}, indent=2)
        elif file_path.endswith(".py"):
            return "# Python code\ndef main():\n    pass\n\nif __name__ == '__main__':\n    main()"
        else:
            return f"Content of {file_path}\n..."
    
    def _generate_api_response(self, endpoint: str, method: str) -> Dict[str, Any]:
        """Generate API response"""
        if "/users" in endpoint:
            return {"id": random.randint(1, 100), "name": f"User{random.randint(1, 100)}"}
        elif "/posts" in endpoint:
            return {"id": random.randint(1, 1000), "title": "Post Title"}
        else:
            return {"status": "ok"}
    
    def _generate_similar_result(self, base_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate similar but not identical result"""
        similar = base_result.copy()
        if "output" in similar:
            similar["output"] = similar["output"] + " (variant)"
        return similar
    
    def _random_timestamp(self) -> str:
        """Generate random timestamp"""
        base_time = datetime(2025, 1, 1)
        random_time = base_time + timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        return random_time.isoformat()
    
    # Helper methods for random data
    def _random_bucket(self) -> str:
        return f"my-bucket-{random.randint(1, 10)}"
    
    def _random_user(self) -> str:
        return random.choice(["admin", "user1", "developer", "operator"])
    
    def _random_instance_id(self) -> str:
        return f"i-{random.randint(100000, 999999)}"
    
    def _random_namespace(self) -> str:
        return random.choice(["default", "production", "staging", "monitoring"])
    
    def _random_pod(self) -> str:
        return f"app-pod-{random.randint(1, 100)}"
    
    def _random_table(self) -> str:
        return random.choice(["users", "posts", "orders", "products", "logs"])
    
    def _random_path(self) -> str:
        return random.choice(["/app", "/src", "/data", "/config", "/var/log"])


def main():
    """Generate and save synthetic test data"""
    generator = SyntheticDataGenerator(seed=42)
    
    # Generate quick test dataset
    print("Generating synthetic test data...")
    dataset_quick = generator.generate_dataset(
        num_tools=20,
        duplicate_ratio=0.25,
        similar_ratio=0.15
    )
    
    # Save quick test dataset
    quick_path = "data/synthetic_test_quick.json"
    with open(quick_path, 'w') as f:
        json.dump(dataset_quick, f, indent=2)
    
    print(f"\n✓ Generated QUICK test dataset (20 tools):")
    print(f"  Total tools: {dataset_quick['metadata']['total_tools']}")
    print(f"  Unique tools: {dataset_quick['metadata']['unique_tools']}")
    print(f"  Exact duplicates: {dataset_quick['metadata']['exact_duplicates']}")
    print(f"  Similar groups: {dataset_quick['metadata']['similar_groups']}")
    print(f"  Saved to: {quick_path}")
    print(f"  Estimated test time: ~2-3 minutes")
    
    # Generate FULL dataset (100 tools = ~20min test time)
    print("\nGenerating full test data...")
    dataset = generator.generate_dataset(
        num_tools=100,
        duplicate_ratio=0.2,
        similar_ratio=0.15
    )
    
    # Save full dataset with metadata
    output_path = "data/synthetic_test_data.json"
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n✓ Generated FULL test dataset (100 tools):")
    print(f"  Total tools: {dataset['metadata']['total_tools']}")
    print(f"  Unique tools: {dataset['metadata']['unique_tools']}")
    print(f"  Exact duplicates: {dataset['metadata']['exact_duplicates']}")
    print(f"  Similar groups: {dataset['metadata']['similar_groups']}")
    print(f"  Saved to: {output_path}")
    print(f"  Estimated test time: ~20 minutes")
    
    print("\n" + "="*60)
    print("USAGE:")
    print("  Quick iteration (2-3 min): --test-data data/synthetic_test_quick.json")
    print("  Full validation (20 min):  --test-data data/synthetic_test_data.json")
    print("="*60)


if __name__ == "__main__":
    main()
