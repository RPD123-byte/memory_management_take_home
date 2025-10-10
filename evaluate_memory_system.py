"""
Evaluation script to compare memory management systems.

This script evaluates and compares the performance of the original vs enhanced
memory management system across multiple metrics:
- Redundant tool call detection rate
- Token usage efficiency  
- Recall accuracy for similar tools
- Latency (response time)
"""

import json
import time
import os
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics
from termcolor import colored
from tabulate import tabulate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from knowledge_graph_service import KnowledgeGraphService
from comprehensive_memory_service import ComprehensiveMemoryService


@dataclass
class EvaluationMetrics:
    """Metrics for evaluating memory management performance"""
    system_name: str
    total_tools: int
    duplicates_detected: int
    duplicate_detection_rate: float
    false_positives: int
    false_negatives: int
    total_tokens: int
    avg_tokens_per_tool: float
    tokens_saved: int
    token_savings_rate: float
    semantic_matches_found: int
    semantic_recall_rate: float
    avg_add_latency_ms: float
    avg_summary_latency_ms: float
    avg_search_latency_ms: float


class MemoryManagementEvaluator:
    """Evaluator for memory management systems"""
    
    def __init__(self, api_key: str = None):
        """Initialize evaluator with API key"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        
    def load_test_data(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load test tool executions.
        
        Format:
        - Full dataset: {"tools": [...], "metadata": {...}}
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Handle both formats
        if isinstance(data, dict) and "tools" in data:
            # Full dataset with metadata
            return data["tools"]
        else:
            raise ValueError(f"Invalid data format in {filepath}")
    
    def generate_synthetic_duplicates(self, original_data: List[Dict[str, Any]], 
                                     duplicate_ratio: float = 0.3) -> Tuple[List[Dict[str, Any]], List[int]]:
        """
        Generate test dataset with synthetic duplicates.
        
        Args:
            original_data: Original tool executions
            duplicate_ratio: Ratio of duplicates to inject (0.3 = 30%)
            
        Returns:
            Tuple of (augmented_data, duplicate_indices)
        """
        import random
        
        augmented = list(original_data)
        duplicate_indices = []
        
        # Number of duplicates to add
        num_duplicates = int(len(original_data) * duplicate_ratio)
        
        for _ in range(num_duplicates):
            # Randomly select a tool to duplicate
            source_idx = random.randint(0, len(original_data) - 1)
            duplicate = original_data[source_idx].copy()
            
            # Add to dataset
            augmented.append(duplicate)
            duplicate_indices.append(len(augmented) - 1)
        
        return augmented, duplicate_indices
    
    def generate_similar_tools(self, base_tool: Dict[str, Any], 
                              num_variants: int = 3) -> List[Dict[str, Any]]:
        """
        Generate semantically similar tool calls for recall testing.
        
        Args:
            base_tool: Base tool execution
            num_variants: Number of similar variants to generate
            
        Returns:
            List of similar tool executions
        """
        variants = []
        
        action_type = base_tool.get("action_type", "")
        action = base_tool.get("action", {})
        
        if action_type == "execute_command" and "command" in action:
            command = action["command"]
            
            # Generate similar commands
            if "aws" in command.lower():
                # Similar AWS commands
                if "iam" in command.lower():
                    similar_commands = [
                        command.replace("list-groups-for-user", "get-user"),
                        command.replace("list-groups-for-user", "list-attached-user-policies"),
                        command + " --output json"
                    ]
                elif "s3" in command.lower():
                    similar_commands = [
                        command.replace("ls --recursive", "ls"),
                        command + " --region us-east-1",
                        command.replace("s3://my-bucket", "s3://my-bucket/folder")
                    ]
                else:
                    similar_commands = [command + f" --variant-{i}" for i in range(num_variants)]
            else:
                similar_commands = [command + f" --option-{i}" for i in range(num_variants)]
            
            for cmd in similar_commands[:num_variants]:
                variant = base_tool.copy()
                variant["action"] = {"command": cmd}
                variants.append(variant)
        
        elif action_type == "create_file" and "file_path" in action:
            file_path = action["file_path"]
            
            # Similar file operations
            base_dir = os.path.dirname(file_path)
            base_name = os.path.basename(file_path)
            
            similar_paths = [
                f"{base_dir}/test_{base_name}",
                file_path.replace(".py", "_v2.py"),
                f"{base_dir}/utils/{base_name}"
            ]
            
            for path in similar_paths[:num_variants]:
                variant = base_tool.copy()
                variant["action"] = {**action, "file_path": path}
                variants.append(variant)
        
        return variants
    
    def evaluate_system(self, system_name: str, service, test_data: List[Dict[str, Any]],
                       true_duplicate_indices: List[int]) -> EvaluationMetrics:
        """
        Evaluate a memory management system.
        
        Args:
            system_name: Name of the system being evaluated
            service: Knowledge graph service instance
            test_data: Test tool executions
            true_duplicate_indices: Indices of actual duplicates
            
        Returns:
            EvaluationMetrics with evaluation results
        """
        print(colored(f"\n{'='*60}", "blue"))
        print(colored(f"Evaluating {system_name}", "blue", attrs=["bold"]))
        print(colored(f"{'='*60}", "blue"))
        
        # Metrics tracking
        total_tokens = 0
        add_latencies = []
        summary_latencies = []
        search_latencies = []
        detected_duplicates = []
        semantic_matches = []
        
        # Process each tool
        for idx, tool_entry in enumerate(test_data):
            print(f"Processing tool {idx + 1}/{len(test_data)}...", end="\r")
            
            # Measure add latency
            start_time = time.time()
            
            if isinstance(service, ComprehensiveMemoryService):
                tool_id, dup_info = service.add_tool_result(tool_entry, check_duplicates=True)
                
                # Track duplicate detection
                if dup_info and dup_info["is_exact_duplicate"]:
                    detected_duplicates.append(idx)
                
                # Track semantic matches
                if dup_info and dup_info["similar_tools"]:
                    semantic_matches.append(idx)
            else:
                tool_id = service.add_tool_result(tool_entry)
            
            add_latency = (time.time() - start_time) * 1000  # Convert to ms
            add_latencies.append(add_latency)
            
            # Measure summary generation latency
            start_time = time.time()
            
            if isinstance(service, ComprehensiveMemoryService):
                service.generate_structured_summary(tool_id)
            else:
                service.generate_summary(tool_id)
            
            summary_latency = (time.time() - start_time) * 1000
            summary_latencies.append(summary_latency)
            
            # Count tokens
            tool_results = service.get_all_tool_results()
            total_tokens = sum(t.token_count for t in tool_results)
        
        print()  # New line after progress
        
        # Wait for all async fact extractions to complete
        if isinstance(service, ComprehensiveMemoryService):
            service.wait_for_pending_extractions()
        
        # Calculate duplicate detection metrics
        true_positives = len(set(detected_duplicates) & set(true_duplicate_indices))
        false_positives = len(set(detected_duplicates) - set(true_duplicate_indices))
        false_negatives = len(set(true_duplicate_indices) - set(detected_duplicates))
        
        duplicate_detection_rate = (
            true_positives / len(true_duplicate_indices) if true_duplicate_indices else 0.0
        )
        
        # Test semantic search and hierarchical memory if supported
        if isinstance(service, ComprehensiveMemoryService):
            print("\nTesting semantic search and memory retrieval...")
            test_queries = [
                "AWS IAM user permissions",
                "database configuration",
                "S3 bucket listing"
            ]
            
            for query in test_queries:
                start_time = time.time()
                results = service.search_memory(query)
                search_latency = (time.time() - start_time) * 1000
                search_latencies.append(search_latency)
                
                vector_results = results.get("vector_search", [])
                if vector_results:
                    print(f"  Query: '{query}' -> {len(vector_results)} vector results")
        
        # Get additional metrics from comprehensive system
        if isinstance(service, ComprehensiveMemoryService):
            system_metrics = service.get_metrics()
            memory_summary = service.get_memory_summary()
        else:
            system_metrics = {}
            memory_summary = {}
        
        # Calculate final metrics
        metrics = EvaluationMetrics(
            system_name=system_name,
            total_tools=len(test_data),
            duplicates_detected=len(detected_duplicates),
            duplicate_detection_rate=duplicate_detection_rate * 100,
            false_positives=false_positives,
            false_negatives=false_negatives,
            total_tokens=total_tokens,
            avg_tokens_per_tool=total_tokens / len(test_data) if test_data else 0,
            tokens_saved=system_metrics.get("tokens_saved", 0),
            token_savings_rate=0,  # Will calculate relative to baseline
            semantic_matches_found=len(semantic_matches),
            semantic_recall_rate=(len(semantic_matches) / len(test_data) * 100) if test_data else 0,
            avg_add_latency_ms=statistics.mean(add_latencies) if add_latencies else 0,
            avg_summary_latency_ms=statistics.mean(summary_latencies) if summary_latencies else 0,
            avg_search_latency_ms=statistics.mean(search_latencies) if search_latencies else 0
        )
        
        return metrics
    
    def run_comparison(self, test_data_path: str, output_path: str = None):
        """
        Run full comparison between original and enhanced systems.
        
        Args:
            test_data_path: Path to test data JSON file
            output_path: Optional path to save results
        """
        print(colored("\n" + "="*80, "magenta", attrs=["bold"]))
        print(colored("MEMORY MANAGEMENT SYSTEM EVALUATION", "magenta", attrs=["bold"]))
        print(colored("="*80 + "\n", "magenta", attrs=["bold"]))
        
        # Load test data
        print("Loading test data...")
        original_data = self.load_test_data(test_data_path)
        
        # Generate test dataset with duplicates
        print("Generating test dataset with synthetic duplicates...")
        test_data, duplicate_indices = self.generate_synthetic_duplicates(
            original_data, duplicate_ratio=0.3
        )
        print(f"  Total tools: {len(test_data)}")
        print(f"  True duplicates: {len(duplicate_indices)}")
        
        # Evaluate original system
        print(colored("\n" + "-"*80, "yellow"))
        workflow_id_1 = f"eval_original_{int(time.time())}"
        original_service = KnowledgeGraphService(workflow_id_1, api_key=self.api_key)
        
        try:
            original_metrics = self.evaluate_system(
                "Original System",
                original_service,
                test_data,
                duplicate_indices
            )
        finally:
            original_service.close()
        
        # Evaluate comprehensive system
        print(colored("\n" + "-"*80, "yellow"))
        workflow_id_2 = f"eval_comprehensive_{int(time.time())}"
        comprehensive_service = ComprehensiveMemoryService(
            workflow_id_2, 
            api_key=self.api_key,
            enable_semantic_memory=True,
            enable_fact_extraction=True
        )
        
        try:
            comprehensive_metrics = self.evaluate_system(
                "Comprehensive System",
                comprehensive_service,
                test_data,
                duplicate_indices
            )
        finally:
            comprehensive_service.close()
        
        # Calculate improvements
        token_improvement = (
            (original_metrics.total_tokens - comprehensive_metrics.total_tokens) / 
            original_metrics.total_tokens * 100
            if original_metrics.total_tokens > 0 else 0
        )
        
        latency_improvement = (
            (original_metrics.avg_add_latency_ms - comprehensive_metrics.avg_add_latency_ms) /
            original_metrics.avg_add_latency_ms * 100
            if original_metrics.avg_add_latency_ms > 0 else 0
        )
        
        # Display comparison
        self._display_comparison(original_metrics, comprehensive_metrics, token_improvement, latency_improvement)
        
        # Save results if output path provided
        if output_path:
            # Add timestamp to output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(output_path)[0]
            timestamped_output = f"{base_name}_{timestamp}.json"
            
            results = {
                "timestamp": timestamp,
                "evaluation_date": datetime.now().isoformat(),
                "original": asdict(original_metrics),
                "comprehensive": asdict(comprehensive_metrics),
                "improvements": {
                    "token_reduction_percent": token_improvement,
                    "latency_improvement_percent": latency_improvement,
                    "duplicate_detection_improvement": (
                        comprehensive_metrics.duplicate_detection_rate - 
                        original_metrics.duplicate_detection_rate
                    ),
                    "facts_extracted": comprehensive_metrics.semantic_matches_found
                },
                "test_config": {
                    "total_tools": len(test_data),
                    "true_duplicates": len(duplicate_indices),
                    "test_data_source": test_data_path
                }
            }
            
            # Save timestamped version
            with open(timestamped_output, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Also save to base output path (latest)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(colored(f"\nResults saved to:", "green"))
            print(colored(f"  - Latest: {output_path}", "green"))
            print(colored(f"  - Timestamped: {timestamped_output}", "green"))
        
        return original_metrics, comprehensive_metrics
    
    def _display_comparison(self, original: EvaluationMetrics, comprehensive: EvaluationMetrics,
                           token_improvement: float, latency_improvement: float):
        """Display comparison table"""
        print(colored("\n" + "="*80, "magenta", attrs=["bold"]))
        print(colored("EVALUATION RESULTS COMPARISON", "magenta", attrs=["bold"]))
        print(colored("="*80 + "\n", "magenta", attrs=["bold"]))
        
        # Comparison table
        comparison_data = [
            ["Metric", "Original", "Comprehensive", "Improvement"],
            ["-" * 30, "-" * 15, "-" * 15, "-" * 15],
            [
                "Duplicate Detection Rate",
                f"{original.duplicate_detection_rate:.1f}%",
                f"{comprehensive.duplicate_detection_rate:.1f}%",
                colored(f"+{comprehensive.duplicate_detection_rate - original.duplicate_detection_rate:.1f}%", "green")
            ],
            [
                "False Positives",
                str(original.false_positives),
                str(comprehensive.false_positives),
                f"{comprehensive.false_positives - original.false_positives:+d}"
            ],
            [
                "False Negatives",
                str(original.false_negatives),
                str(comprehensive.false_negatives),
                colored(f"{comprehensive.false_negatives - original.false_negatives:+d}", "green" if comprehensive.false_negatives < original.false_negatives else "red")
            ],
            [
                "Total Tokens",
                f"{original.total_tokens:,}",
                f"{comprehensive.total_tokens:,}",
                colored(f"{token_improvement:+.1f}%", "green" if token_improvement > 0 else "red")
            ],
            [
                "Avg Tokens/Tool",
                f"{original.avg_tokens_per_tool:.0f}",
                f"{comprehensive.avg_tokens_per_tool:.0f}",
                f"{comprehensive.avg_tokens_per_tool - original.avg_tokens_per_tool:+.0f}"
            ],
            [
                "Semantic Matches",
                "0 (N/A)",
                str(comprehensive.semantic_matches_found),
                colored(f"+{comprehensive.semantic_matches_found}", "green")
            ],
            [
                "Semantic Recall Rate",
                "0.0%",
                f"{comprehensive.semantic_recall_rate:.1f}%",
                colored(f"+{comprehensive.semantic_recall_rate:.1f}%", "green")
            ],
            [
                "Avg Add Latency (ms)",
                f"{original.avg_add_latency_ms:.2f}",
                f"{comprehensive.avg_add_latency_ms:.2f}",
                f"{latency_improvement:+.1f}%"
            ],
            [
                "Avg Summary Latency (ms)",
                f"{original.avg_summary_latency_ms:.2f}",
                f"{comprehensive.avg_summary_latency_ms:.2f}",
                f"{((comprehensive.avg_summary_latency_ms - original.avg_summary_latency_ms) / original.avg_summary_latency_ms * 100):+.1f}%"
            ],
            [
                "Avg Search Latency (ms)",
                "N/A",
                f"{comprehensive.avg_search_latency_ms:.2f}" if comprehensive.avg_search_latency_ms > 0 else "N/A",
                "New Feature"
            ]
        ]
        
        print(tabulate(comparison_data, headers="firstrow", tablefmt="grid"))
        
        # Key improvements summary
        print(colored("\n[KEY IMPROVEMENTS]", "cyan", attrs=["bold"]))
        print(colored(f"  • Duplicate detection: {comprehensive.duplicate_detection_rate:.1f}% accuracy", "green"))
        print(colored(f"  • Token usage: {token_improvement:.1f}% reduction", "green"))
        print(colored(f"  • Semantic search: {comprehensive.semantic_matches_found} relevant matches found", "green"))
        print(colored(f"  • False negatives: {original.false_negatives - comprehensive.false_negatives} fewer missed duplicates", "green"))
        
        print()


def main():
    """Main evaluation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate memory management systems")
    parser.add_argument(
        "--test-data",
        default="examples/tool_execution_trace.json",
        help="Path to test data JSON file"
    )
    parser.add_argument(
        "--output",
        default="evaluation_results.json",
        help="Path to save evaluation results"
    )
    
    args = parser.parse_args()
    
    evaluator = MemoryManagementEvaluator()
    evaluator.run_comparison(args.test_data, args.output)


if __name__ == "__main__":
    main()
