import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Type

from termcolor import colored
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from knowledge_graph_service import KnowledgeGraphService
from optimized_knowledge_graph_service import OptimizedKnowledgeGraphService
from token_counter import TokenCounter
from llm_service import LLMService

def _load_dataset(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    assert isinstance(data, list), "Dataset JSON must be a list of dicts"
    return data


from transformers.pipelines import pipeline
import bert_score

def faithfulness(src, hyp, w_bs=.6, w_nli=.4):
    # 1) BERTScore - F1 score (meaning: how much of the source text is captured in the summary)
    all_preds = bert_score.score([hyp], [src], lang="en", rescale_with_baseline=False)
    avg_scores = [s.mean(dim=0) for s in all_preds]
    P = avg_scores[0].cpu().item()
    R = avg_scores[1].cpu().item()
    F1 = avg_scores[2].cpu().item()


    # 2) NLI entailment - entailment probability (meaning: how much of the summary is entailed by the source text)
    nli = pipeline("text-classification", model="roberta-large-mnli")
    ent  = nli(f"{src} </s> {hyp}")[0]["score"]

    print(f"P: {P}, R: {R}, F1: {F1}, ent: {ent}")

    return w_bs*F1 + w_nli*ent



def run_benchmark(
    service_cls: Type,
    tool_entries: List[Dict[str, Any]],
    workflow_id: str = "benchmark",
    
) -> Dict[str, Any]:
    tc = TokenCounter()

    inst_llm = LLMService(api_key=os.getenv("OPENAI_API_KEY")) # type: ignore

    service = service_cls(workflow_id=workflow_id, llm_service=inst_llm)
    service.reset_workflow()

    add_latencies: List[float] = []
    summary_latencies: List[float] = []

    for idx, entry in enumerate(tool_entries, 1):
        t0 = time.perf_counter()
        tool_id = service.add_tool_result(entry)
        add_latencies.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        service.generate_summary(tool_id)
        summary_latencies.append(time.perf_counter() - t1)

    dashboard_start = time.perf_counter()
    all_compress = {
        "all" : {
            "tool_ids" : [tool.tool_id for tool in service.get_all_tool_results()]
        }
    }
    dashboard = service.generate_dashboard(compressed_tool_groups=all_compress)
    dashboard_latency = time.perf_counter() - dashboard_start
    print(dashboard)

    print("======FINISHED GENERATING DASHBOARD======\n")

    dashboard_tokens = tc.count_tokens(dashboard)
    original_tokens = sum([tc.count_tokens(json.dumps(e)) for e in tool_entries])

    sims = []
    tool_strings = service.generate_tool_strings_for_dashboard()
    for i, entry in enumerate(tool_entries, 1):
        tool_id = f"TR-{i}"
        sims.append(
            faithfulness(json.dumps(entry), tool_strings[i - 1])
        )

    quality_score = statistics.mean(sims) if sims else 0.0

    result = {
        "service": service_cls.__name__,
        "entries": len(tool_entries),
        "add_p50_ms": statistics.median(add_latencies) * 1000,
        "summary_p50_ms": statistics.median(summary_latencies) * 1000,
        "dashboard_latency_ms": dashboard_latency * 1000,
        "dashboard_tokens": dashboard_tokens,
        "original_tokens": original_tokens,
        "compression_%": round((original_tokens - dashboard_tokens) / original_tokens * 100, 2) if original_tokens else 0,
        "quality_mean": round(quality_score, 3),
    }

    return result


def format_metric_value(key: str, value: Any) -> str:
    """Format metric values for display."""
    if isinstance(value, float):
        if key.endswith('_ms'):
            return f"{value:.1f} ms"
        elif key.endswith('_%'):
            return f"{value:.1f}%"
        elif key == 'quality_mean':
            return f"{value:.3f}"
        else:
            return f"{value:.2f}"
    elif isinstance(value, int):
        if 'tokens' in key:
            return f"{value:,}"
        else:
            return str(value)
    else:
        return str(value)


def create_comparison_table(results: Dict[str, Dict[str, Any]], output_path: str):
    """Create a visual comparison table as JPG."""
    
    metric_display_names = {
        'entries': 'Entries Processed',
        'add_p50_ms': 'Add Latency (P50)',
        'summary_p50_ms': 'Summary Latency (P50)',
        'dashboard_latency_ms': 'Dashboard Latency',
        'dashboard_tokens': 'Dashboard Tokens',
        'original_tokens': 'Original Tokens',
        'compression_%': 'Compression Rate',
        'quality_mean': 'Quality Score'
    }
    
    # Prepare data for table
    services = list(results.keys())
    metrics = list(metric_display_names.keys())
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    colors = []
    
    header_row = ['Metric'] + services
    table_data.append(header_row)
    colors.append(['lightgray'] * len(header_row))
    
    # Data rows
    for metric in metrics:
        if metric == 'service':
            continue
            
        row = [metric_display_names.get(metric, metric)]
        row_colors = ['lightblue']
        
        values = []
        for service in services:
            value = results[service].get(metric, 'N/A')
            formatted_value = format_metric_value(metric, value)
            row.append(formatted_value)
            values.append(value if isinstance(value, (int, float)) else 0)
        
        # Determine color (green = better, red = worse)
        if len(values) == 2 and all(isinstance(v, (int, float)) for v in values):
            if any(keyword in metric for keyword in ['latency', 'tokens', 'ms']):
                if values[0] < values[1]:
                    row_colors.extend(['lightgreen', 'lightcoral'])
                elif values[0] > values[1]:
                    row_colors.extend(['lightcoral', 'lightgreen'])
                else:
                    row_colors.extend(['white', 'white'])
            elif any(keyword in metric for keyword in ['quality', 'compression']):
                if values[0] > values[1]:
                    row_colors.extend(['lightgreen', 'lightcoral'])
                elif values[0] < values[1]:
                    row_colors.extend(['lightcoral', 'lightgreen'])
                else:
                    row_colors.extend(['white', 'white'])
            else:
                row_colors.extend(['white', 'white'])
        else:
            row_colors.extend(['white'] * len(services))
        
        table_data.append(row)
        colors.append(row_colors)
    
    table = ax.table(cellText=table_data, 
                     cellColours=colors,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.35] + [0.325] * len(services))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    for i in range(len(header_row)):
        table[(0, i)].set_text_props(weight='bold')
    
    for i in range(1, len(table_data)):
        table[(i, 0)].set_text_props(weight='bold')
    
    plt.title('Memory Management Service Benchmark Comparison', 
              fontsize=16, fontweight='bold', pad=20)
    
    green_patch = mpatches.Patch(color='lightgreen', label='Better Performance')
    red_patch = mpatches.Patch(color='lightcoral', label='Worse Performance')
    plt.legend(handles=[green_patch, red_patch], loc='upper center', 
               bbox_to_anchor=(0.5, -0.05), ncol=2)
    
    # Save jpg
    plt.tight_layout()
    plt.savefig(output_path, format='jpg', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Comparison table saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Memory pipeline benchmark runner - compares baseline and optimized services")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to JSON list of tool-result dicts.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark.jpg",
        help="Output path for comparison table JPG",
    )
    args = parser.parse_args()

    entries = _load_dataset(Path(args.dataset))
    print(f"Loaded {len(entries)} entries from dataset")
   
    services: List[Tuple[Type, str]] = [
        (KnowledgeGraphService, "Baseline"),
        (OptimizedKnowledgeGraphService, "Optimized")
    ]

    results = {}
    for service_cls, label in services:
        print(colored(f"\n{'='*60}", "cyan"))
        print(colored(f"Running benchmark for {label} service...", "cyan"))
        print(colored(f"{'='*60}", "cyan"))
        
        res = run_benchmark(service_cls, entries, workflow_id=f"bench_{label.lower()}")
        results[label] = res
    create_comparison_table(results, args.output)
    
    print(colored(f"\n{'='*60}", "green"))
    print(colored("Performance Improvements (Optimized vs Baseline):", "green"))
    print(colored(f"{'='*60}", "green"))
    
    baseline = results['Baseline']
    optimized = results['Optimized']
    
    # Latency comparison
    for metric in ['add_p50_ms', 'summary_p50_ms', 'dashboard_latency_ms']:
        if metric in baseline and metric in optimized:
            improvement = (baseline[metric] - optimized[metric]) / baseline[metric] * 100
            color = "green" if improvement > 0 else "red"
            print(colored(f"{metric}: {improvement:+.1f}% {'faster' if improvement > 0 else 'slower'}", color))
    
    # Token usage
    if 'dashboard_tokens' in baseline and 'dashboard_tokens' in optimized:
        token_reduction = (baseline['dashboard_tokens'] - optimized['dashboard_tokens']) / baseline['dashboard_tokens'] * 100
        color = "green" if token_reduction > 0 else "red"
        print(colored(f"Dashboard tokens: {token_reduction:+.1f}% {'reduction' if token_reduction > 0 else 'increase'}", color))
    
    # Quality comparison
    if 'quality_mean' in baseline and 'quality_mean' in optimized:
        quality_diff = optimized['quality_mean'] - baseline['quality_mean']
        color = "green" if quality_diff > 0 else "red"
        print(colored(f"Quality score: {quality_diff:+.3f} {'better' if quality_diff > 0 else 'worse'}", color))


if __name__ == "__main__":
    main() 