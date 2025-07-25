# Memory Management Take-Home Assignment

## Assignment

**Understand the current memory management system, improve it, and prove your improvement works better.**

The current system manages memory for an AI agent that executes DevOps tools. Your job is to:

1. **Understand** how the current memory management works
2. **Improve** the memory management approach 
3. **Test** that your approach is better than the current one

## What You're Working With

### Agent Execution Data

- `examples/agent_knowledge_sequence.txt` - Real agent execution showing how tools are compressed/expanded in practice and how they're displayed in the prompt of our agent
- `examples/tool_execution_trace.json` - Raw tool execution data that needs to be managed

### Current System

The agent executes tools like:
- `execute_command` (AWS CLI, terraform, etc.)
- `create_file`, `modify_code`, `read_file_contents`  
- `query_codebase`
- Integration method calls
- etc

Each tool execution generates results that consume the token space within the prompt. 
We want a system that allows the agent to, much like a human, remove information from immediate information as it's less needed, add stored information back into context in the future as needed, etc. 

Your goal is to understand how the tool calls for our DevOps agent work, understand how the current method of memory management works, improve the method in some set of ways, and implement some level of evaluation to the best of your ability to demonstrate that your method was an actual improvement over the current setup. The method of evaluation and the metrics you are using for evaluation are mostly up to you, but make sure to use metrics that makes the most sense given the background you have been given.

The system uses:
- **Neo4j graph database** for storing tool results and relationships
- **LLM summarization** for compressing tool results
- **Token counting** (tiktoken) for memory tracking
- **Compression/expansion** for managing what's visible vs. stored

## Codebase Structure

```
src/
├── knowledge_graph_service.py  # Main memory management logic
├── neo4j_service.py           # Database operations  
├── neo4j_adapter.py           # Database connection
├── llm_service.py             # LLM integration for summarization
├── tool_summary_prompts.py    # Prompts for summarization
├── token_counter.py           # Token counting utility
└── models.py                  # Data structures
```

### Key Components

**KnowledgeGraphService** - Main class that:
- Stores tool results in Neo4j with full context
- Generates LLM summaries with salient data extraction  
- Compresses multiple tool results into summary groups
- Retrieves full details or summaries as needed
- Tracks relationships between tools

## Setup

```bash
python -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```

**Environment variables needed for full functionality:**
```
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j  
NEO4J_PASSWORD=your-password
OPENAI_API_KEY=sk-your-key
```

## Live Demo

**`demo_compression.py`** - A demonstration of how the agent's memory management cam work in practice:

The agent continuously summarizes tool results as they're executed, running in parallel with planning the next action to maximize efficiency

Demonstrates how related tool executions (e.g., AWS operations, file modifications) get grouped and compressed when memory pressure increases. Shows how the agent selectively expands compressed tools back to full detail when that specific information becomes relevant for future actions. Illustrates the complete workflow from raw tool results → summaries → compression → selective expansion based on what the agent needs to know

Run with:

```bash
python demo_compression.py
```

This gives you a concrete example of how an agent would manage its memory throughout a complex DevOps workflow, maintaining efficiency while preserving access to detailed information when needed.

## Expected Deliverable
You have starting Saturday at 12 AM PST to Sunday 8 PM PST to finish the task. Within this time frame feel free to send questions to rithvikprakki@a37.ai, but the codebase should be clear enough for you to make great progress without much need for questions. 

After the alotted time period has completed we'll have a call where you will need to walk us through your exploration process, code, and final results.

The goal is to have working code that demonstrates a better memory management approach with some evidence that it's superior to the current system.
