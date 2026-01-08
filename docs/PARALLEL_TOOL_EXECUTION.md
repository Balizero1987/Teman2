# Parallel Tool Execution Architecture

## Overview
As of Jan 2026, the Nuzantara Agentic RAG engine supports **Parallel Tool Execution**. This enhancement allows the reasoning engine to execute multiple tool calls concurrently within a single ReAct step, significantly reducing latency for complex user queries.

## Motivation
Previously, the ReAct loop executed tools sequentially: `Thought -> Action -> Observation -> Thought -> Action -> Observation`.
For queries requiring multiple pieces of information (e.g., "Check visa requirements and tell me the cost for a KITAS"), this resulted in multiple round-trips to the LLM and sequential waits for tool execution.

**With Parallel Execution:**
`Thought -> Actions (Tool A, Tool B, Tool C) -> Observations (Result A, Result B, Result C) -> Thought`

This reduces the total response time to `max(tool_duration)` instead of `sum(tool_durations)`.

## Implementation Details

### Core Logic (`reasoning.py`)
- **Detection**: The system detects multiple tool calls from the LLM response (supporting both Native Gemini Function Calling and Regex fallback).
- **Execution**: identified tool calls are queued as `asyncio` tasks.
- **Concurrency**: `asyncio.gather(*tasks)` executes all tools simultaneously.
- **Context**: Results are collected, and the `AgentState` is updated with all observations.

### Streaming Support
The parallel execution model is fully supported in the streaming endpoint (`execute_react_loop_stream`). Events are emitted in this order:
1. `thinking`: "Step X: Processing..."
2. `tool_call`: Emitted for *all* tools in the batch immediately.
3. `observation`: Emitted as each tool finishes (or all together depending on the gather implementation, currently batched after gather).

## Observability

### Tracing (OpenTelemetry)
- **Span**: `react.step.N` contains attributes:
  - `parallel_tools_count`: Number of tools executed in this step.
  - `parallel_execution_time_ms`: Time taken for the `asyncio.gather` block.
- **Span Events**: `tool.call` events are added for each tool in the batch.

### Metrics (Prometheus)
- **`zantara_rag_parallel_batch_size`** (Histogram): Tracks the distribution of batch sizes (e.g., how often 2, 3, or 5 tools are called at once).
- **`zantara_rag_tool_calls_total`** (Counter): Incremented for each individual tool execution.

## Developer Notes
- **Thread Safety**: Ensure tools are thread-safe/async-safe. The current toolset (Vector Search, Pricing, Calculator) is stateless and safe.
- **Error Handling**: If one tool fails in a batch, it logs the error but does not crash the entire batch (individual try/except in `execute_tool`).
- **Dependencies**: Requires `asyncio` and `pytest-asyncio` for testing.

## Example
**User Query**: "Dammi i requisiti per il visto B211A e i prezzi del servizio VIP."

**Old Behavior (Sequential)**:
1. Call `vector_search("B211A requirements")` -> Wait 2s
2. LLM receives context -> Calls `get_pricing("VIP service")` -> Wait 1s
3. LLM Generates answer.
**Total Latency**: ~3s + 2x LLM generation.

**New Behavior (Parallel)**:
1. LLM identifies need for both info -> Calls `vector_search` AND `get_pricing` together.
2. System executes both. `vector_search` takes 2s, `get_pricing` takes 1s.
3. `asyncio.gather` returns after 2s (max duration).
4. LLM Generates answer.
**Total Latency**: ~2s + 1x LLM generation.
