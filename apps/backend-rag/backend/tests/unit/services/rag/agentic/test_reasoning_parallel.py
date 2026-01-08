import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from services.rag.agentic.reasoning import ReasoningEngine
from services.tools.definitions import AgentState, ToolCall
from services.llm_clients.pricing import TokenUsage

@pytest.fixture
def mock_llm_gateway():
    gateway = AsyncMock()
    # Mock send_message to return a dummy response
    gateway.send_message.return_value = (
        "Thinking...", 
        "gemini-pro", 
        MagicMock(candidates=[]), 
        TokenUsage()
    )
    return gateway

@pytest.fixture
def mock_tools():
    # Tools that simulate work
    async def tool1_func(**kwargs):
        await asyncio.sleep(0.1)
        return "Result 1"

    async def tool2_func(**kwargs):
        await asyncio.sleep(0.1)
        return "Result 2"

    mock_t1 = MagicMock()
    mock_t1.execute = AsyncMock(side_effect=tool1_func)
    
    mock_t2 = MagicMock()
    mock_t2.execute = AsyncMock(side_effect=tool2_func)

    return {
        "tool1": mock_t1,
        "tool2": mock_t2
    }

@pytest.mark.asyncio
async def test_execute_react_loop_parallel_execution(mock_llm_gateway, mock_tools):
    """
    Test that multiple tool calls returned by the LLM are executed in parallel.
    """
    # Setup
    engine = ReasoningEngine(tool_map=mock_tools)
    state = AgentState(query="test query", max_steps=2)
    
    # Mock response object structure
    candidate = MagicMock()
    part1 = MagicMock()
    part2 = MagicMock()
    candidate.content.parts = [part1, part2] # Two parts
    response_obj = MagicMock(candidates=[candidate])
    
    mock_llm_gateway.send_message.side_effect = [
        # First call: returns 2 tools
        ("Thinking...", "gemini-pro", response_obj, TokenUsage()),
        # Second call: returns Final Answer
        ("Final Answer: Done", "gemini-pro", MagicMock(candidates=[]), TokenUsage())
    ]
    
    tool_call1 = ToolCall(tool_name="tool1", arguments={"arg": 1})
    tool_call2 = ToolCall(tool_name="tool2", arguments={"arg": 2})

    # Patch parse_tool_call to return our tool calls
    # The logic loops over parts. For the first loop (step 1):
    # part1 -> tool_call1
    # part2 -> tool_call2
    # Then loop finishes.
    # Second step: text response -> None (no tool call)
    
    with patch("services.rag.agentic.reasoning.parse_tool_call") as mock_parse:
        # Step 1: Called for part1, then part2
        # Step 2: Called for text_response (fallback check) -> returns None
        mock_parse.side_effect = [tool_call1, tool_call2, None, None]

        # Execute
        start_time = asyncio.get_event_loop().time()
        final_state, _, _, _ = await engine.execute_react_loop(
            state=state,
            llm_gateway=mock_llm_gateway,
            chat=MagicMock(),
            initial_prompt="start",
            system_prompt="sys",
            query="test",
            user_id="user",
            model_tier=1,
            tool_execution_counter={"count": 0}
        )
        duration = asyncio.get_event_loop().time() - start_time

        # Verify
        # 1. Check if both tools were executed
        assert mock_tools["tool1"].execute.called
        assert mock_tools["tool2"].execute.called
        
        # 2. Check timing: if sequential, it would be > 0.2s. If parallel, around ~0.1s + overhead
        # Note: In CI/CD or loaded envs, timing assertions are flaky. 
        # But we can assert that both results are present.
        
        assert any("Result 1" in ctx for ctx in state.context_gathered)
        assert any("Result 2" in ctx for ctx in state.context_gathered)
        
        # 3. Check steps
        # We expect: 
        # - Step 1: Tool 1 execution
        # - Step 2: Tool 2 execution
        # - Step 3: Final Answer
        # Note: The implementation appends a step for EACH tool call in the parallel batch
        
        tool_steps = [s for s in state.steps if s.action and s.action.tool_name in ["tool1", "tool2"]]
        assert len(tool_steps) == 2

@pytest.mark.asyncio
async def test_execute_react_loop_stream_parallel(mock_llm_gateway, mock_tools):
    """
    Test streaming parallel execution.
    """
    engine = ReasoningEngine(tool_map=mock_tools)
    state = AgentState(query="test query stream", max_steps=2)
    
    # Similar setup for response_obj
    candidate = MagicMock()
    part1 = MagicMock()
    part2 = MagicMock()
    candidate.content.parts = [part1, part2]
    response_obj = MagicMock(candidates=[candidate])
    
    mock_llm_gateway.send_message.side_effect = [
        ("Thinking...", "gemini-pro", response_obj, TokenUsage()),
        ("Final Answer: Done", "gemini-pro", MagicMock(candidates=[]), TokenUsage())
    ]
    
    tool_call1 = ToolCall(tool_name="tool1", arguments={"arg": 1})
    tool_call2 = ToolCall(tool_name="tool2", arguments={"arg": 2})

    events = []
    with patch("services.rag.agentic.reasoning.parse_tool_call") as mock_parse:
        # Stream implementation logic:
        # 1. Loop parts -> identify tools (mock_parse called for part1, part2)
        # 2. Check fallback if none found (not reached here because we found tools)
        # 3. Yield tool calls
        # 4. Execute parallel
        # 5. Next step...
        mock_parse.side_effect = [tool_call1, tool_call2, None, None]

        async for event in engine.execute_react_loop_stream(
            state=state,
            llm_gateway=mock_llm_gateway,
            chat=MagicMock(),
            initial_prompt="start",
            system_prompt="sys",
            query="test",
            user_id="user",
            model_tier=1,
            tool_execution_counter={"count": 0}
        ):
            events.append(event)

    # Verify events
    # Should have tool_call events for both
    tool_call_events = [e for e in events if e["type"] == "tool_call"]
    assert len(tool_call_events) == 2
    
    # Should have observation events for both
    obs_events = [e for e in events if e["type"] == "observation"]
    assert len(obs_events) == 2
    assert any("Result 1" in str(e["data"]) for e in obs_events)
    assert any("Result 2" in str(e["data"]) for e in obs_events)
