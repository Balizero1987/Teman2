import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from services.tools.intel_management import IntelManagementTool
from services.rag.agentic.reasoning import ReasoningEngine
from services.tools.definitions import AgentState, ToolCall

# Mock Connection
class MockConn:
    def __init__(self):
        self.fetchval = AsyncMock()
        self.fetchrow = AsyncMock()
        self.execute = AsyncMock()
        self.close = AsyncMock()

@pytest.fixture
def mock_conn():
    return MockConn()

@pytest.mark.asyncio
async def test_agent_adds_keyword(mock_conn):
    """
    Test that the IntelManagementTool works correctly when invoked by the agent logic.
    """
    tool = IntelManagementTool()
    
    with patch.object(tool, "_get_connection", return_value=mock_conn):
        # Setup mock DB behavior
        mock_conn.fetchval.return_value = None # Not exists
        mock_conn.fetchrow.return_value = {"id": 1, "term": "crypto", "category": "finance"}
        
        # Execute tool directly
        result = await tool.execute(term="crypto", category="finance", level="high")
        
        # Verify
        assert "Successfully added new keyword" in result
        assert "crypto" in result
        
        # Verify SQL was called
        assert mock_conn.fetchval.called
        assert mock_conn.execute.called

@pytest.mark.asyncio
async def test_agent_updates_keyword(mock_conn):
    tool = IntelManagementTool()
    
    with patch.object(tool, "_get_connection", return_value=mock_conn):
        # Setup mock: exists returns ID 5
        mock_conn.fetchval.return_value = 5
        
        result = await tool.execute(term="crypto", category="finance", level="low")
        
        assert "Updated existing keyword" in result
        assert "low" in result

@pytest.mark.asyncio
async def test_agent_tool_in_loop(mock_conn):
    """
    Simulate the full loop: LLM decides to add keyword -> Tool executes.
    """
    tool = IntelManagementTool()
    tools_map = {"manage_intelligence": tool}
    engine = ReasoningEngine(tool_map=tools_map)
    
    state = AgentState(query="Add 'AI' to keywords", max_steps=2)
    # Bypass evidence check for general task or by ensuring trusted tool used logic
    # But manage_intelligence is NOT a trusted tool in reasoning.py
    # So we should set skip_rag=True or expect low evidence warning
    state.skip_rag = True 
    
    # Mock LLM
    llm = AsyncMock()
    candidate = MagicMock()
    
    tool_call = ToolCall(
        tool_name="manage_intelligence",
        arguments={"term": "AI", "category": "tech", "level": "high"}
    )
    
    # We patch _get_connection on the tool instance we created
    with patch("services.rag.agentic.reasoning.parse_tool_call", side_effect=[tool_call, None, None]), \
         patch.object(tool, "_get_connection", return_value=mock_conn):
         
        mock_conn.fetchval.return_value = None

        llm.send_message.side_effect = [
            ("Thinking...", "gemini-pro", MagicMock(candidates=[candidate]), MagicMock()),
            ("Done", "gemini-pro", MagicMock(candidates=[]), MagicMock())
        ]
        
        # Need to ensure the tool inside engine map is the patched one
        # It is, because we passed `tools_map` with `tool` instance.
        
        final_state, _, _, _ = await engine.execute_react_loop(
            state=state,
            llm_gateway=llm,
            chat=MagicMock(),
            initial_prompt="start",
            system_prompt="sys",
            query="test",
            user_id="user",
            model_tier=1,
            tool_execution_counter={"count": 0}
        )
        
        # Check if tool was executed and result is in context
        assert any("Successfully added new keyword" in ctx for ctx in final_state.context_gathered)
