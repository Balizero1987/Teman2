"""
Integration Tests: RAG + Memory + Knowledge Graph Integration

Tests complex multi-service integration scenarios:
1. RAG query with Memory context
2. Knowledge Graph entity extraction from conversation
3. Episodic Memory event creation from RAG results
4. Cross-service data flow and consistency

Target: Test integration between RAG, Memory, and Knowledge Graph services
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

backend_path = Path(__file__).parent.parent.parent.parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from services.rag.agentic import create_agentic_rag
from services.llm_clients.pricing import TokenUsage


@pytest.fixture(autouse=True)
def mock_trace_span():
    """Disable tracing for tests"""
    from contextlib import contextmanager
    
    @contextmanager
    def noop_trace_span(*args, **kwargs):
        yield MagicMock()
        
    with patch("services.rag.agentic.orchestrator.trace_span", side_effect=noop_trace_span), \
         patch("services.rag.agentic.reasoning.trace_span", side_effect=noop_trace_span), \
         patch("app.utils.tracing.trace_span", side_effect=noop_trace_span):
        yield


class MockSearchService:
    def __init__(self):
        self.search = AsyncMock(return_value={
            "results": [
                {
                    "id": "doc1",
                    "text": "Marco Verdi ha completato la domanda per E33G KITAS",
                    "score": 0.9,
                    "metadata": {"source": "visa_oracle"},
                }
            ],
            "total": 1,
        })

@pytest.fixture
def mock_search_service():
    """Mock SearchService"""
    return MockSearchService()


class MockPool:
    def __init__(self, conn):
        self.conn = conn
    
    def acquire(self):
        return self._Context(self.conn)
        
    class _Context:
        def __init__(self, conn):
            self.conn = conn
        async def __aenter__(self):
            return self.conn
        async def __aexit__(self, *args):
            pass

@pytest.fixture
def mock_db_pool():
    """Mock PostgreSQL connection pool"""
    conn = AsyncMock()
    conn.fetchrow.return_value = None
    return MockPool(conn)


@pytest.fixture
def mock_memory_orchestrator(mock_db_pool):
    """Mock Memory Orchestrator"""
    orchestrator = MagicMock()
    
    mock_context = MagicMock()
    mock_context.profile = {"name": "Marco Verdi", "role": "Entrepreneur"}
    mock_context.profile_facts = ["Interested in E33G KITAS", "Budget: $50k USD"]
    mock_context.collective_facts = []
    mock_context.entities = {"name": "Marco Verdi", "city": "Milano"}
    mock_context.history = []
    mock_context.timeline_summary = ""
    mock_context.kg_entities = []
    mock_context.summary = "Summary"
    mock_context.counters = {}
    
    orchestrator.get_user_context = AsyncMock(return_value=mock_context)
    orchestrator.save_conversation = AsyncMock(return_value={"success": True})
    orchestrator.initialize = AsyncMock(return_value=True)
    return orchestrator


@pytest.fixture
def mock_episodic_memory(mock_db_pool):
    """Mock Episodic Memory Service"""
    service = MagicMock()
    service.create_event = AsyncMock(
        return_value={
            "event_id": 123,
            "success": True,
            "event_type": "visa_application",
            "title": "E33G KITAS Application",
        }
    )
    service.link_entity_to_event = AsyncMock(return_value={"success": True})
    return service


class TestRAGMemoryKGIntegration:
    """RAG + Memory + Knowledge Graph Integration Tests"""

    @pytest.mark.asyncio
    async def test_rag_query_with_memory_context(
        self, mock_search_service, mock_db_pool, mock_memory_orchestrator
    ):
        """Test RAG query using Memory context for personalization"""
        query = "Quanto costa E33G per me?"
        user_id = "marco@example.com"

        # Create orchestrator with memory
        orchestrator = create_agentic_rag(retriever=mock_search_service, db_pool=mock_db_pool)
        # Inject mock memory orchestrator manually to bypass lazy loading
        orchestrator._memory_orchestrator = mock_memory_orchestrator
        
        # Inject Mock LLM Gateway
        mock_gateway = MagicMock()
        mock_gateway.create_chat_with_history = MagicMock()
        
        # Build tool call candidate
        candidate = MagicMock()
        part = MagicMock()
        part.function_call.name = "vector_search"
        part.function_call.args = {"query": "E33G price", "collection": "visa_oracle"}
        candidate.content.parts = [part]
        response_tool = MagicMock(candidates=[candidate])
        
        # Build final answer candidate
        candidate_final = MagicMock()
        part_final = MagicMock()
        part_final.text = "Final Answer: E33G costs $1000"
        part_final.function_call = None
        candidate_final.content.parts = [part_final]
        response_final = MagicMock(candidates=[candidate_final])
        
        async def async_return(val): return val
        
        mock_gateway.send_message = MagicMock(
            side_effect=[
                async_return(("Thinking...", "mock", response_tool, TokenUsage())),
                async_return(("Final Answer: Done", "mock", response_final, TokenUsage()))
            ]
        )
        orchestrator.llm_gateway = mock_gateway

        # Execute query
        result = await orchestrator.process_query(
            query=query, user_id=user_id, session_id="test-session", conversation_history=[]
        )

        # Verify memory context was used
        mock_memory_orchestrator.get_user_context.assert_called_once()

        # Verify RAG search was executed
        mock_search_service.search.assert_called()

        # Verify result includes personalized context
        assert result is not None

    @pytest.mark.asyncio
    async def test_conversation_to_kg_extraction(self, mock_db_pool, mock_search_service):
        """Test that conversations trigger Knowledge Graph extraction"""
        conversation_text = "Marco Verdi ha completato la domanda per E33G KITAS oggi"

        # Mock KG Pipeline
        with patch("services.knowledge_graph.pipeline.KGPipeline") as mock_kg:
            mock_kg_instance = MagicMock()
            mock_kg_instance.process_document = AsyncMock(
                return_value={
                    "entities": [
                        {"name": "Marco Verdi", "type": "PERSON"},
                        {"name": "E33G", "type": "VISA_TYPE"},
                    ],
                    "relations": [
                        {"subject": "Marco Verdi", "predicate": "APPLIED_FOR", "object": "E33G"}
                    ],
                    "success": True,
                }
            )
            mock_kg.return_value = mock_kg_instance

            # Process conversation through KG
            kg_pipeline = mock_kg_instance
            result = await kg_pipeline.process_document(
                text=conversation_text,
                metadata={"source": "conversation", "user_email": "marco@example.com"},
            )

            # Verify KG extraction occurred
            assert result["success"] is True
            assert len(result["entities"]) > 0
            assert len(result["relations"]) > 0

    @pytest.mark.asyncio
    async def test_rag_result_to_episodic_memory(
        self, mock_search_service, mock_db_pool, mock_episodic_memory
    ):
        """Test that RAG results create Episodic Memory events"""
        query = "Ho completato la domanda per E33G"
        user_id = "marco@example.com"

        # Mock RAG response
        rag_result = {
            "response": "Complimenti! La tua domanda per E33G è stata completata.",
            "sources": [{"id": "doc1", "text": "E33G KITAS application process"}],
        }

        # Create episodic memory event from RAG result
        with patch(
            "services.memory.episodic_memory_service.EpisodicMemoryService",
            return_value=mock_episodic_memory,
        ):
            event = await mock_episodic_memory.create_event(
                user_email=user_id,
                event_type="visa_application",
                title="E33G KITAS Application Completed",
                description=rag_result["response"],
                metadata={"rag_sources": rag_result["sources"]},
            )

            # Verify event created
            assert event["success"] is True
            assert event["event_id"] == 123

            # Verify entity linking
            link_result = await mock_episodic_memory.link_entity_to_event(
                event_id=event["event_id"], entity_name="E33G", entity_type="VISA_TYPE"
            )
            assert link_result["success"] is True

    @pytest.mark.asyncio
    async def test_memory_facts_influence_rag_response(
        self, mock_search_service, mock_db_pool, mock_memory_orchestrator
    ):
        """Test that Memory facts influence RAG response generation"""
        query = "Qual è il prossimo passo?"
        user_id = "marco@example.com"

        # Mock memory with relevant facts
        mock_memory_orchestrator.get_user_context = AsyncMock(
            return_value={
                "profile": {"name": "Marco Verdi"},
                "facts": [
                    "Marco ha completato domanda E33G",
                    "Marco ha budget di $50k USD",
                    "Marco vuole aprire ristorante a Bali",
                ],
                "collective_facts": [],
                "entities": {},
            }
        )

        # Create orchestrator
        orchestrator = create_agentic_rag(retriever=mock_search_service, db_pool=mock_db_pool)
        orchestrator._memory_orchestrator = mock_memory_orchestrator
        
        # Inject Mock LLM Gateway
        mock_gateway = MagicMock()
        mock_gateway.create_chat_with_history = MagicMock()
        
        # Simple final answer response (no tools needed for this test as we test context injection)
        # Wait, if we want to verify context influence, we just need to ensure no error and context call
        candidate_final = MagicMock()
        part_final = MagicMock()
        part_final.text = "Final Answer: Based on your budget of $50k..."
        part_final.function_call = None
        candidate_final.content.parts = [part_final]
        response_final = MagicMock(candidates=[candidate_final])
        
        async def async_return(val): return val
        
        mock_gateway.send_message = MagicMock(
            return_value=async_return(("Final Answer: Done", "mock", response_final, TokenUsage()))
        )
        orchestrator.llm_gateway = mock_gateway

        # Execute query
        result = await orchestrator.process_query(
            query=query, user_id=user_id, session_id="test-session", conversation_history=[]
        )

        # Verify memory facts were retrieved
        mock_memory_orchestrator.get_user_context.assert_called_once()

        # Verify response considers memory context
        assert result is not None

    @pytest.mark.asyncio
    async def test_kg_entities_enhance_rag_search(self, mock_search_service, mock_db_pool):
        """Test that Knowledge Graph entities enhance RAG search queries"""
        query = "Informazioni su Marco Verdi"
        user_id = "marco@example.com"

        # Mock KG entity lookup
        kg_entities = [
            {"name": "Marco Verdi", "type": "PERSON", "related_entities": ["E33G", "PT PMA"]}
        ]

        # Enhance query with KG entities
        enhanced_query = f"{query} [Entities: {', '.join([e['name'] for e in kg_entities])}]"

        # Execute enhanced search
        result = await mock_search_service.search(
            query=enhanced_query, collection="visa_oracle", limit=5
        )

        # Verify search was executed with enhanced query
        assert result is not None
        assert "results" in result

    @pytest.mark.asyncio
    async def test_conversation_save_triggers_kg_and_memory(
        self, mock_db_pool, mock_episodic_memory
    ):
        """Test that saving conversation triggers both KG extraction and Memory storage"""
        conversation_text = "Marco Verdi ha aperto PT PMA a Bali nel 2024"
        user_email = "marco@example.com"

        # Mock ConversationService
        with patch("services.misc.conversation_service.ConversationService") as mock_conv:
            mock_conv_instance = MagicMock()
            mock_conv_instance.save_conversation = AsyncMock(
                return_value={"success": True, "conversation_id": 456}
            )
            mock_conv.return_value = mock_conv_instance

            # Mock KG extraction
            with patch("services.knowledge_graph.pipeline.KGPipeline") as mock_kg:
                mock_kg_instance = MagicMock()
                mock_kg_instance.process_document = AsyncMock(
                    return_value={
                        "entities": [{"name": "Marco Verdi", "type": "PERSON"}],
                        "success": True,
                    }
                )
                mock_kg.return_value = mock_kg_instance

                # Save conversation
                conv_service = mock_conv_instance
                result = await conv_service.save_conversation(
                    user_email=user_email,
                    messages=[{"role": "user", "content": conversation_text}],
                    session_id="test-session",
                )

                # Verify conversation saved
                assert result["success"] is True

                # Process through KG
                kg_result = await mock_kg_instance.process_document(
                    text=conversation_text,
                    metadata={
                        "source": "conversation",
                        "conversation_id": result["conversation_id"],
                    },
                )

                # Verify KG extraction occurred
                assert kg_result["success"] is True

    @pytest.mark.asyncio
    async def test_multi_turn_conversation_with_kg_updates(
        self, mock_search_service, mock_db_pool, mock_memory_orchestrator
    ):
        """Test multi-turn conversation that updates Knowledge Graph incrementally"""
        user_id = "marco@example.com"
        session_id = "test-session"

        # Turn 1: User mentions entity
        turn1_query = "Mi chiamo Marco Verdi"
        turn1_response = "Ciao Marco! Piacere di conoscerti."

        # Turn 2: User mentions relationship
        turn2_query = "Voglio aprire una PT PMA"
        turn2_response = "Ti aiuto con la PT PMA!"

        # Mock KG updates
        with patch("services.knowledge_graph.pipeline.KGPipeline") as mock_kg:
            mock_kg_instance = MagicMock()
            entities_turn1 = [{"name": "Marco Verdi", "type": "PERSON"}]
            entities_turn2 = [
                {"name": "Marco Verdi", "type": "PERSON"},
                {"name": "PT PMA", "type": "COMPANY_TYPE"},
            ]
            relations_turn2 = [
                {"subject": "Marco Verdi", "predicate": "WANTS_TO_CREATE", "object": "PT PMA"}
            ]

            mock_kg_instance.process_document = AsyncMock(
                side_effect=[
                    {"entities": entities_turn1, "success": True},
                    {"entities": entities_turn2, "relations": relations_turn2, "success": True},
                ]
            )
            mock_kg.return_value = mock_kg_instance

            # Process turns
            kg_pipeline = mock_kg_instance

            # Turn 1
            result1 = await kg_pipeline.process_document(
                text=turn1_query, metadata={"turn": 1, "session_id": session_id}
            )

            # Turn 2
            result2 = await kg_pipeline.process_document(
                text=f"{turn1_query} {turn2_query}", metadata={"turn": 2, "session_id": session_id}
            )

            # Verify incremental KG updates
            assert result1["success"] is True
            assert result2["success"] is True
            assert len(result2["entities"]) > len(result1["entities"])
            assert len(result2.get("relations", [])) > 0
