import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from app.modules.intel.router import router
from app.dependencies import get_database_pool, get_current_user

# Mock dependencies
class MockPool:
    def __init__(self):
        self.conn = AsyncMock()
    
    def acquire(self):
        class ContextManager:
            def __init__(self, conn):
                self.conn = conn
            async def __aenter__(self):
                return self.conn
            async def __aexit__(self, exc_type, exc, tb):
                pass
        return ContextManager(self.conn)

async def mock_get_database_pool():
    return MockPool()

async def mock_get_current_user():
    return {"user_id": "admin", "role": "admin", "email": "admin@balizero.com"}

@pytest.fixture
def client():
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    app.dependency_overrides[get_database_pool] = mock_get_database_pool
    app.dependency_overrides[get_current_user] = mock_get_current_user
    
    return TestClient(app)

@pytest.mark.asyncio
async def test_list_keywords(client):
    # Setup mock
    pool = await mock_get_database_pool()
    conn = pool.acquire().conn
    conn.fetch.return_value = [
        {"id": 1, "term": "visa", "category": "immigration", "level": "high", "is_active": True, "created_at": "2023-01-01", "updated_at": "2023-01-01"}
    ]
    
    # We need to ensure the dependency override returns THIS pool instance
    # But dependency override is a function.
    # Let's use a simpler approach: define the mock pool globally or capture it
    
    # Actually, TestClient calls the endpoint which calls get_database_pool.
    # The fixture sets up the override to call mock_get_database_pool.
    # mock_get_database_pool returns a NEW MockPool each time.
    # We need to control the MockPool returned by the override.
    
    mock_pool_instance = MockPool()
    async def get_controlled_pool():
        return mock_pool_instance
    
    client.app.dependency_overrides[get_database_pool] = get_controlled_pool
    
    conn = mock_pool_instance.conn
    conn.fetch.return_value = [
        {"id": 1, "term": "visa", "category": "immigration", "level": "high", "is_active": True, "created_at": "2023-01-01", "updated_at": "2023-01-01"}
    ]
    
    response = client.get("/intel/keywords")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["term"] == "visa"

@pytest.mark.asyncio
async def test_create_keyword(client):
    mock_pool_instance = MockPool()
    async def get_controlled_pool():
        return mock_pool_instance
    client.app.dependency_overrides[get_database_pool] = get_controlled_pool
    conn = mock_pool_instance.conn
    
    conn.fetchval.return_value = None
    conn.fetchrow.return_value = {
        "id": 2, 
        "term": "golden visa", 
        "category": "immigration", 
        "level": "direct", 
        "score_override": None,
        "is_active": True,
        "created_at": "2023-01-01", 
        "updated_at": "2023-01-01"
    }
    
    payload = {
        "term": "golden visa",
        "category": "immigration",
        "level": "direct"
    }
    
    response = client.post("/intel/keywords", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["term"] == "golden visa"

@pytest.mark.asyncio
async def test_create_keyword_duplicate(client):
    mock_pool_instance = MockPool()
    async def get_controlled_pool():
        return mock_pool_instance
    client.app.dependency_overrides[get_database_pool] = get_controlled_pool
    conn = mock_pool_instance.conn
    
    conn.fetchval.return_value = 1
    
    payload = {
        "term": "visa",
        "category": "immigration",
        "level": "high"
    }
    
    response = client.post("/intel/keywords", json=payload)
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]

@pytest.mark.asyncio
async def test_update_keyword(client):
    mock_pool_instance = MockPool()
    async def get_controlled_pool():
        return mock_pool_instance
    client.app.dependency_overrides[get_database_pool] = get_controlled_pool
    conn = mock_pool_instance.conn
    
    conn.fetchrow.side_effect = [
        {"id": 1, "term": "visa", "category": "immigration"}, 
        {
            "id": 1, 
            "term": "visa", 
            "category": "immigration", 
            "level": "direct", 
            "is_active": True,
            "created_at": "2023-01-01", 
            "updated_at": "2023-01-01"
        }
    ]
    
    payload = {"level": "direct"}
    response = client.put("/intel/keywords/1", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["level"] == "direct"

@pytest.mark.asyncio
async def test_delete_keyword(client):
    mock_pool_instance = MockPool()
    async def get_controlled_pool():
        return mock_pool_instance
    client.app.dependency_overrides[get_database_pool] = get_controlled_pool
    conn = mock_pool_instance.conn
    
    conn.execute.return_value = "UPDATE 1"
    
    response = client.delete("/intel/keywords/1")
    assert response.status_code == 200
    assert response.json()["success"] is True

@pytest.mark.asyncio
async def test_list_authority(client):
    mock_pool_instance = MockPool()
    async def get_controlled_pool():
        return mock_pool_instance
    client.app.dependency_overrides[get_database_pool] = get_controlled_pool
    conn = mock_pool_instance.conn
    
    conn.fetch.return_value = [
        {"id": 1, "domain": "reuters.com", "name": "Reuters", "score": 90, "category": "major_media", "is_active": True, "created_at": "2023-01-01", "updated_at": "2023-01-01"}
    ]
    
    response = client.get("/intel/authority")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["name"] == "Reuters"
