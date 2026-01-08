"""
Setup Intel DB
Creates the necessary tables for the Intelligence Center dynamic scoring.
"""

import os
import sys
import asyncio
from sqlmodel import SQLModel, create_engine, Session, select

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../backend"))

from app.modules.intel.models import IntelKeyword, IntelSourceAuthority
from app.core.config import settings

# Connection string
DATABASE_URL = settings.DATABASE_URL
if not DATABASE_URL:
    print("❌ DATABASE_URL not set")
    sys.exit(1)

# Convert async pg url to sync if needed for sqlmodel sync engine
# or just use create_engine with correct driver
if "+asyncpg" in DATABASE_URL:
    SYNC_DATABASE_URL = DATABASE_URL.replace("+asyncpg", "")
else:
    SYNC_DATABASE_URL = DATABASE_URL

engine = create_engine(SYNC_DATABASE_URL)

def init_db():
    print(f"🔌 Connecting to {SYNC_DATABASE_URL.split('@')[-1]}...")
    SQLModel.metadata.create_all(engine)
    print("✅ Tables created (if not exist)")

def seed_defaults():
    """Seed the DB with the hardcoded defaults from the scorer if empty"""
    # We import the defaults from the scraper script (if available)
    # OR we just define some basic ones here.
    
    # Since we don't want to depend on the scraper code here, let's just check if empty
    with Session(engine) as session:
        statement = select(IntelKeyword)
        results = session.exec(statement).first()
        if not results:
            print("🌱 Seeding default keywords...")
            # Example seed
            defaults = [
                IntelKeyword(term="kitas", category="immigration", level="direct"),
                IntelKeyword(term="visa", category="immigration", level="high"),
                IntelKeyword(term="pt pma", category="business", level="direct"),
                IntelKeyword(term="bali", category="geographic", level="direct"),
            ]
            for d in defaults:
                session.add(d)
            session.commit()
            print("✅ Seeded default keywords")
        else:
            print("ℹ️ Keywords table already populated")

        statement = select(IntelSourceAuthority)
        results = session.exec(statement).first()
        if not results:
            print("🌱 Seeding default authority...")
            defaults = [
                IntelSourceAuthority(domain="imigrasi.go.id", name="Imigrasi", score=98, category="government"),
                IntelSourceAuthority(domain="reuters", name="Reuters", score=88, category="major_media"),
                IntelSourceAuthority(domain="thebalisun", name="The Bali Sun", score=75, category="expert_trade"),
            ]
            for d in defaults:
                session.add(d)
            session.commit()
            print("✅ Seeded default authority")
        else:
            print("ℹ️ Authority table already populated")

if __name__ == "__main__":
    try:
        init_db()
        seed_defaults()
        print("🚀 Intel DB Setup Complete")
    except Exception as e:
        print(f"❌ Setup failed: {e}")
