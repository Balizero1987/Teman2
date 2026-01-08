#!/usr/bin/env python3
"""
Migration CLI Tool - Unified migration management

Usage:
    python -m backend.db.migrate status           # Show migration status
    python -m backend.db.migrate apply-all        # Apply all pending migrations
    python -m backend.db.migrate apply 028        # Apply specific migration
    python -m backend.db.migrate rollback 028     # Rollback specific migration
    python -m backend.db.migrate list             # List all migrations
    python -m backend.db.migrate verify           # Verify all migrations have rollback

On Fly.io:
    fly ssh console -a nuzantara-rag -C "cd /app && python -m backend.db.migrate status"
"""

import argparse
import asyncio
import importlib
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import asyncpg

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("migrate")

# Add backend to path
backend_path = Path(__file__).parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from app.core.config import settings


async def get_db_connection():
    """Get database connection"""
    if not settings.database_url:
        logger.error("DATABASE_URL not set")
        sys.exit(1)
    return await asyncpg.connect(settings.database_url)


async def ensure_schema_migrations(conn: asyncpg.Connection):
    """Ensure schema_migrations table exists with all columns"""
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id SERIAL PRIMARY KEY,
            migration_name VARCHAR(255) UNIQUE NOT NULL,
            migration_number INTEGER NOT NULL,
            executed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            checksum VARCHAR(64) DEFAULT '',
            description TEXT,
            execution_time_ms INTEGER DEFAULT 0,
            rollback_sql TEXT,
            applied_by VARCHAR(255) DEFAULT 'system'
        );

        ALTER TABLE schema_migrations ADD COLUMN IF NOT EXISTS migration_number INTEGER DEFAULT 0;
        ALTER TABLE schema_migrations ADD COLUMN IF NOT EXISTS description TEXT;
        ALTER TABLE schema_migrations ADD COLUMN IF NOT EXISTS rollback_sql TEXT;
        ALTER TABLE schema_migrations ADD COLUMN IF NOT EXISTS checksum VARCHAR(64) DEFAULT '';
        ALTER TABLE schema_migrations ADD COLUMN IF NOT EXISTS execution_time_ms INTEGER DEFAULT 0;
        ALTER TABLE schema_migrations ADD COLUMN IF NOT EXISTS applied_by VARCHAR(255) DEFAULT 'system';
    """)


def discover_migrations() -> list[dict]:
    """Discover all Python migration files"""
    migrations_dir = Path(__file__).parent.parent / "migrations"
    migrations = []

    for py_file in sorted(migrations_dir.glob("migration_*.py")):
        name = py_file.stem
        parts = name.split("_")

        if len(parts) >= 2:
            try:
                num_str = parts[1]
                migration_number = int(num_str)
                migrations.append({
                    "number": migration_number,
                    "file": py_file.name,
                    "path": py_file,
                    "module": f"migrations.{name}",
                })
            except ValueError:
                logger.warning(f"Could not parse migration number from {py_file.name}")
                continue

    migrations.sort(key=lambda x: x["number"])
    return migrations


def load_migration_module(migration: dict) -> Any:
    """Load migration module dynamically"""
    try:
        module = importlib.import_module(migration["module"])
        return module
    except ImportError as e:
        logger.error(f"Failed to import {migration['module']}: {e}")
        return None


async def get_applied_migrations(conn: asyncpg.Connection) -> set[int]:
    """Get set of applied migration numbers"""
    await ensure_schema_migrations(conn)
    rows = await conn.fetch("SELECT migration_number FROM schema_migrations")
    return {row["migration_number"] for row in rows}


async def cmd_status():
    """Show migration status"""
    conn = await get_db_connection()
    try:
        await ensure_schema_migrations(conn)
        rows = await conn.fetch("""
            SELECT migration_name, migration_number, executed_at, description
            FROM schema_migrations ORDER BY migration_number
        """)
        applied = {row["migration_number"] for row in rows}
        discovered = discover_migrations()
        discovered_nums = {m["number"] for m in discovered}
        pending = sorted(discovered_nums - applied)

        print("\n" + "=" * 60)
        print("MIGRATION STATUS")
        print("=" * 60)
        print(f"Total discovered: {len(discovered)}")
        print(f"Applied: {len(applied)}")
        print(f"Pending: {len(pending)}")

        if rows:
            print("\n Applied Migrations:")
            print(f"{'#':<6} {'Name':<40} {'Applied At':<20}")
            print("-" * 70)
            for row in rows:
                executed = row["executed_at"].strftime("%Y-%m-%d %H:%M") if row["executed_at"] else "N/A"
                print(f"{row['migration_number']:<6} {row['migration_name'][:40]:<40} {executed:<20}")

        if pending:
            print(f"\n Pending Migrations: {pending}")
        else:
            print("\n All migrations applied!")
    finally:
        await conn.close()


async def cmd_list():
    """List all migrations"""
    discovered = discover_migrations()
    conn = await get_db_connection()
    try:
        applied = await get_applied_migrations(conn)
        print("\n" + "=" * 70)
        print("ALL MIGRATIONS")
        print("=" * 70)
        print(f"{'#':<6} {'File':<50} {'Status':<10} {'Rollback':<8}")
        print("-" * 80)

        for m in discovered:
            status = "Applied" if m["number"] in applied else "Pending"
            module = load_migration_module(m)
            has_rollback = "Yes" if module and hasattr(module, "rollback") else "No"
            print(f"{m['number']:<6} {m['file'][:50]:<50} {status:<10} {has_rollback:<8}")
    finally:
        await conn.close()


async def cmd_apply_all(dry_run: bool = False):
    """Apply all pending migrations"""
    discovered = discover_migrations()
    conn = await get_db_connection()
    try:
        applied = await get_applied_migrations(conn)
        pending = [m for m in discovered if m["number"] not in applied]

        if not pending:
            print("No pending migrations")
            return

        print(f"\nFound {len(pending)} pending migrations")

        if dry_run:
            print("\nDRY RUN - Would apply:")
            for m in pending:
                print(f"  - {m['number']:03d}: {m['file']}")
            return

        for m in pending:
            print(f"\nApplying migration {m['number']}: {m['file']}")
            module = load_migration_module(m)
            if not module or not hasattr(module, "apply"):
                print(f"  Failed to load module or no apply() function")
                continue

            try:
                start = datetime.now()
                await module.apply(conn)
                duration_ms = int((datetime.now() - start).total_seconds() * 1000)

                rollback_sql = None
                if hasattr(module, "rollback"):
                    rollback_sql = f"-- Python rollback in {m['module']}"

                await conn.execute("""
                    INSERT INTO schema_migrations
                    (migration_name, migration_number, description, execution_time_ms, rollback_sql)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (migration_name) DO NOTHING
                """, f"{m['number']:03d}_{m['file'].replace('.py', '')}", m["number"],
                     f"Migration {m['number']}", duration_ms, rollback_sql)

                print(f"  Applied in {duration_ms}ms")
            except Exception as e:
                print(f"  Failed: {e}")
                raise
    finally:
        await conn.close()


async def cmd_apply(migration_number: int):
    """Apply a specific migration"""
    discovered = discover_migrations()
    target = next((m for m in discovered if m["number"] == migration_number), None)

    if not target:
        print(f"Migration {migration_number} not found")
        return

    conn = await get_db_connection()
    try:
        applied = await get_applied_migrations(conn)
        if migration_number in applied:
            print(f"Migration {migration_number} already applied")
            return

        module = load_migration_module(target)
        if not module or not hasattr(module, "apply"):
            print(f"Invalid migration module")
            return

        print(f"Applying migration {migration_number}: {target['file']}")
        start = datetime.now()
        await module.apply(conn)
        duration_ms = int((datetime.now() - start).total_seconds() * 1000)

        rollback_sql = f"-- Python rollback in {target['module']}" if hasattr(module, "rollback") else None
        await conn.execute("""
            INSERT INTO schema_migrations
            (migration_name, migration_number, description, execution_time_ms, rollback_sql)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (migration_name) DO NOTHING
        """, f"{migration_number:03d}_{target['file'].replace('.py', '')}", migration_number,
             f"Migration {migration_number}", duration_ms, rollback_sql)

        print(f"Applied migration {migration_number} in {duration_ms}ms")
    finally:
        await conn.close()


async def cmd_rollback(migration_number: int, force: bool = False):
    """Rollback a specific migration"""
    discovered = discover_migrations()
    target = next((m for m in discovered if m["number"] == migration_number), None)

    if not target:
        print(f"Migration {migration_number} not found")
        return

    conn = await get_db_connection()
    try:
        applied = await get_applied_migrations(conn)
        if migration_number not in applied:
            print(f"Migration {migration_number} not applied, nothing to rollback")
            return

        module = load_migration_module(target)
        if not module:
            print(f"Could not load migration module")
            return

        if not hasattr(module, "rollback"):
            print(f"Migration {migration_number} has no rollback() function")
            if not force:
                print("Use --force to remove from tracking anyway")
                return
        else:
            if not force:
                print(f"About to rollback migration {migration_number}: {target['file']}")
                confirm = input("Are you sure? (yes/no): ")
                if confirm.lower() != "yes":
                    print("Cancelled")
                    return

            print(f"Rolling back migration {migration_number}")
            try:
                await module.rollback(conn)
                print(f"Rollback executed")
            except Exception as e:
                print(f"Rollback failed: {e}")
                if not force:
                    return

        await conn.execute("DELETE FROM schema_migrations WHERE migration_number = $1", migration_number)
        print(f"Migration {migration_number} rolled back")
    finally:
        await conn.close()


async def cmd_verify():
    """Verify all migrations have rollback functions"""
    discovered = discover_migrations()
    print("\n" + "=" * 60)
    print("MIGRATION ROLLBACK VERIFICATION")
    print("=" * 60)

    missing_rollback = []
    for m in discovered:
        module = load_migration_module(m)
        has_apply = module and hasattr(module, "apply")
        has_rollback = module and hasattr(module, "rollback")

        status = "OK" if has_apply and has_rollback else "WARN"
        if not has_rollback:
            missing_rollback.append(m["number"])

        apply_status = "Y" if has_apply else "N"
        rollback_status = "Y" if has_rollback else "N"
        print(f"[{status}] {m['number']:03d}: apply={apply_status} rollback={rollback_status} - {m['file']}")

    print()
    if missing_rollback:
        print(f"WARNING: {len(missing_rollback)} migrations missing rollback: {missing_rollback}")
    else:
        print("All migrations have rollback functions!")


def main():
    parser = argparse.ArgumentParser(description="Migration CLI Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    subparsers.add_parser("status", help="Show migration status")
    subparsers.add_parser("list", help="List all migrations")

    apply_all_parser = subparsers.add_parser("apply-all", help="Apply all pending")
    apply_all_parser.add_argument("--dry-run", action="store_true")

    apply_parser = subparsers.add_parser("apply", help="Apply specific migration")
    apply_parser.add_argument("number", type=int)

    rollback_parser = subparsers.add_parser("rollback", help="Rollback migration")
    rollback_parser.add_argument("number", type=int)
    rollback_parser.add_argument("--force", action="store_true")

    subparsers.add_parser("verify", help="Verify rollback functions")

    args = parser.parse_args()

    if args.command == "status":
        asyncio.run(cmd_status())
    elif args.command == "list":
        asyncio.run(cmd_list())
    elif args.command == "apply-all":
        asyncio.run(cmd_apply_all(dry_run=args.dry_run))
    elif args.command == "apply":
        asyncio.run(cmd_apply(args.number))
    elif args.command == "rollback":
        asyncio.run(cmd_rollback(args.number, force=args.force))
    elif args.command == "verify":
        asyncio.run(cmd_verify())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
