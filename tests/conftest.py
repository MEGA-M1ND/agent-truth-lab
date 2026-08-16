"""Shared fixtures: a freshly seeded in-memory environment per test."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest

from agent_truth_lab.environment import db

DEFAULT_SEED = 42


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    connection = db.connect(":memory:")
    db.init_db(connection)
    db.seed(connection, DEFAULT_SEED)
    yield connection
    connection.close()


@pytest.fixture
def clock() -> db.SimClock:
    return db.SimClock()
