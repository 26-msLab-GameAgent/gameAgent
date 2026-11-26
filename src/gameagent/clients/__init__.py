"""Model client adapters."""

from gameagent.clients.http import HttpModelClient
from gameagent.clients.mock import MockModelClient

__all__ = ["HttpModelClient", "MockModelClient"]
