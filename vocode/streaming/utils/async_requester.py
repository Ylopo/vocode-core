from typing import Optional

import aiohttp
import httpx
from pydantic import BaseModel

from vocode.streaming.utils.singleton import Singleton


class AsyncRequestor(Singleton):
    def __init__(self, connection_limit: Optional[int] = None):
        # aiohttp sessions and connectors need a running event loop, so they are created on first use
        self.session: Optional[aiohttp.ClientSession] = None
        self.async_client = httpx.AsyncClient()
        self.connection_limit = connection_limit

    def get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            connector = (
                aiohttp.TCPConnector(limit=self.connection_limit) if self.connection_limit else None
            )
            self.session = aiohttp.ClientSession(connector=connector)
        return self.session

    def get_client(self):
        return self.async_client

    async def close_session(self):
        if self.session is not None and not self.session.closed:
            await self.session.close()

    async def close_client(self):
        await self.async_client.aclose()
