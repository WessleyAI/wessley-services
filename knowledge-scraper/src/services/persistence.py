"""Persistence service for storing scraped knowledge."""

from typing import Any
from uuid import UUID

from neo4j import AsyncGraphDatabase
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.core.config import settings
from src.core.logging import get_logger
from src.core.schemas import KnowledgeEntry, ScrapedPost

logger = get_logger(__name__)


class PersistenceService:
    """Service for persisting scraped data to Neo4j and Qdrant."""

    EMBEDDING_DIM = 1536

    def __init__(self):
        """Initialize persistence service."""
        self._neo4j_driver = None
        self._qdrant_client: AsyncQdrantClient | None = None
        self._openai_client = None

    async def initialize(self) -> None:
        """Initialize database connections."""
        if settings.neo4j_uri and settings.neo4j_password:
            self._neo4j_driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
            logger.info("neo4j_connected", uri=settings.neo4j_uri)

        if settings.qdrant_url:
            self._qdrant_client = AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
            )
            await self._ensure_collection()
            logger.info("qdrant_connected", url=settings.qdrant_url)

    async def _ensure_collection(self) -> None:
        """Ensure Qdrant collection exists."""
        if not self._qdrant_client:
            return

        collections = await self._qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if settings.qdrant_collection not in collection_names:
            await self._qdrant_client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(
                    size=self.EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("collection_created", name=settings.qdrant_collection)

    async def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text using OpenAI."""
        from openai import AsyncOpenAI

        if self._openai_client is None:
            self._openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

        response = await self._openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text[:8000],
        )

        return response.data[0].embedding

    async def store_post(self, post: ScrapedPost) -> None:
        """Store a scraped post in the databases.

        Args:
            post: The scraped post to store
        """
        if self._qdrant_client:
            try:
                text = f"{post.title or ''}\n\n{post.content}"
                embedding = await self._get_embedding(text)

                payload = {
                    "source": post.source.value,
                    "source_id": post.source_id,
                    "url": post.url,
                    "title": post.title,
                    "content": post.content[:2000] if post.content else None,
                    "author": post.author,
                    "scraped_at": post.scraped_at.isoformat(),
                }

                if post.vehicle:
                    payload["vehicle_make"] = post.vehicle.make
                    payload["vehicle_model"] = post.vehicle.model
                    payload["vehicle_year_start"] = post.vehicle.year_start
                    payload["vehicle_year_end"] = post.vehicle.year_end

                if post.subreddit:
                    payload["subreddit"] = post.subreddit
                if post.forum_name:
                    payload["forum_name"] = post.forum_name

                await self._qdrant_client.upsert(
                    collection_name=settings.qdrant_collection,
                    points=[
                        PointStruct(
                            id=str(post.id),
                            vector=embedding,
                            payload=payload,
                        )
                    ],
                )

                logger.debug("post_stored_qdrant", post_id=str(post.id))

            except Exception as e:
                logger.error("qdrant_store_error", post_id=str(post.id), error=str(e))

    async def store_knowledge(self, entry: KnowledgeEntry) -> None:
        """Store a knowledge entry in Neo4j.

        Args:
            entry: The knowledge entry to store
        """
        if not self._neo4j_driver:
            return

        try:
            async with self._neo4j_driver.session() as session:
                if entry.location:
                    await session.run(
                        """
                        MERGE (c:Component {type: $component_type})
                        MERGE (z:Zone {name: $zone})
                        MERGE (c)-[:LOCATED_IN {
                            sub_zone: $sub_zone,
                            side: $side,
                            access: $access,
                            confidence: $confidence
                        }]->(z)
                        """,
                        component_type=entry.component_type or "unknown",
                        zone=entry.location.zone,
                        sub_zone=entry.location.sub_zone,
                        side=entry.location.side,
                        access=entry.location.access,
                        confidence=entry.confidence,
                    )

                for conn in entry.connections:
                    await session.run(
                        """
                        MERGE (s:Component {type: $source})
                        MERGE (t:Component {type: $target})
                        MERGE (s)-[:CONNECTED_TO {
                            connection_type: $connection_type,
                            wire_color: $wire_color,
                            wire_gauge: $wire_gauge,
                            confidence: $confidence
                        }]->(t)
                        """,
                        source=conn.source,
                        target=conn.target,
                        connection_type=conn.connection_type,
                        wire_color=conn.wire_color,
                        wire_gauge=conn.wire_gauge,
                        confidence=entry.confidence,
                    )

                if entry.vehicle:
                    await session.run(
                        """
                        MERGE (v:Vehicle {
                            make: $make,
                            model: $model,
                            year_start: $year_start,
                            year_end: $year_end
                        })
                        WITH v
                        MATCH (c:Component {type: $component_type})
                        MERGE (v)-[:HAS_COMPONENT]->(c)
                        """,
                        make=entry.vehicle.make or "unknown",
                        model=entry.vehicle.model or "unknown",
                        year_start=entry.vehicle.year_start or 0,
                        year_end=entry.vehicle.year_end or 9999,
                        component_type=entry.component_type or "unknown",
                    )

                logger.debug("knowledge_stored_neo4j", entry_id=str(entry.id))

        except Exception as e:
            logger.error("neo4j_store_error", entry_id=str(entry.id), error=str(e))

    async def search(
        self,
        query: str,
        vehicle_make: str | None = None,
        vehicle_model: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search for knowledge.

        Args:
            query: Search query
            vehicle_make: Optional filter by make
            vehicle_model: Optional filter by model
            limit: Maximum results

        Returns:
            List of matching knowledge entries
        """
        if not self._qdrant_client:
            return []

        try:
            embedding = await self._get_embedding(query)

            filter_conditions = []
            if vehicle_make:
                filter_conditions.append(
                    {"key": "vehicle_make", "match": {"value": vehicle_make}}
                )
            if vehicle_model:
                filter_conditions.append(
                    {"key": "vehicle_model", "match": {"value": vehicle_model}}
                )

            filter_obj = None
            if filter_conditions:
                filter_obj = {"must": filter_conditions}

            results = await self._qdrant_client.search(
                collection_name=settings.qdrant_collection,
                query_vector=embedding,
                query_filter=filter_obj,
                limit=limit,
            )

            return [
                {
                    "id": str(r.id),
                    "score": r.score,
                    **r.payload,
                }
                for r in results
            ]

        except Exception as e:
            logger.error("search_error", query=query, error=str(e))
            return []

    async def close(self) -> None:
        """Close database connections."""
        if self._neo4j_driver:
            await self._neo4j_driver.close()
            self._neo4j_driver = None

        if self._qdrant_client:
            await self._qdrant_client.close()
            self._qdrant_client = None

        logger.info("persistence_service_closed")
