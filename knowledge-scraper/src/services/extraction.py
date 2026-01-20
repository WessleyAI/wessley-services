"""LLM-based knowledge extraction from scraped posts."""

import json
from typing import Any

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import settings
from src.core.logging import get_logger
from src.core.schemas import (
    ComponentConnection,
    ComponentLocation,
    KnowledgeEntry,
    KnowledgeType,
    ScrapedPost,
    VehicleSignature,
)

logger = get_logger(__name__)


EXTRACTION_PROMPT = """You are an expert automotive electrician analyzing a forum post or video transcript.

Extract structured knowledge about automotive electrical systems from the following text.
Focus on:
1. Component locations (where components are located in the vehicle)
2. Connections (how components are wired together)
3. Failure patterns (common issues and symptoms)
4. Specifications (part numbers, voltages, amperages)

Text to analyze:
---
{text}
---

Return a JSON object with the following structure:
{{
    "vehicle": {{
        "make": "string or null",
        "model": "string or null",
        "year_start": "number or null",
        "year_end": "number or null"
    }},
    "knowledge": [
        {{
            "type": "location|connection|failure|specification",
            "component_type": "string (e.g., relay, fuse, motor, switch, BCM)",
            "location": {{
                "zone": "engine_bay|cabin|door|trunk|under_dash",
                "sub_zone": "string (e.g., fuse_box, glove_box, door_panel)",
                "side": "driver|passenger|center|null",
                "access": "easy|moderate|difficult"
            }},
            "connections": [
                {{
                    "source": "component name",
                    "target": "component name",
                    "connection_type": "power|ground|signal|control",
                    "wire_color": "string or null",
                    "wire_gauge": "string or null"
                }}
            ],
            "symptom": "string or null (for failure type)",
            "root_cause": "string or null",
            "solution": "string or null",
            "cost_estimate": "number or null",
            "confidence": "number 0-1"
        }}
    ]
}}

Only extract information that is explicitly stated or strongly implied in the text.
If no automotive electrical knowledge is present, return {{"vehicle": null, "knowledge": []}}.
"""


class KnowledgeExtractor:
    """Extract structured knowledge from scraped posts using LLM."""

    def __init__(self):
        """Initialize the knowledge extractor."""
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        """Get or create OpenAI client."""
        if self._client is None:
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key not configured")
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def _call_llm(self, text: str) -> dict[str, Any]:
        """Call LLM to extract knowledge."""
        client = self._get_client()

        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": "You extract structured automotive electrical knowledge from text. Always respond with valid JSON.",
                },
                {
                    "role": "user",
                    "content": EXTRACTION_PROMPT.format(text=text[:4000]),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=2000,
        )

        content = response.choices[0].message.content
        if not content:
            return {"vehicle": None, "knowledge": []}

        return json.loads(content)

    def _parse_knowledge(
        self,
        data: dict[str, Any],
        post: ScrapedPost,
    ) -> list[KnowledgeEntry]:
        """Parse LLM response into KnowledgeEntry objects."""
        entries = []

        vehicle_data = data.get("vehicle")
        vehicle = None
        if vehicle_data:
            vehicle = VehicleSignature(
                make=vehicle_data.get("make"),
                model=vehicle_data.get("model"),
                year_start=vehicle_data.get("year_start"),
                year_end=vehicle_data.get("year_end"),
            )

        if post.vehicle:
            vehicle = post.vehicle

        for item in data.get("knowledge", []):
            try:
                knowledge_type = KnowledgeType(item.get("type", "specification"))

                location = None
                if item.get("location"):
                    loc_data = item["location"]
                    location = ComponentLocation(
                        component_type=item.get("component_type", "unknown"),
                        zone=loc_data.get("zone", "unknown"),
                        sub_zone=loc_data.get("sub_zone"),
                        side=loc_data.get("side"),
                        access=loc_data.get("access"),
                    )

                connections = []
                for conn_data in item.get("connections", []):
                    connections.append(
                        ComponentConnection(
                            source=conn_data.get("source", "unknown"),
                            target=conn_data.get("target", "unknown"),
                            connection_type=conn_data.get("connection_type", "unknown"),
                            wire_color=conn_data.get("wire_color"),
                            wire_gauge=conn_data.get("wire_gauge"),
                        )
                    )

                entry = KnowledgeEntry(
                    post_id=post.id,
                    knowledge_type=knowledge_type,
                    vehicle=vehicle,
                    component_type=item.get("component_type"),
                    location=location,
                    connections=connections,
                    symptom=item.get("symptom"),
                    root_cause=item.get("root_cause"),
                    solution=item.get("solution"),
                    cost_estimate=item.get("cost_estimate"),
                    confidence=item.get("confidence", 0.5),
                    raw_text=post.content[:500] if post.content else None,
                )
                entries.append(entry)

            except Exception as e:
                logger.error("parse_knowledge_error", error=str(e), item=item)
                continue

        return entries

    async def extract(self, post: ScrapedPost) -> list[KnowledgeEntry]:
        """Extract knowledge from a scraped post.

        Args:
            post: The scraped post to analyze

        Returns:
            List of extracted KnowledgeEntry objects
        """
        if not settings.enable_llm_extraction:
            logger.debug("llm_extraction_disabled")
            return []

        text = f"{post.title or ''}\n\n{post.content}"

        if len(text.strip()) < 50:
            logger.debug("text_too_short", post_id=str(post.id))
            return []

        try:
            data = await self._call_llm(text)
            entries = self._parse_knowledge(data, post)

            logger.info(
                "knowledge_extracted",
                post_id=str(post.id),
                entries_count=len(entries),
            )

            return entries

        except Exception as e:
            logger.error(
                "extraction_error",
                post_id=str(post.id),
                error=str(e),
            )
            return []

    async def batch_extract(
        self,
        posts: list[ScrapedPost],
        max_concurrent: int = 5,
    ) -> list[KnowledgeEntry]:
        """Extract knowledge from multiple posts.

        Args:
            posts: List of posts to analyze
            max_concurrent: Maximum concurrent extractions

        Returns:
            Combined list of all extracted knowledge entries
        """
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrent)
        all_entries = []

        async def extract_with_semaphore(post: ScrapedPost) -> list[KnowledgeEntry]:
            async with semaphore:
                return await self.extract(post)

        tasks = [extract_with_semaphore(post) for post in posts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_entries.extend(result)
            elif isinstance(result, Exception):
                logger.error("batch_extraction_error", error=str(result))

        return all_entries
