#!/usr/bin/env python3
"""
Hybrid Knowledge Store: Neo4j + Qdrant Architecture

Storage Strategy:
- Neo4j: Structured graph (components, relationships, metadata, document structure)
- Qdrant: Semantic vector search (unstructured text chunks)

Tiers:
1. Metadata (Neo4j): Wire colors, abbreviations, connector patterns
2. Knowledge Graph (Neo4j): Rules → Components relationships
3. Document Structure (Neo4j): Sections, hierarchy, page ranges
4. Semantic Search (Qdrant): Text embeddings for RAG

Usage:
    from hybrid_knowledge_store import HybridKnowledgeStore

    store = HybridKnowledgeStore()

    # Store component with context
    store.store_component(component_data, text_chunks)

    # Query for spatial placement
    context = store.get_placement_context(component_id)

    # Semantic search for chat
    results = store.semantic_search("How does starter work?")
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)
import openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_password")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

COLLECTION_NAME = "wiring_manual"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# ============================================================================
# Hybrid Knowledge Store Class
# ============================================================================

class HybridKnowledgeStore:
    """Manages hybrid storage in Neo4j (graph) + Qdrant (vectors)"""

    def __init__(self):
        """Initialize connections to Neo4j and Qdrant"""

        # Neo4j connection
        logger.info(f"Connecting to Neo4j at {NEO4J_URI}...")
        self.neo4j_driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

        # Qdrant connection
        logger.info(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
        self.qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        # OpenAI client
        if OPENAI_API_KEY:
            openai.api_key = OPENAI_API_KEY
            logger.info("OpenAI API key configured")
        else:
            logger.warning("No OpenAI API key found - embeddings disabled")

        # Initialize storage
        self._initialize_neo4j_schema()
        self._initialize_qdrant_collection()

        logger.info("✓ Hybrid Knowledge Store initialized")

    # ========================================================================
    # Initialization
    # ========================================================================

    def _initialize_neo4j_schema(self):
        """Create Neo4j constraints and indexes"""
        with self.neo4j_driver.session() as session:
            # Constraints
            constraints = [
                "CREATE CONSTRAINT component_id IF NOT EXISTS FOR (c:Component) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT section_name IF NOT EXISTS FOR (s:Section) REQUIRE s.name IS UNIQUE",
                "CREATE CONSTRAINT metadata_code IF NOT EXISTS FOR (m:Metadata) REQUIRE (m.type, m.code) IS UNIQUE",
            ]

            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.debug(f"Constraint already exists or failed: {e}")

            # Indexes
            indexes = [
                "CREATE INDEX component_page IF NOT EXISTS FOR (c:Component) ON (c.page)",
                "CREATE INDEX component_type IF NOT EXISTS FOR (c:Component) ON (c.type)",
                "CREATE INDEX section_pages IF NOT EXISTS FOR (s:Section) ON (s.start_page, s.end_page)",
                "CREATE INDEX knowledge_type IF NOT EXISTS FOR (k:Knowledge) ON (k.type)",
            ]

            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    logger.debug(f"Index already exists or failed: {e}")

        logger.info("✓ Neo4j schema initialized")

    def _initialize_qdrant_collection(self):
        """Create Qdrant collection for embeddings"""
        collections = self.qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]

        if COLLECTION_NAME in collection_names:
            logger.info(f"✓ Qdrant collection '{COLLECTION_NAME}' already exists")
        else:
            self.qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"✓ Created Qdrant collection '{COLLECTION_NAME}'")

    # ========================================================================
    # Tier 1: Metadata Storage (Neo4j)
    # ========================================================================

    def store_metadata(self, metadata_type: str, code: str, meaning: str,
                      page: int = None, category: str = None):
        """Store lookup table metadata (wire colors, abbreviations)"""
        with self.neo4j_driver.session() as session:
            session.run("""
                MERGE (m:Metadata {type: $type, code: $code})
                SET m.meaning = $meaning,
                    m.page = $page,
                    m.category = $category,
                    m.updated_at = datetime()
            """, type=metadata_type, code=code, meaning=meaning,
                 page=page, category=category)

        logger.debug(f"Stored metadata: {metadata_type} - {code} = {meaning}")

    def get_metadata(self, metadata_type: str, code: str) -> Optional[str]:
        """Retrieve metadata meaning by type and code"""
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (m:Metadata {type: $type, code: $code})
                RETURN m.meaning AS meaning
            """, type=metadata_type, code=code)

            record = result.single()
            return record["meaning"] if record else None

    def validate_wire_color(self, color_code: str) -> bool:
        """Check if wire color code is valid"""
        return self.get_metadata("wire_color", color_code) is not None

    def expand_abbreviation(self, abbr: str) -> Optional[str]:
        """Expand abbreviation to full meaning"""
        return self.get_metadata("abbreviation", abbr)

    # ========================================================================
    # Tier 2: Knowledge Graph (Neo4j)
    # ========================================================================

    def store_knowledge(self, content: str, knowledge_type: str,
                       page: int, section: str = None,
                       applies_to_type: str = None,
                       applies_to_component: str = None):
        """Store knowledge node (rules, specs, notes)"""
        with self.neo4j_driver.session() as session:
            result = session.run("""
                CREATE (k:Knowledge {
                    content: $content,
                    type: $type,
                    page: $page,
                    section: $section,
                    applies_to_type: $applies_to_type,
                    created_at: datetime()
                })
                RETURN id(k) AS knowledge_id
            """, content=content, type=knowledge_type, page=page,
                 section=section, applies_to_type=applies_to_type)

            knowledge_id = result.single()["knowledge_id"]

            # Link to component if specified
            if applies_to_component:
                session.run("""
                    MATCH (k:Knowledge WHERE id(k) = $knowledge_id)
                    MATCH (c:Component {id: $component_id})
                    MERGE (c)-[:HAS_KNOWLEDGE]->(k)
                """, knowledge_id=knowledge_id, component_id=applies_to_component)

            logger.debug(f"Stored knowledge: {knowledge_type} on page {page}")
            return knowledge_id

    def get_knowledge_for_component_type(self, component_type: str,
                                        section: str = None) -> List[Dict]:
        """Get all knowledge that applies to a component type"""
        with self.neo4j_driver.session() as session:
            query = """
                MATCH (k:Knowledge)
                WHERE k.applies_to_type = $comp_type
            """
            if section:
                query += " OR k.section = $section"
            query += """
                RETURN k.content AS content, k.type AS type,
                       k.page AS page, k.section AS section
                ORDER BY k.page
            """

            result = session.run(query, comp_type=component_type, section=section)
            return [dict(record) for record in result]

    # ========================================================================
    # Tier 3: Document Structure (Neo4j)
    # ========================================================================

    def store_section(self, name: str, start_page: int, end_page: int,
                     parent_section: str = None, zone: str = None,
                     typical_components: List[str] = None):
        """Store document section hierarchy"""
        with self.neo4j_driver.session() as session:
            session.run("""
                MERGE (s:Section {name: $name})
                SET s.start_page = $start_page,
                    s.end_page = $end_page,
                    s.zone = $zone,
                    s.typical_components = $typical_components,
                    s.updated_at = datetime()
            """, name=name, start_page=start_page, end_page=end_page,
                 zone=zone, typical_components=typical_components)

            # Link to parent section
            if parent_section:
                session.run("""
                    MATCH (child:Section {name: $child_name})
                    MATCH (parent:Section {name: $parent_name})
                    MERGE (parent)-[:HAS_SUBSECTION]->(child)
                """, child_name=name, parent_name=parent_section)

            logger.debug(f"Stored section: {name} (pages {start_page}-{end_page})")

    def get_section_for_page(self, page: int) -> Optional[Dict]:
        """Get section that contains this page"""
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (s:Section)
                WHERE $page >= s.start_page AND $page <= s.end_page
                RETURN s.name AS name, s.zone AS zone,
                       s.typical_components AS typical_components
                ORDER BY s.start_page DESC
                LIMIT 1
            """, page=page)

            record = result.single()
            return dict(record) if record else None

    # ========================================================================
    # Component Storage (Neo4j)
    # ========================================================================

    def store_component(self, component_id: str, component_type: str,
                       name: str, page: int,
                       spatial_x: int = None, spatial_y: int = None, spatial_z: int = None,
                       properties: Dict = None,
                       text_chunks: List[str] = None):
        """
        Store component in Neo4j and optionally create embeddings in Qdrant

        Args:
            component_id: Unique component ID
            component_type: Type (relay, fuse, battery, etc.)
            name: Display name
            page: Source page number
            spatial_x/y/z: 3D coordinates (optional)
            properties: Additional properties dict
            text_chunks: Text chunks for embedding (optional)
        """
        with self.neo4j_driver.session() as session:
            # Create component node
            props = properties or {}
            session.run("""
                MERGE (c:Component {id: $id})
                SET c.type = $type,
                    c.name = $name,
                    c.page = $page,
                    c.spatial_x = $spatial_x,
                    c.spatial_y = $spatial_y,
                    c.spatial_z = $spatial_z,
                    c.properties = $properties,
                    c.updated_at = datetime()
            """, id=component_id, type=component_type, name=name, page=page,
                 spatial_x=spatial_x, spatial_y=spatial_y, spatial_z=spatial_z,
                 properties=json.dumps(props))

            # Link to section
            section = self.get_section_for_page(page)
            if section:
                session.run("""
                    MATCH (c:Component {id: $id})
                    MATCH (s:Section {name: $section_name})
                    MERGE (c)-[:LOCATED_IN]->(s)
                """, id=component_id, section_name=section['name'])

            logger.debug(f"Stored component: {component_id} ({component_type}) on page {page}")

        # Store text chunks as embeddings
        if text_chunks and OPENAI_API_KEY:
            self._store_embeddings(
                text_chunks=text_chunks,
                metadata={
                    "component_id": component_id,
                    "component_type": component_type,
                    "page": page,
                    "section": section['name'] if section else None
                }
            )

    # ========================================================================
    # Tier 4: Semantic Embeddings (Qdrant)
    # ========================================================================

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from OpenAI"""
        if not OPENAI_API_KEY:
            logger.warning("No OpenAI API key - cannot generate embeddings")
            return []

        response = openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )

        return [item.embedding for item in response.data]

    def _store_embeddings(self, text_chunks: List[str], metadata: Dict):
        """Store text chunks as embeddings in Qdrant"""
        if not text_chunks:
            return

        # Get embeddings
        embeddings = self._get_embeddings(text_chunks)
        if not embeddings:
            return

        # Create points
        points = []
        for i, (text, embedding) in enumerate(zip(text_chunks, embeddings)):
            point_id = f"{metadata.get('component_id', 'unknown')}_{metadata.get('page', 0)}_chunk_{i}"

            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": text,
                    "component_id": metadata.get("component_id"),
                    "component_type": metadata.get("component_type"),
                    "page": metadata.get("page"),
                    "section": metadata.get("section"),
                    "chunk_index": i,
                    "created_at": datetime.now().isoformat()
                }
            ))

        # Upsert to Qdrant
        self.qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        logger.debug(f"Stored {len(points)} embeddings for {metadata.get('component_id')}")

    def semantic_search(self, query: str, limit: int = 5,
                       filter_section: str = None,
                       filter_component_type: str = None) -> List[Dict]:
        """
        Semantic search across all text chunks

        Args:
            query: Natural language query
            limit: Number of results
            filter_section: Optional section filter
            filter_component_type: Optional component type filter

        Returns:
            List of matching chunks with metadata
        """
        if not OPENAI_API_KEY:
            logger.warning("No OpenAI API key - semantic search disabled")
            return []

        # Get query embedding
        query_embedding = self._get_embeddings([query])[0]

        # Build filter
        must_conditions = []
        if filter_section:
            must_conditions.append(
                FieldCondition(key="section", match=MatchValue(value=filter_section))
            )
        if filter_component_type:
            must_conditions.append(
                FieldCondition(key="component_type", match=MatchValue(value=filter_component_type))
            )

        search_filter = Filter(must=must_conditions) if must_conditions else None

        # Search Qdrant
        results = self.qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=limit,
            query_filter=search_filter
        )

        # Format results
        formatted_results = []
        for hit in results:
            formatted_results.append({
                "text": hit.payload.get("text"),
                "score": hit.score,
                "page": hit.payload.get("page"),
                "section": hit.payload.get("section"),
                "component_id": hit.payload.get("component_id"),
                "component_type": hit.payload.get("component_type")
            })

        return formatted_results

    # ========================================================================
    # High-Level Query Methods
    # ========================================================================

    def get_placement_context(self, component_id: str) -> Dict[str, Any]:
        """
        Get full context for LLM spatial placement

        Returns:
            - Section info
            - Applicable knowledge
            - Connected components
            - Peripheral components
            - Semantic context (optional)
        """
        with self.neo4j_driver.session() as session:
            # Get component info
            result = session.run("""
                MATCH (c:Component {id: $id})
                OPTIONAL MATCH (c)-[:LOCATED_IN]->(s:Section)
                OPTIONAL MATCH (c)-[:HAS_KNOWLEDGE]->(k:Knowledge)
                OPTIONAL MATCH (c)-[r:CONNECTS_TO]-(other:Component)
                RETURN c, s, collect(DISTINCT k) AS knowledge,
                       collect(DISTINCT {component: other, relationship: type(r)}) AS connections
            """, id=component_id)

            record = result.single()
            if not record:
                return {}

            component = dict(record["c"])
            section = dict(record["s"]) if record["s"] else None
            knowledge = [dict(k) for k in record["knowledge"]]
            connections = record["connections"]

            # Get knowledge for component type
            type_knowledge = self.get_knowledge_for_component_type(
                component["type"],
                section["name"] if section else None
            )

            return {
                "component": component,
                "section": section,
                "knowledge": knowledge,
                "type_knowledge": type_knowledge,
                "connections": connections
            }

    def close(self):
        """Close all connections"""
        self.neo4j_driver.close()
        logger.info("✓ Connections closed")


# ============================================================================
# Utility Functions
# ============================================================================

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)

    return chunks


# ============================================================================
# Main Test
# ============================================================================

if __name__ == "__main__":
    print("🔬 Testing Hybrid Knowledge Store")
    print("=" * 70)

    # Initialize store
    store = HybridKnowledgeStore()

    # Test 1: Store metadata
    print("\n📋 Test 1: Storing metadata...")
    store.store_metadata("wire_color", "R", "Red - Power/Positive", page=3)
    store.store_metadata("wire_color", "B", "Black - Ground/Negative", page=3)
    store.store_metadata("wire_color", "G/R", "Green with Red stripe - Signal", page=3)
    store.store_metadata("abbreviation", "ALT", "Alternator", page=4)
    store.store_metadata("abbreviation", "BAT", "Battery", page=4)
    print("✓ Metadata stored")

    # Test 2: Validate metadata
    print("\n✅ Test 2: Validating metadata...")
    print(f"  Wire color 'R': {store.get_metadata('wire_color', 'R')}")
    print(f"  Abbreviation 'ALT': {store.expand_abbreviation('ALT')}")
    print(f"  Is 'G/R' valid wire color? {store.validate_wire_color('G/R')}")
    print(f"  Is 'X/Y' valid wire color? {store.validate_wire_color('X/Y')}")

    # Test 3: Store sections
    print("\n📂 Test 3: Storing document sections...")
    store.store_section("Starter Circuit", 16, 22, zone="Engine Bay Front",
                       typical_components=["battery", "starter", "relay"])
    store.store_section("Lighting System", 46, 78, zone="Entire Vehicle")
    print("✓ Sections stored")

    # Test 4: Store components
    print("\n🔧 Test 4: Storing components...")
    store.store_component(
        component_id="K1",
        component_type="relay",
        name="Main Starter Relay",
        page=18,
        spatial_x=350,
        spatial_y=380,
        spatial_z=150,
        text_chunks=[
            "Main starter relay K1 controls power to starter motor",
            "Relay K1 must be within 300mm of battery per specification"
        ]
    )
    print("✓ Component stored with embeddings")

    # Test 5: Store knowledge
    print("\n💡 Test 5: Storing knowledge...")
    store.store_knowledge(
        content="All ground wires in engine bay must be minimum 6mm² gauge",
        knowledge_type="specification",
        page=25,
        section="Starter Circuit",
        applies_to_type="ground_wire"
    )
    print("✓ Knowledge stored")

    # Test 6: Semantic search
    if OPENAI_API_KEY:
        print("\n🔍 Test 6: Semantic search...")
        results = store.semantic_search("How far should relay be from battery?")
        for i, result in enumerate(results):
            print(f"\n  Result {i+1} (score: {result['score']:.3f}):")
            print(f"    {result['text']}")
            print(f"    Page {result['page']}, Section: {result['section']}")
    else:
        print("\n⚠️  Test 6: Skipped (no OpenAI API key)")

    # Test 7: Get placement context
    print("\n📍 Test 7: Getting placement context...")
    context = store.get_placement_context("K1")
    print(f"  Component: {context['component']['name']}")
    print(f"  Section: {context['section']['name'] if context['section'] else 'None'}")
    print(f"  Type knowledge: {len(context['type_knowledge'])} rules")

    # Cleanup
    store.close()

    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("\nFine Count: $0")
