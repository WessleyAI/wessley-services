"""
Neo4j graph database persistence for electrical schematic data.
"""
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
from datetime import datetime
import uuid

try:
    from neo4j import GraphDatabase, AsyncGraphDatabase
    from neo4j.exceptions import ServiceUnavailable, TransientError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None
    AsyncGraphDatabase = None

from ..core.schemas import Component, Net, NetConnection, TextSpan, ComponentType


@dataclass
class GraphNode:
    """Represents a node in the graph database."""
    id: str
    labels: List[str]
    properties: Dict[str, Any]


@dataclass
class GraphRelationship:
    """Represents a relationship in the graph database."""
    start_node_id: str
    end_node_id: str
    type: str
    properties: Dict[str, Any]


@dataclass
class GraphWriteResult:
    """Result of graph write operations."""
    nodes_created: int
    relationships_created: int
    properties_set: int
    constraints_created: int
    indexes_created: int
    execution_time_ms: float
    warnings: List[str]


class Neo4jPersistence:
    """
    Handles persistence of schematic data to Neo4j graph database.

    Graph Schema:
    - (:Vehicle {make, model, year, project_id})
    - (:Component {id, type, value, page, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                   suggested_spatial, spatial_confidence, spatial_zone, spatial_reasoning,
                   spatial_method, spatial_generated_at})
    - (:Pin {name, position})
    - (:Net {name, voltage_level, is_bus, bus_width})
    - (:TextSpan {text, page, confidence, engine})
    - (:Junction {type, position, confidence})

    Relationships:
    - (Vehicle)-[:CONTAINS]->(Component)
    - (Component)-[:HAS_PIN]->(Pin)
    - (Pin)-[:ON_NET]->(Net)
    - (Component)-[:EXTRACTED_FROM]->(TextSpan)
    - (Net)-[:LABELED_BY]->(TextSpan)

    Note: suggested_spatial and related fields are initially null and populated
    by the LLM spatial placement service after ingestion.
    """
    
    def __init__(
        self,
        uri: str = None,
        username: str = None,
        password: str = None,
        database: str = "neo4j"
    ):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            database: Database name
        """
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver not available. Install with: pip install neo4j")
        
        # Get connection details from environment if not provided
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.database = database
        
        self.driver = None
        self.logger = logging.getLogger(__name__)
        
    async def connect(self) -> bool:
        """
        Establish connection to Neo4j database.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            
            # Test connection
            await self.driver.verify_connectivity()
            self.logger.info(f"Connected to Neo4j at {self.uri}")
            
            # Initialize schema
            await self._initialize_schema()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            return False
    
    async def disconnect(self):
        """Close Neo4j connection."""
        if self.driver:
            await self.driver.close()
            self.logger.info("Disconnected from Neo4j")
    
    async def _initialize_schema(self):
        """Create constraints and indexes for the schema."""
        constraints_and_indexes = [
            # Constraints for uniqueness
            "CREATE CONSTRAINT component_id IF NOT EXISTS FOR (c:Component) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT net_name_project IF NOT EXISTS FOR (n:Net) REQUIRE (n.name, n.project_id) IS UNIQUE",
            "CREATE CONSTRAINT pin_component_name IF NOT EXISTS FOR (p:Pin) REQUIRE (p.component_id, p.name) IS UNIQUE",
            "CREATE CONSTRAINT vehicle_project IF NOT EXISTS FOR (v:Vehicle) REQUIRE v.project_id IS UNIQUE",
            
            # Indexes for performance
            "CREATE INDEX component_type IF NOT EXISTS FOR (c:Component) ON (c.type)",
            "CREATE INDEX component_page IF NOT EXISTS FOR (c:Component) ON (c.page)",
            "CREATE INDEX net_voltage IF NOT EXISTS FOR (n:Net) ON (n.voltage_level)",
            "CREATE INDEX textspan_page IF NOT EXISTS FOR (t:TextSpan) ON (t.page)",
            "CREATE INDEX textspan_engine IF NOT EXISTS FOR (t:TextSpan) ON (t.engine)",
            "CREATE INDEX pin_position IF NOT EXISTS FOR (p:Pin) ON (p.position)",
        ]
        
        async with self.driver.session(database=self.database) as session:
            for statement in constraints_and_indexes:
                try:
                    await session.run(statement)
                    self.logger.debug(f"Executed: {statement}")
                except Exception as e:
                    # Constraint/index might already exist
                    if "already exists" not in str(e).lower():
                        self.logger.warning(f"Schema statement failed: {statement} - {e}")
    
    async def store_vehicle_schematic(
        self,
        project_id: uuid.UUID,
        vehicle_info: Dict[str, Any],
        components: List[Component],
        nets: List[Net],
        text_spans: List[TextSpan],
        junctions: List[Dict[str, Any]] = None
    ) -> GraphWriteResult:
        """
        Store complete vehicle schematic data in graph.
        
        Args:
            project_id: Unique project identifier
            vehicle_info: Vehicle metadata (make, model, year)
            components: List of detected components
            nets: List of electrical nets
            text_spans: List of OCR text spans
            junctions: List of junction data
            
        Returns:
            Result summary of graph operations
        """
        start_time = datetime.now()
        result = GraphWriteResult(
            nodes_created=0,
            relationships_created=0,
            properties_set=0,
            constraints_created=0,
            indexes_created=0,
            execution_time_ms=0.0,
            warnings=[]
        )
        
        try:
            async with self.driver.session(database=self.database) as session:
                # Use a transaction for atomicity
                async with session.begin_transaction() as tx:
                    
                    # 1. Create/update vehicle node
                    vehicle_result = await self._create_vehicle_node(tx, project_id, vehicle_info)
                    result.nodes_created += vehicle_result.get('nodes_created', 0)
                    result.properties_set += vehicle_result.get('properties_set', 0)
                    
                    # 2. Create component nodes and relationships
                    comp_result = await self._create_component_nodes(tx, project_id, components)
                    result.nodes_created += comp_result.get('nodes_created', 0)
                    result.relationships_created += comp_result.get('relationships_created', 0)
                    result.properties_set += comp_result.get('properties_set', 0)
                    
                    # 3. Create net nodes and pin relationships
                    net_result = await self._create_net_topology(tx, project_id, nets, components)
                    result.nodes_created += net_result.get('nodes_created', 0)
                    result.relationships_created += net_result.get('relationships_created', 0)
                    result.properties_set += net_result.get('properties_set', 0)
                    
                    # 4. Create text span nodes and relationships
                    text_result = await self._create_text_nodes(tx, project_id, text_spans, components)
                    result.nodes_created += text_result.get('nodes_created', 0)
                    result.relationships_created += text_result.get('relationships_created', 0)
                    result.properties_set += text_result.get('properties_set', 0)
                    
                    # 5. Create junction nodes if provided
                    if junctions:
                        junction_result = await self._create_junction_nodes(tx, project_id, junctions)
                        result.nodes_created += junction_result.get('nodes_created', 0)
                        result.properties_set += junction_result.get('properties_set', 0)
                    
                    # Commit transaction
                    await tx.commit()
            
            # Calculate execution time
            end_time = datetime.now()
            result.execution_time_ms = (end_time - start_time).total_seconds() * 1000
            
            self.logger.info(
                f"Stored schematic data: {result.nodes_created} nodes, "
                f"{result.relationships_created} relationships in {result.execution_time_ms:.2f}ms"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store schematic data: {e}")
            result.warnings.append(f"Storage failed: {e}")
            raise
        
        return result
    
    async def _create_vehicle_node(self, tx, project_id: uuid.UUID, vehicle_info: Dict[str, Any]) -> Dict[str, int]:
        """Create or update vehicle node."""
        query = """
        MERGE (v:Vehicle {project_id: $project_id})
        SET v.make = $make,
            v.model = $model,
            v.year = $year,
            v.updated_at = datetime()
        RETURN v
        """
        
        result = await tx.run(query, 
            project_id=str(project_id),
            make=vehicle_info.get('make'),
            model=vehicle_info.get('model'),
            year=vehicle_info.get('year')
        )
        
        summary = await result.consume()
        return {
            'nodes_created': summary.counters.nodes_created,
            'properties_set': summary.counters.properties_set
        }
    
    async def _create_component_nodes(self, tx, project_id: uuid.UUID, components: List[Component]) -> Dict[str, int]:
        """Create component nodes and their pins."""
        if not components:
            return {'nodes_created': 0, 'relationships_created': 0, 'properties_set': 0}
        
        # Batch create components
        component_data = []
        pin_data = []
        
        for component in components:
            # Component node data
            comp_props = {
                'id': component.id,
                'project_id': str(project_id),
                'type': component.type.value if hasattr(component.type, 'value') else str(component.type),
                'value': component.value,
                'page': component.page,
                'bbox_x1': component.bbox[0] if component.bbox else 0,
                'bbox_y1': component.bbox[1] if component.bbox else 0,
                'bbox_x2': component.bbox[2] if component.bbox else 0,
                'bbox_y2': component.bbox[3] if component.bbox else 0,
                'confidence': component.confidence,
                'created_at': datetime.now().isoformat(),
                # Spatial placement fields (to be populated by LLM spatial placer)
                'suggested_spatial': None,  # Will be {x, y, z} dict
                'spatial_confidence': None,
                'spatial_zone': None,
                'spatial_reasoning': None,
                'spatial_method': None,
                'spatial_generated_at': None
            }
            component_data.append(comp_props)
            
            # Pin node data
            for pin in component.pins or []:
                pin_props = {
                    'component_id': component.id,
                    'project_id': str(project_id),
                    'name': pin.name,
                    'page': pin.page,
                    'bbox_x1': pin.bbox[0] if pin.bbox else 0,
                    'bbox_y1': pin.bbox[1] if pin.bbox else 0,
                    'bbox_x2': pin.bbox[2] if pin.bbox else 0,
                    'bbox_y2': pin.bbox[3] if pin.bbox else 0,
                    'position': f"{(pin.bbox[0] + pin.bbox[2])/2},{(pin.bbox[1] + pin.bbox[3])/2}" if pin.bbox else "0,0"
                }
                pin_data.append(pin_props)
        
        # Create components
        comp_query = """
        UNWIND $components AS comp
        MERGE (c:Component {id: comp.id, project_id: comp.project_id})
        SET c += comp
        WITH c
        MATCH (v:Vehicle {project_id: c.project_id})
        MERGE (v)-[:CONTAINS]->(c)
        """
        
        comp_result = await tx.run(comp_query, components=component_data)
        comp_summary = await comp_result.consume()
        
        # Create pins and relationships
        pin_result_summary = {'nodes_created': 0, 'relationships_created': 0, 'properties_set': 0}
        
        if pin_data:
            pin_query = """
            UNWIND $pins AS pin
            MERGE (p:Pin {component_id: pin.component_id, name: pin.name})
            SET p += pin
            WITH p
            MATCH (c:Component {id: p.component_id})
            MERGE (c)-[:HAS_PIN]->(p)
            """
            
            pin_result = await tx.run(pin_query, pins=pin_data)
            pin_summary = await pin_result.consume()
            pin_result_summary = {
                'nodes_created': pin_summary.counters.nodes_created,
                'relationships_created': pin_summary.counters.relationships_created,
                'properties_set': pin_summary.counters.properties_set
            }
        
        return {
            'nodes_created': comp_summary.counters.nodes_created + pin_result_summary['nodes_created'],
            'relationships_created': comp_summary.counters.relationships_created + pin_result_summary['relationships_created'],
            'properties_set': comp_summary.counters.properties_set + pin_result_summary['properties_set']
        }
    
    async def _create_net_topology(self, tx, project_id: uuid.UUID, nets: List[Net], components: List[Component]) -> Dict[str, int]:
        """Create net nodes and pin-to-net relationships."""
        if not nets:
            return {'nodes_created': 0, 'relationships_created': 0, 'properties_set': 0}
        
        # Create net nodes
        net_data = []
        connection_data = []
        
        for net in nets:
            net_props = {
                'name': net.name,
                'project_id': str(project_id),
                'confidence': net.confidence,
                'page_spans': net.page_spans,
                'created_at': datetime.now().isoformat()
            }
            net_data.append(net_props)
            
            # Prepare connection data
            for connection in net.connections:
                connection_data.append({
                    'net_name': net.name,
                    'component_id': connection.component_id,
                    'pin_name': connection.pin,
                    'project_id': str(project_id)
                })
        
        # Create nets
        net_query = """
        UNWIND $nets AS net
        MERGE (n:Net {name: net.name, project_id: net.project_id})
        SET n += net
        """
        
        net_result = await tx.run(net_query, nets=net_data)
        net_summary = await net_result.consume()
        
        # Create pin-to-net relationships
        conn_result_summary = {'relationships_created': 0}
        
        if connection_data:
            conn_query = """
            UNWIND $connections AS conn
            MATCH (n:Net {name: conn.net_name, project_id: conn.project_id})
            MATCH (p:Pin {component_id: conn.component_id, name: conn.pin_name})
            MERGE (p)-[:ON_NET]->(n)
            """
            
            conn_result = await tx.run(conn_query, connections=connection_data)
            conn_summary = await conn_result.consume()
            conn_result_summary['relationships_created'] = conn_summary.counters.relationships_created
        
        return {
            'nodes_created': net_summary.counters.nodes_created,
            'relationships_created': net_summary.counters.relationships_created + conn_result_summary['relationships_created'],
            'properties_set': net_summary.counters.properties_set
        }
    
    async def _create_text_nodes(self, tx, project_id: uuid.UUID, text_spans: List[TextSpan], components: List[Component]) -> Dict[str, int]:
        """Create text span nodes and relationships to components."""
        if not text_spans:
            return {'nodes_created': 0, 'relationships_created': 0, 'properties_set': 0}
        
        # Create text span nodes
        text_data = []
        for i, text_span in enumerate(text_spans):
            text_props = {
                'id': f"{project_id}_{text_span.page}_{i}",
                'project_id': str(project_id),
                'text': text_span.text,
                'page': text_span.page,
                'bbox_x1': text_span.bbox[0] if text_span.bbox else 0,
                'bbox_y1': text_span.bbox[1] if text_span.bbox else 0,
                'bbox_x2': text_span.bbox[2] if text_span.bbox else 0,
                'bbox_y2': text_span.bbox[3] if text_span.bbox else 0,
                'confidence': text_span.confidence,
                'engine': text_span.engine.value if hasattr(text_span.engine, 'value') else str(text_span.engine),
                'rotation': getattr(text_span, 'rotation', 0),
                'created_at': datetime.now().isoformat()
            }
            text_data.append(text_props)
        
        text_query = """
        UNWIND $texts AS text
        CREATE (t:TextSpan)
        SET t += text
        """
        
        text_result = await tx.run(text_query, texts=text_data)
        text_summary = await text_result.consume()
        
        # Create relationships between text spans and components based on provenance
        rel_count = 0
        for component in components:
            if component.provenance and 'text_spans' in component.provenance:
                for text_span_id in component.provenance['text_spans']:
                    rel_query = """
                    MATCH (c:Component {id: $component_id})
                    MATCH (t:TextSpan {id: $text_span_id})
                    MERGE (c)-[:EXTRACTED_FROM]->(t)
                    """
                    try:
                        rel_result = await tx.run(rel_query, 
                            component_id=component.id, 
                            text_span_id=text_span_id
                        )
                        rel_summary = await rel_result.consume()
                        rel_count += rel_summary.counters.relationships_created
                    except Exception as e:
                        self.logger.warning(f"Failed to create text relationship: {e}")
        
        return {
            'nodes_created': text_summary.counters.nodes_created,
            'relationships_created': rel_count,
            'properties_set': text_summary.counters.properties_set
        }
    
    async def _create_junction_nodes(self, tx, project_id: uuid.UUID, junctions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Create junction nodes."""
        if not junctions:
            return {'nodes_created': 0, 'properties_set': 0}
        
        junction_data = []
        for i, junction in enumerate(junctions):
            junction_props = {
                'id': f"{project_id}_junction_{i}",
                'project_id': str(project_id),
                'type': junction.get('type', 'unknown'),
                'position': f"{junction.get('position', [0, 0])[0]},{junction.get('position', [0, 0])[1]}",
                'confidence': junction.get('confidence', 0.0),
                'connected_lines': junction.get('connected_lines', []),
                'created_at': datetime.now().isoformat()
            }
            junction_data.append(junction_props)
        
        junction_query = """
        UNWIND $junctions AS junction
        CREATE (j:Junction)
        SET j += junction
        """
        
        junction_result = await tx.run(junction_query, junctions=junction_data)
        junction_summary = await junction_result.consume()
        
        return {
            'nodes_created': junction_summary.counters.nodes_created,
            'properties_set': junction_summary.counters.properties_set
        }
    
    async def query_vehicle_components(self, project_id: uuid.UUID) -> List[Dict[str, Any]]:
        """Query all components for a vehicle project."""
        query = """
        MATCH (v:Vehicle {project_id: $project_id})-[:CONTAINS]->(c:Component)
        OPTIONAL MATCH (c)-[:HAS_PIN]->(p:Pin)
        OPTIONAL MATCH (p)-[:ON_NET]->(n:Net)
        RETURN c.id as component_id, c.type as type, c.value as value,
               collect(DISTINCT {pin: p.name, net: n.name}) as connections
        ORDER BY c.id
        """
        
        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, project_id=str(project_id))
            return [record.data() async for record in result]
    
    async def query_net_connectivity(self, project_id: uuid.UUID, net_name: str) -> Dict[str, Any]:
        """Query connectivity information for a specific net."""
        query = """
        MATCH (n:Net {name: $net_name, project_id: $project_id})<-[:ON_NET]-(p:Pin)<-[:HAS_PIN]-(c:Component)
        RETURN n.name as net_name, 
               collect({component: c.id, pin: p.name, type: c.type}) as connections,
               n.confidence as confidence
        """
        
        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, project_id=str(project_id), net_name=net_name)
            record = await result.single()
            return record.data() if record else {}
    
    async def delete_project_data(self, project_id: uuid.UUID) -> bool:
        """Delete all data for a project."""
        query = """
        MATCH (v:Vehicle {project_id: $project_id})
        OPTIONAL MATCH (v)-[:CONTAINS]->(c:Component)
        OPTIONAL MATCH (c)-[:HAS_PIN]->(p:Pin)
        OPTIONAL MATCH (p)-[:ON_NET]->(n:Net {project_id: $project_id})
        OPTIONAL MATCH (t:TextSpan {project_id: $project_id})
        OPTIONAL MATCH (j:Junction {project_id: $project_id})
        DETACH DELETE v, c, p, n, t, j
        """
        
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(query, project_id=str(project_id))
                summary = await result.consume()
                
                self.logger.info(f"Deleted project {project_id}: {summary.counters.nodes_deleted} nodes")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to delete project data: {e}")
            return False
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        queries = {
            'vehicles': "MATCH (v:Vehicle) RETURN count(v) as count",
            'components': "MATCH (c:Component) RETURN count(c) as count",
            'nets': "MATCH (n:Net) RETURN count(n) as count",
            'pins': "MATCH (p:Pin) RETURN count(p) as count",
            'text_spans': "MATCH (t:TextSpan) RETURN count(t) as count",
            'junctions': "MATCH (j:Junction) RETURN count(j) as count"
        }
        
        stats = {}
        
        async with self.driver.session(database=self.database) as session:
            for name, query in queries.items():
                try:
                    result = await session.run(query)
                    record = await result.single()
                    stats[name] = record['count'] if record else 0
                except Exception as e:
                    self.logger.warning(f"Failed to get {name} count: {e}")
                    stats[name] = 0
        
        return stats


# Convenience function for creating Neo4j persistence instance
def create_neo4j_persistence(**kwargs) -> Neo4jPersistence:
    """Create Neo4j persistence instance with environment configuration."""
    return Neo4jPersistence(**kwargs)