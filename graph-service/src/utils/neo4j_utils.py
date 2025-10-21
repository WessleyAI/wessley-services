"""Neo4j database utilities and connection management"""

import asyncio
from typing import Dict, List, Any, Optional, Union
from neo4j import GraphDatabase, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
import logging

logger = logging.getLogger(__name__)

class Neo4jClient:
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver = None
    
    async def connect(self):
        """Establish connection to Neo4j database"""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            # Test connection
            await self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.uri}")
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    async def close(self):
        """Close database connection"""
        if self.driver:
            await self.driver.close()
            logger.info("Neo4j connection closed")
    
    async def run(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute Cypher query and return results"""
        if not self.driver:
            await self.connect()
        
        async with self.driver.session(database=self.database) as session:
            try:
                result = await session.run(query, parameters or {})
                records = await result.data()
                return records
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                logger.error(f"Query: {query}")
                logger.error(f"Parameters: {parameters}")
                raise
    
    async def run_transaction(self, transaction_func, **kwargs):
        """Execute queries within a transaction"""
        if not self.driver:
            await self.connect()
        
        async with self.driver.session(database=self.database) as session:
            return await session.execute_write(transaction_func, **kwargs)
    
    async def get_constraints(self) -> List[Dict[str, Any]]:
        """Get all database constraints"""
        query = "SHOW CONSTRAINTS"
        return await self.run(query)
    
    async def get_indexes(self) -> List[Dict[str, Any]]:
        """Get all database indexes"""
        query = "SHOW INDEXES"
        return await self.run(query)
    
    async def create_constraint(self, label: str, property_name: str, constraint_type: str = "UNIQUE"):
        """Create constraint on node property"""
        if constraint_type.upper() == "UNIQUE":
            query = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{property_name} IS UNIQUE"
        elif constraint_type.upper() == "EXISTS":
            query = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{property_name} IS NOT NULL"
        else:
            raise ValueError(f"Unsupported constraint type: {constraint_type}")
        
        await self.run(query)
        logger.info(f"Created {constraint_type} constraint on {label}.{property_name}")
    
    async def create_index(self, label: str, property_name: str):
        """Create index on node property"""
        query = f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.{property_name})"
        await self.run(query)
        logger.info(f"Created index on {label}.{property_name}")
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        queries = {
            "node_count": "MATCH (n) RETURN count(n) as count",
            "relationship_count": "MATCH ()-[r]->() RETURN count(r) as count",
            "label_stats": """
                MATCH (n) 
                RETURN labels(n) as labels, count(n) as count 
                ORDER BY count DESC
            """,
            "relationship_stats": """
                MATCH ()-[r]->() 
                RETURN type(r) as type, count(r) as count 
                ORDER BY count DESC
            """
        }
        
        stats = {}
        for stat_name, query in queries.items():
            try:
                result = await self.run(query)
                stats[stat_name] = result
            except Exception as e:
                logger.warning(f"Failed to get {stat_name}: {e}")
                stats[stat_name] = []
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on database"""
        try:
            # Test basic connectivity
            result = await self.run("RETURN 1 as test")
            
            # Get basic stats
            stats = await self.get_database_stats()
            
            return {
                "status": "healthy",
                "connected": True,
                "database": self.database,
                "node_count": stats.get("node_count", [{}])[0].get("count", 0),
                "relationship_count": stats.get("relationship_count", [{}])[0].get("count", 0)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e)
            }


class Neo4jConnectionManager:
    """Singleton connection manager for Neo4j"""
    _instance = None
    _client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Neo4jConnectionManager, cls).__new__(cls)
        return cls._instance
    
    def initialize(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """Initialize the Neo4j client"""
        self._client = Neo4jClient(uri, username, password, database)
    
    async def get_client(self) -> Neo4jClient:
        """Get the Neo4j client instance"""
        if self._client is None:
            raise RuntimeError("Neo4j client not initialized")
        
        if self._client.driver is None:
            await self._client.connect()
        
        return self._client
    
    async def close(self):
        """Close the connection"""
        if self._client:
            await self._client.close()


# Dependency injection helper
async def get_neo4j_client() -> Neo4jClient:
    """FastAPI dependency for Neo4j client"""
    manager = Neo4jConnectionManager()
    return await manager.get_client()


class CypherQueryBuilder:
    """Helper class for building Cypher queries"""
    
    @staticmethod
    def match_nodes(label: str, properties: Dict[str, Any] = None) -> str:
        """Build MATCH clause for nodes"""
        props_str = ""
        if properties:
            props_list = [f"{k}: ${k}" for k in properties.keys()]
            props_str = f" {{{', '.join(props_list)}}}"
        
        return f"MATCH (n:{label}{props_str})"
    
    @staticmethod
    def create_node(label: str, properties: Dict[str, str] = None) -> str:
        """Build CREATE clause for nodes"""
        props_str = ""
        if properties:
            props_list = [f"{k}: {v}" for k, v in properties.items()]
            props_str = f" {{{', '.join(props_list)}}}"
        
        return f"CREATE (n:{label}{props_str})"
    
    @staticmethod
    def merge_node(label: str, match_props: Dict[str, str], set_props: Dict[str, str] = None) -> str:
        """Build MERGE clause for nodes"""
        match_str = ", ".join([f"{k}: {v}" for k, v in match_props.items()])
        query = f"MERGE (n:{label} {{{match_str}}})"
        
        if set_props:
            set_str = ", ".join([f"n.{k} = {v}" for k, v in set_props.items()])
            query += f" SET {set_str}"
        
        return query
    
    @staticmethod
    def vehicle_filter(vehicle_signature_param: str = "vehicle_signature") -> str:
        """Add vehicle signature filter"""
        return f"WHERE n.vehicle_signature = ${vehicle_signature_param}"