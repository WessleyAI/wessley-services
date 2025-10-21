"""Advanced Cypher query builder utilities"""

from typing import Dict, List, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass

class MatchType(Enum):
    MATCH = "MATCH"
    OPTIONAL_MATCH = "OPTIONAL MATCH"

class RelationshipDirection(Enum):
    OUTGOING = "->"
    INCOMING = "<-"
    UNDIRECTED = "-"

@dataclass
class NodePattern:
    variable: str
    labels: List[str]
    properties: Dict[str, Any] = None
    
    def to_cypher(self) -> str:
        labels_str = ":".join(self.labels)
        if self.labels:
            labels_str = ":" + labels_str
        
        props_str = ""
        if self.properties:
            props_list = []
            for key, value in self.properties.items():
                if isinstance(value, str) and value.startswith('$'):
                    props_list.append(f"{key}: {value}")
                else:
                    props_list.append(f"{key}: ${key}")
            props_str = f" {{{', '.join(props_list)}}}"
        
        return f"({self.variable}{labels_str}{props_str})"

@dataclass
class RelationshipPattern:
    variable: Optional[str] = None
    types: List[str] = None
    properties: Dict[str, Any] = None
    direction: RelationshipDirection = RelationshipDirection.OUTGOING
    
    def to_cypher(self) -> str:
        rel_str = ""
        if self.variable or self.types or self.properties:
            rel_str = "["
            if self.variable:
                rel_str += self.variable
            if self.types:
                rel_str += ":" + "|".join(self.types)
            if self.properties:
                props_list = []
                for key, value in self.properties.items():
                    if isinstance(value, str) and value.startswith('$'):
                        props_list.append(f"{key}: {value}")
                    else:
                        props_list.append(f"{key}: ${key}")
                rel_str += f" {{{', '.join(props_list)}}}"
            rel_str += "]"
        
        if self.direction == RelationshipDirection.OUTGOING:
            return f"-{rel_str}->"
        elif self.direction == RelationshipDirection.INCOMING:
            return f"<-{rel_str}-"
        else:
            return f"-{rel_str}-"

class CypherQueryBuilder:
    def __init__(self):
        self.query_parts = []
        self.parameters = {}
        self.return_clauses = []
        self.where_clauses = []
        self.order_by_clauses = []
        self.limit_value = None
        self.skip_value = None
    
    def match(self, pattern: str, match_type: MatchType = MatchType.MATCH) -> 'CypherQueryBuilder':
        """Add MATCH clause"""
        self.query_parts.append(f"{match_type.value} {pattern}")
        return self
    
    def match_node(self, node: NodePattern, match_type: MatchType = MatchType.MATCH) -> 'CypherQueryBuilder':
        """Add MATCH clause for node"""
        self.query_parts.append(f"{match_type.value} {node.to_cypher()}")
        if node.properties:
            self.parameters.update(node.properties)
        return self
    
    def match_path(self, start_node: NodePattern, relationship: RelationshipPattern, 
                   end_node: NodePattern, match_type: MatchType = MatchType.MATCH) -> 'CypherQueryBuilder':
        """Add MATCH clause for path"""
        path = f"{start_node.to_cypher()}{relationship.to_cypher()}{end_node.to_cypher()}"
        self.query_parts.append(f"{match_type.value} {path}")
        
        # Add parameters
        if start_node.properties:
            self.parameters.update(start_node.properties)
        if end_node.properties:
            self.parameters.update(end_node.properties)
        if relationship.properties:
            self.parameters.update(relationship.properties)
        
        return self
    
    def where(self, condition: str) -> 'CypherQueryBuilder':
        """Add WHERE condition"""
        self.where_clauses.append(condition)
        return self
    
    def where_vehicle_signature(self, variable: str, param_name: str = "vehicle_signature") -> 'CypherQueryBuilder':
        """Add vehicle signature filter"""
        self.where_clauses.append(f"{variable}.vehicle_signature = ${param_name}")
        return self
    
    def where_in(self, field: str, param_name: str) -> 'CypherQueryBuilder':
        """Add IN condition"""
        self.where_clauses.append(f"{field} IN ${param_name}")
        return self
    
    def where_exists(self, pattern: str) -> 'CypherQueryBuilder':
        """Add EXISTS condition"""
        self.where_clauses.append(f"EXISTS({pattern})")
        return self
    
    def create(self, pattern: str) -> 'CypherQueryBuilder':
        """Add CREATE clause"""
        self.query_parts.append(f"CREATE {pattern}")
        return self
    
    def merge(self, pattern: str) -> 'CypherQueryBuilder':
        """Add MERGE clause"""
        self.query_parts.append(f"MERGE {pattern}")
        return self
    
    def set_properties(self, variable: str, properties: Dict[str, str]) -> 'CypherQueryBuilder':
        """Add SET clause for properties"""
        set_clauses = [f"{variable}.{key} = {value}" for key, value in properties.items()]
        self.query_parts.append(f"SET {', '.join(set_clauses)}")
        return self
    
    def set_property(self, variable: str, property_name: str, value: str) -> 'CypherQueryBuilder':
        """Add SET clause for single property"""
        self.query_parts.append(f"SET {variable}.{property_name} = {value}")
        return self
    
    def delete(self, variables: Union[str, List[str]]) -> 'CypherQueryBuilder':
        """Add DELETE clause"""
        if isinstance(variables, str):
            variables = [variables]
        self.query_parts.append(f"DELETE {', '.join(variables)}")
        return self
    
    def detach_delete(self, variables: Union[str, List[str]]) -> 'CypherQueryBuilder':
        """Add DETACH DELETE clause"""
        if isinstance(variables, str):
            variables = [variables]
        self.query_parts.append(f"DETACH DELETE {', '.join(variables)}")
        return self
    
    def with_clause(self, *variables: str, **expressions) -> 'CypherQueryBuilder':
        """Add WITH clause"""
        items = list(variables)
        for alias, expr in expressions.items():
            items.append(f"{expr} as {alias}")
        self.query_parts.append(f"WITH {', '.join(items)}")
        return self
    
    def unwind(self, list_expr: str, variable: str) -> 'CypherQueryBuilder':
        """Add UNWIND clause"""
        self.query_parts.append(f"UNWIND {list_expr} as {variable}")
        return self
    
    def return_clause(self, *items: str) -> 'CypherQueryBuilder':
        """Add items to RETURN clause"""
        self.return_clauses.extend(items)
        return self
    
    def return_distinct(self, *items: str) -> 'CypherQueryBuilder':
        """Add RETURN DISTINCT clause"""
        self.return_clauses.extend(items)
        if not any("DISTINCT" in part for part in self.query_parts):
            self.query_parts.append("RETURN DISTINCT")
        return self
    
    def order_by(self, *fields: str) -> 'CypherQueryBuilder':
        """Add ORDER BY clause"""
        self.order_by_clauses.extend(fields)
        return self
    
    def limit(self, count: int) -> 'CypherQueryBuilder':
        """Add LIMIT clause"""
        self.limit_value = count
        return self
    
    def skip(self, count: int) -> 'CypherQueryBuilder':
        """Add SKIP clause"""
        self.skip_value = count
        return self
    
    def add_parameter(self, name: str, value: Any) -> 'CypherQueryBuilder':
        """Add query parameter"""
        self.parameters[name] = value
        return self
    
    def build(self) -> tuple[str, Dict[str, Any]]:
        """Build the complete Cypher query"""
        query_parts = self.query_parts.copy()
        
        # Add WHERE clause
        if self.where_clauses:
            query_parts.append(f"WHERE {' AND '.join(self.where_clauses)}")
        
        # Add RETURN clause if not already present
        if self.return_clauses and not any("RETURN" in part for part in query_parts):
            query_parts.append(f"RETURN {', '.join(self.return_clauses)}")
        
        # Add ORDER BY clause
        if self.order_by_clauses:
            query_parts.append(f"ORDER BY {', '.join(self.order_by_clauses)}")
        
        # Add SKIP clause
        if self.skip_value is not None:
            query_parts.append(f"SKIP {self.skip_value}")
        
        # Add LIMIT clause
        if self.limit_value is not None:
            query_parts.append(f"LIMIT {self.limit_value}")
        
        query = "\n".join(query_parts)
        return query, self.parameters


class QueryTemplates:
    """Pre-built query templates for common operations"""
    
    @staticmethod
    def find_component_by_id() -> tuple[str, Dict[str, Any]]:
        """Template for finding component by ID"""
        return CypherQueryBuilder() \
            .match_node(NodePattern("c", ["Component"], {"id": "$component_id", "vehicle_signature": "$vehicle_signature"})) \
            .return_clause("c") \
            .build()
    
    @staticmethod
    def find_connected_components() -> tuple[str, Dict[str, Any]]:
        """Template for finding connected components"""
        start_node = NodePattern("start", ["Component"], {"id": "$component_id", "vehicle_signature": "$vehicle_signature"})
        relationship = RelationshipPattern("r", ["CONNECTED_TO"], direction=RelationshipDirection.UNDIRECTED)
        end_node = NodePattern("connected", ["Component"])
        
        return CypherQueryBuilder() \
            .match_path(start_node, relationship, end_node) \
            .where_vehicle_signature("connected") \
            .return_clause("connected", "r") \
            .build()
    
    @staticmethod
    def get_circuit_components() -> tuple[str, Dict[str, Any]]:
        """Template for getting all components in a circuit"""
        circuit_node = NodePattern("circuit", ["Circuit"], {"id": "$circuit_id", "vehicle_signature": "$vehicle_signature"})
        relationship = RelationshipPattern("r", ["PART_OF"], direction=RelationshipDirection.INCOMING)
        component_node = NodePattern("comp", ["Component"])
        
        return CypherQueryBuilder() \
            .match_path(component_node, relationship, circuit_node) \
            .return_clause("comp", "r") \
            .order_by("comp.name") \
            .build()
    
    @staticmethod
    def bulk_create_components() -> tuple[str, Dict[str, Any]]:
        """Template for bulk creating components"""
        return CypherQueryBuilder() \
            .unwind("$components", "comp") \
            .merge("(c:Component {id: comp.id, vehicle_signature: comp.vehicle_signature})") \
            .set_properties("c", {"updated_at": "datetime()"}) \
            .set_property("c", "+", "comp") \
            .return_clause("count(c) as created_count") \
            .build()
    
    @staticmethod
    def delete_vehicle_data() -> tuple[str, Dict[str, Any]]:
        """Template for deleting all vehicle data"""
        return CypherQueryBuilder() \
            .match_node(NodePattern("n", [], {"vehicle_signature": "$vehicle_signature"})) \
            .detach_delete("n") \
            .return_clause("count(n) as deleted_count") \
            .build()
    
    @staticmethod
    def get_system_overview() -> tuple[str, Dict[str, Any]]:
        """Template for system overview statistics"""
        builder = CypherQueryBuilder()
        builder.match_node(NodePattern("c", ["Component"], {"vehicle_signature": "$vehicle_signature"}))
        builder.with_clause("c")
        builder.match_node(NodePattern("circuit", ["Circuit"], {"vehicle_signature": "$vehicle_signature"}), MatchType.OPTIONAL_MATCH)
        builder.return_clause(
            "count(DISTINCT c) as total_components",
            "count(DISTINCT circuit) as total_circuits",
            "count(DISTINCT c.type) as component_types",
            "collect(DISTINCT c.type) as types_list"
        )
        return builder.build()