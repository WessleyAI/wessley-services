"""
Spatial data management service for 3D positioning and layout optimization
"""

import logging
import math
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass

from ..utils.neo4j_utils import Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class Position3D:
    """3D position coordinates"""
    x: float
    y: float
    z: float
    
    def distance_to(self, other: 'Position3D') -> float:
        """Calculate Euclidean distance to another position"""
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )


@dataclass
class BoundingBox:
    """3D bounding box"""
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float
    
    @property
    def center(self) -> Position3D:
        """Get center point of bounding box"""
        return Position3D(
            x=(self.min_x + self.max_x) / 2,
            y=(self.min_y + self.max_y) / 2,
            z=(self.min_z + self.max_z) / 2
        )
    
    @property
    def volume(self) -> float:
        """Calculate volume of bounding box"""
        return (
            (self.max_x - self.min_x) *
            (self.max_y - self.min_y) *
            (self.max_z - self.min_z)
        )


@dataclass
class SpatialCluster:
    """Spatial cluster of components"""
    cluster_id: str
    center: Position3D
    radius: float
    components: List[str]
    density: float
    zone: Optional[str]


@dataclass
class RoutingPath:
    """Wire routing path between components"""
    from_component: str
    to_component: str
    waypoints: List[Position3D]
    total_length: float
    routing_complexity: str
    clearance_violations: List[str]


@dataclass
class SpatialAnalysisResult:
    """Spatial analysis result"""
    vehicle_signature: str
    vehicle_bounds: BoundingBox
    component_distribution: Dict[str, Any]
    spatial_clusters: List[SpatialCluster]
    routing_paths: List[RoutingPath]
    space_utilization: Dict[str, float]
    accessibility_analysis: Dict[str, Any]
    optimization_recommendations: List[str]


class SpatialService:
    """
    Service for spatial data management and 3D layout optimization
    """
    
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client
    
    # ==================== Spatial Data Management ====================
    
    async def update_component_position(
        self,
        component_id: str,
        vehicle_signature: str,
        position: Position3D,
        zone: Optional[str] = None
    ) -> bool:
        """Update component 3D position"""
        
        query = """
        MATCH (c:Component {id: $component_id, vehicle_signature: $vehicle_signature})
        SET c.position = [$x, $y, $z]
        """
        
        params = {
            "component_id": component_id,
            "vehicle_signature": vehicle_signature,
            "x": position.x,
            "y": position.y,
            "z": position.z
        }
        
        if zone:
            query += ", c.zone = $zone"
            params["zone"] = zone
        
        query += " RETURN c.id as updated_id"
        
        result = await self.neo4j.run(query, **params)
        
        if result.records:
            logger.info(f"Updated position for component {component_id}")
            return True
        
        return False
    
    async def batch_update_positions(
        self,
        vehicle_signature: str,
        position_updates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Batch update component positions"""
        
        updated_count = 0
        failed_updates = []
        
        for update in position_updates:
            try:
                component_id = update["component_id"]
                position = Position3D(
                    x=update["x"],
                    y=update["y"],
                    z=update["z"]
                )
                zone = update.get("zone")
                
                success = await self.update_component_position(
                    component_id, vehicle_signature, position, zone
                )
                
                if success:
                    updated_count += 1
                else:
                    failed_updates.append(component_id)
                    
            except Exception as e:
                failed_updates.append({
                    "component_id": update.get("component_id"),
                    "error": str(e)
                })
        
        return {
            "updated_count": updated_count,
            "failed_count": len(failed_updates),
            "failed_updates": failed_updates
        }
    
    async def get_components_with_positions(
        self,
        vehicle_signature: str,
        zone: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all components with spatial positions"""
        
        where_clause = "c.vehicle_signature = $vehicle_signature AND c.position IS NOT NULL"
        params = {"vehicle_signature": vehicle_signature}
        
        if zone:
            where_clause += " AND c.zone = $zone"
            params["zone"] = zone
        
        query = f"""
        MATCH (c:Component)
        WHERE {where_clause}
        
        RETURN c.id as component_id, c.name as component_name, c.type as component_type,
               c.position as position, c.zone as zone,
               c.voltage_rating as voltage_rating, c.current_rating as current_rating
        ORDER BY c.zone, c.type, c.name
        """
        
        result = await self.neo4j.run(query, **params)
        
        components = []
        for record in result.records:
            position_list = record["position"]
            position = None
            if position_list and len(position_list) >= 3:
                position = Position3D(
                    x=position_list[0],
                    y=position_list[1],
                    z=position_list[2]
                )
            
            components.append({
                "component_id": record["component_id"],
                "component_name": record["component_name"],
                "component_type": record["component_type"],
                "position": position,
                "zone": record["zone"],
                "voltage_rating": record["voltage_rating"],
                "current_rating": record["current_rating"]
            })
        
        return components
    
    # ==================== Spatial Analysis ====================
    
    async def analyze_spatial_layout(
        self,
        vehicle_signature: str
    ) -> SpatialAnalysisResult:
        """Perform comprehensive spatial layout analysis"""
        
        # Get components with positions
        components = await self.get_components_with_positions(vehicle_signature)
        
        if not components:
            return SpatialAnalysisResult(
                vehicle_signature=vehicle_signature,
                vehicle_bounds=BoundingBox(0, 0, 0, 0, 0, 0),
                component_distribution={},
                spatial_clusters=[],
                routing_paths=[],
                space_utilization={},
                accessibility_analysis={},
                optimization_recommendations=["No spatial data available for analysis"]
            )
        
        # Calculate vehicle bounds
        vehicle_bounds = self._calculate_vehicle_bounds(components)
        
        # Analyze component distribution
        component_distribution = self._analyze_component_distribution(components)
        
        # Find spatial clusters
        spatial_clusters = await self._find_spatial_clusters(components)
        
        # Analyze wire routing
        routing_paths = await self._analyze_wire_routing(vehicle_signature, components)
        
        # Calculate space utilization
        space_utilization = self._calculate_space_utilization(components, vehicle_bounds)
        
        # Analyze accessibility
        accessibility_analysis = await self._analyze_accessibility(vehicle_signature)
        
        # Generate optimization recommendations
        optimization_recommendations = self._generate_spatial_recommendations(
            component_distribution,
            spatial_clusters,
            routing_paths,
            space_utilization,
            accessibility_analysis
        )
        
        return SpatialAnalysisResult(
            vehicle_signature=vehicle_signature,
            vehicle_bounds=vehicle_bounds,
            component_distribution=component_distribution,
            spatial_clusters=spatial_clusters,
            routing_paths=routing_paths,
            space_utilization=space_utilization,
            accessibility_analysis=accessibility_analysis,
            optimization_recommendations=optimization_recommendations
        )
    
    def _calculate_vehicle_bounds(self, components: List[Dict[str, Any]]) -> BoundingBox:
        """Calculate overall vehicle spatial bounds"""
        
        positions = [comp["position"] for comp in components if comp["position"]]
        
        if not positions:
            return BoundingBox(0, 0, 0, 0, 0, 0)
        
        min_x = min(pos.x for pos in positions)
        max_x = max(pos.x for pos in positions)
        min_y = min(pos.y for pos in positions)
        max_y = max(pos.y for pos in positions)
        min_z = min(pos.z for pos in positions)
        max_z = max(pos.z for pos in positions)
        
        # Add some padding
        padding = 100  # mm
        
        return BoundingBox(
            min_x - padding, max_x + padding,
            min_y - padding, max_y + padding,
            min_z - padding, max_z + padding
        )
    
    def _analyze_component_distribution(self, components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze component distribution across zones and types"""
        
        distribution = {
            "total_components": len(components),
            "by_zone": {},
            "by_type": {},
            "position_coverage": 0.0
        }
        
        # Count by zone
        for comp in components:
            zone = comp.get("zone", "unknown")
            if zone not in distribution["by_zone"]:
                distribution["by_zone"][zone] = 0
            distribution["by_zone"][zone] += 1
        
        # Count by type
        for comp in components:
            comp_type = comp.get("component_type", "unknown")
            if comp_type not in distribution["by_type"]:
                distribution["by_type"][comp_type] = 0
            distribution["by_type"][comp_type] += 1
        
        # Calculate position coverage
        positioned_components = sum(1 for comp in components if comp["position"])
        distribution["position_coverage"] = (
            positioned_components / len(components) * 100 if components else 0
        )
        
        return distribution
    
    async def _find_spatial_clusters(self, components: List[Dict[str, Any]]) -> List[SpatialCluster]:
        """Find spatial clusters of components using simple clustering"""
        
        positioned_components = [comp for comp in components if comp["position"]]
        
        if len(positioned_components) < 2:
            return []
        
        clusters = []
        cluster_radius = 300  # mm
        
        # Simple clustering algorithm
        unprocessed = positioned_components.copy()
        cluster_id = 0
        
        while unprocessed:
            # Start new cluster with first unprocessed component
            seed = unprocessed.pop(0)
            cluster_components = [seed]
            
            # Find nearby components
            i = 0
            while i < len(unprocessed):
                comp = unprocessed[i]
                
                # Check distance to any component in current cluster
                is_nearby = any(
                    comp["position"].distance_to(cluster_comp["position"]) <= cluster_radius
                    for cluster_comp in cluster_components
                )
                
                if is_nearby:
                    cluster_components.append(unprocessed.pop(i))
                else:
                    i += 1
            
            # Create cluster if it has multiple components
            if len(cluster_components) > 1:
                # Calculate cluster center
                center_x = sum(comp["position"].x for comp in cluster_components) / len(cluster_components)
                center_y = sum(comp["position"].y for comp in cluster_components) / len(cluster_components)
                center_z = sum(comp["position"].z for comp in cluster_components) / len(cluster_components)
                center = Position3D(center_x, center_y, center_z)
                
                # Calculate actual radius
                max_distance = max(
                    comp["position"].distance_to(center)
                    for comp in cluster_components
                )
                
                # Calculate density
                cluster_volume = (4/3) * math.pi * (max_distance ** 3)
                density = len(cluster_components) / cluster_volume if cluster_volume > 0 else 0
                
                # Determine zone (most common zone in cluster)
                zones = [comp.get("zone") for comp in cluster_components if comp.get("zone")]
                most_common_zone = max(set(zones), key=zones.count) if zones else None
                
                clusters.append(SpatialCluster(
                    cluster_id=f"cluster_{cluster_id}",
                    center=center,
                    radius=max_distance,
                    components=[comp["component_id"] for comp in cluster_components],
                    density=density,
                    zone=most_common_zone
                ))
                
                cluster_id += 1
        
        return clusters
    
    async def _analyze_wire_routing(
        self,
        vehicle_signature: str,
        components: List[Dict[str, Any]]
    ) -> List[RoutingPath]:
        """Analyze wire routing paths between connected components"""
        
        # Get connections between components
        query = """
        MATCH (c1:Component {vehicle_signature: $vehicle_signature})-[r:CONNECTS_TO]->(c2:Component {vehicle_signature: $vehicle_signature})
        WHERE c1.position IS NOT NULL AND c2.position IS NOT NULL
        
        RETURN c1.id as from_id, c1.position as from_pos,
               c2.id as to_id, c2.position as to_pos,
               r.wire_gauge as wire_gauge, r.wire_color as wire_color
        """
        
        result = await self.neo4j.run(query, vehicle_signature=vehicle_signature)
        
        routing_paths = []
        
        for record in result.records:
            from_pos_list = record["from_pos"]
            to_pos_list = record["to_pos"]
            
            if from_pos_list and to_pos_list and len(from_pos_list) >= 3 and len(to_pos_list) >= 3:
                from_pos = Position3D(from_pos_list[0], from_pos_list[1], from_pos_list[2])
                to_pos = Position3D(to_pos_list[0], to_pos_list[1], to_pos_list[2])
                
                # Simple direct routing for now
                waypoints = [from_pos, to_pos]
                total_length = from_pos.distance_to(to_pos)
                
                # Assess routing complexity
                if total_length > 1000:  # > 1 meter
                    complexity = "high"
                elif total_length > 500:  # > 0.5 meter
                    complexity = "medium"
                else:
                    complexity = "low"
                
                routing_paths.append(RoutingPath(
                    from_component=record["from_id"],
                    to_component=record["to_id"],
                    waypoints=waypoints,
                    total_length=total_length,
                    routing_complexity=complexity,
                    clearance_violations=[]  # TODO: Implement clearance checking
                ))
        
        return routing_paths
    
    def _calculate_space_utilization(
        self,
        components: List[Dict[str, Any]],
        vehicle_bounds: BoundingBox
    ) -> Dict[str, float]:
        """Calculate space utilization metrics"""
        
        # Estimate component volumes (simplified)
        total_component_volume = 0.0
        
        for comp in components:
            # Simplified volume estimation based on component type
            type_volumes = {
                "battery": 5000000,  # mm³
                "alternator": 2000000,
                "ecu": 500000,
                "relay": 50000,
                "fuse": 10000,
                "sensor": 25000,
                "connector": 15000
            }
            
            comp_type = comp.get("component_type", "unknown")
            volume = type_volumes.get(comp_type, 100000)  # Default 100cm³
            total_component_volume += volume
        
        # Calculate utilization
        vehicle_volume = vehicle_bounds.volume
        utilization_percentage = (total_component_volume / vehicle_volume * 100) if vehicle_volume > 0 else 0
        
        return {
            "vehicle_volume_mm3": vehicle_volume,
            "component_volume_mm3": total_component_volume,
            "utilization_percentage": utilization_percentage,
            "available_space_mm3": vehicle_volume - total_component_volume
        }
    
    async def _analyze_accessibility(self, vehicle_signature: str) -> Dict[str, Any]:
        """Analyze component accessibility for maintenance"""
        
        query = """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})-[r:LOCATED_IN]->(z:Zone)
        
        RETURN c.id as component_id, c.type as component_type,
               r.accessibility as accessibility,
               r.service_access_required as service_required,
               z.access_level as zone_access_level
        """
        
        result = await self.neo4j.run(query, vehicle_signature=vehicle_signature)
        
        accessibility_stats = {
            "total_analyzed": 0,
            "accessibility_distribution": {
                "easy": 0,
                "moderate": 0,
                "difficult": 0
            },
            "service_access_issues": [],
            "recommendations": []
        }
        
        for record in result.records:
            accessibility_stats["total_analyzed"] += 1
            
            accessibility = record.get("accessibility", "moderate")
            if accessibility in accessibility_stats["accessibility_distribution"]:
                accessibility_stats["accessibility_distribution"][accessibility] += 1
            
            # Check for service access issues
            if record.get("service_required") and accessibility == "difficult":
                accessibility_stats["service_access_issues"].append({
                    "component_id": record["component_id"],
                    "component_type": record["component_type"],
                    "issue": "requires_service_but_difficult_access"
                })
        
        # Generate recommendations
        difficult_access = accessibility_stats["accessibility_distribution"]["difficult"]
        if difficult_access > 0:
            accessibility_stats["recommendations"].append(
                f"Consider relocating {difficult_access} components with difficult access"
            )
        
        return accessibility_stats
    
    def _generate_spatial_recommendations(
        self,
        component_distribution: Dict[str, Any],
        spatial_clusters: List[SpatialCluster],
        routing_paths: List[RoutingPath],
        space_utilization: Dict[str, float],
        accessibility_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate spatial optimization recommendations"""
        
        recommendations = []
        
        # Position coverage recommendations
        coverage = component_distribution.get("position_coverage", 0)
        if coverage < 80:
            recommendations.append(
                f"Only {coverage:.1f}% of components have position data - complete spatial mapping needed"
            )
        
        # Clustering recommendations
        dense_clusters = [c for c in spatial_clusters if c.density > 0.001]  # Arbitrary threshold
        if len(dense_clusters) > 2:
            recommendations.append(
                f"Found {len(dense_clusters)} dense component clusters - consider distribution optimization"
            )
        
        # Routing recommendations
        long_routes = [r for r in routing_paths if r.total_length > 1000]
        if len(long_routes) > 5:
            recommendations.append(
                f"{len(long_routes)} wire routes exceed 1m length - consider component repositioning"
            )
        
        # Space utilization recommendations
        utilization = space_utilization.get("utilization_percentage", 0)
        if utilization > 80:
            recommendations.append("High space utilization detected - optimize component placement")
        elif utilization < 20:
            recommendations.append("Low space utilization - opportunity for more compact design")
        
        # Accessibility recommendations
        difficult_access = accessibility_analysis.get("accessibility_distribution", {}).get("difficult", 0)
        if difficult_access > 0:
            recommendations.append(
                f"{difficult_access} components have difficult access - improve serviceability"
            )
        
        return recommendations
    
    # ==================== 3D Model Generation Support ====================
    
    async def get_3d_model_data(
        self,
        vehicle_signature: str,
        include_wire_harnesses: bool = True
    ) -> Dict[str, Any]:
        """Get spatial data formatted for 3D model generation"""
        
        components = await self.get_components_with_positions(vehicle_signature)
        
        # Format component data for 3D rendering
        component_data = []
        for comp in components:
            if comp["position"]:
                component_data.append({
                    "id": comp["component_id"],
                    "name": comp["component_name"],
                    "type": comp["component_type"],
                    "position": {
                        "x": comp["position"].x,
                        "y": comp["position"].y,
                        "z": comp["position"].z
                    },
                    "zone": comp["zone"],
                    "electrical_properties": {
                        "voltage_rating": comp["voltage_rating"],
                        "current_rating": comp["current_rating"]
                    }
                })
        
        # Get wire harness data if requested
        harness_data = []
        if include_wire_harnesses:
            routing_paths = await self._analyze_wire_routing(vehicle_signature, components)
            
            for path in routing_paths:
                harness_data.append({
                    "from_component": path.from_component,
                    "to_component": path.to_component,
                    "waypoints": [
                        {"x": wp.x, "y": wp.y, "z": wp.z}
                        for wp in path.waypoints
                    ],
                    "length": path.total_length,
                    "complexity": path.routing_complexity
                })
        
        # Get vehicle bounds
        vehicle_bounds = self._calculate_vehicle_bounds(components)
        
        return {
            "vehicle_signature": vehicle_signature,
            "components": component_data,
            "wire_harnesses": harness_data,
            "vehicle_bounds": {
                "min": {"x": vehicle_bounds.min_x, "y": vehicle_bounds.min_y, "z": vehicle_bounds.min_z},
                "max": {"x": vehicle_bounds.max_x, "y": vehicle_bounds.max_y, "z": vehicle_bounds.max_z}
            },
            "metadata": {
                "component_count": len(component_data),
                "harness_count": len(harness_data),
                "generated_at": datetime.utcnow().isoformat()
            }
        }
    
    # ==================== Auto-positioning ====================
    
    async def auto_generate_positions(
        self,
        vehicle_signature: str,
        zone_constraints: Optional[Dict[str, BoundingBox]] = None
    ) -> Dict[str, Any]:
        """Auto-generate positions for components without spatial data"""
        
        # Get components without positions
        query = """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        WHERE c.position IS NULL
        
        RETURN c.id as component_id, c.name as component_name, 
               c.type as component_type, c.zone as zone
        ORDER BY c.zone, c.type
        """
        
        result = await self.neo4j.run(query, vehicle_signature=vehicle_signature)
        
        components_to_position = []
        for record in result.records:
            components_to_position.append({
                "component_id": record["component_id"],
                "component_name": record["component_name"],
                "component_type": record["component_type"],
                "zone": record["zone"]
            })
        
        if not components_to_position:
            return {"message": "All components already have positions"}
        
        # Generate positions based on zones and component types
        position_updates = []
        zone_positions = {}
        
        for comp in components_to_position:
            zone = comp["zone"] or "default"
            comp_type = comp["component_type"]
            
            # Initialize zone position tracking
            if zone not in zone_positions:
                zone_positions[zone] = {"x": 0, "y": 0, "z": 0, "count": 0}
            
            # Get zone constraints or use defaults
            if zone_constraints and zone in zone_constraints:
                bounds = zone_constraints[zone]
            else:
                # Default zone bounds based on zone name
                bounds = self._get_default_zone_bounds(zone)
            
            # Calculate position within zone bounds
            zone_pos = zone_positions[zone]
            spacing = 100  # mm spacing between components
            
            x = bounds.min_x + (zone_pos["count"] % 10) * spacing
            y = bounds.min_y + (zone_pos["count"] // 10) * spacing
            z = bounds.min_z + self._get_component_height_offset(comp_type)
            
            position_updates.append({
                "component_id": comp["component_id"],
                "x": x,
                "y": y,
                "z": z,
                "zone": zone
            })
            
            zone_positions[zone]["count"] += 1
        
        # Apply position updates
        update_result = await self.batch_update_positions(vehicle_signature, position_updates)
        
        return {
            "components_positioned": len(position_updates),
            "update_result": update_result,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def _get_default_zone_bounds(self, zone: str) -> BoundingBox:
        """Get default spatial bounds for a zone"""
        
        zone_bounds = {
            "engine_bay": BoundingBox(-1000, 1000, -500, 500, 0, 800),
            "dashboard": BoundingBox(-800, 800, -300, 100, 800, 1200),
            "trunk": BoundingBox(-600, 600, -800, -400, 0, 600),
            "cabin": BoundingBox(-800, 800, -200, 800, 600, 1400),
            "default": BoundingBox(-500, 500, -500, 500, 0, 500)
        }
        
        return zone_bounds.get(zone, zone_bounds["default"])
    
    def _get_component_height_offset(self, component_type: str) -> float:
        """Get height offset for component type"""
        
        height_offsets = {
            "battery": 0,
            "alternator": 100,
            "ecu": 200,
            "relay": 150,
            "fuse": 100,
            "sensor": 50,
            "connector": 75
        }
        
        return height_offsets.get(component_type, 100)