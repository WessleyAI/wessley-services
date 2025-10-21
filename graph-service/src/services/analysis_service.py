"""
Electrical system analysis service for advanced diagnostics and optimization
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
import asyncio

from ..utils.neo4j_utils import Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class CircuitLoadAnalysis:
    """Circuit load analysis result"""
    circuit_id: str
    circuit_name: str
    max_capacity: float
    current_load: float
    load_percentage: float
    safety_margin: float
    components: List[Dict[str, Any]]
    recommendations: List[str]


@dataclass
class FaultAnalysisResult:
    """Fault analysis result"""
    vehicle_signature: str
    potential_faults: List[Dict[str, Any]]
    single_points_of_failure: List[str]
    redundancy_gaps: List[str]
    reliability_score: float
    critical_components: List[str]


@dataclass
class PowerEfficiencyAnalysis:
    """Power efficiency analysis result"""
    vehicle_signature: str
    overall_efficiency: float
    power_losses: Dict[str, float]
    inefficient_paths: List[Dict[str, Any]]
    optimization_opportunities: List[str]
    estimated_savings: Dict[str, float]


@dataclass
class ThermalAnalysisResult:
    """Thermal analysis result"""
    vehicle_signature: str
    hot_spots: List[Dict[str, Any]]
    thermal_zones: Dict[str, Dict[str, Any]]
    cooling_requirements: List[str]
    thermal_stress_components: List[str]


class AnalysisService:
    """
    Advanced electrical system analysis service
    """
    
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client
    
    # ==================== Circuit Load Analysis ====================
    
    async def analyze_circuit_loads(
        self, 
        vehicle_signature: str,
        circuit_id: Optional[str] = None
    ) -> List[CircuitLoadAnalysis]:
        """Analyze circuit loads and capacity utilization"""
        
        where_clause = "c.vehicle_signature = $vehicle_signature"
        params = {"vehicle_signature": vehicle_signature}
        
        if circuit_id:
            where_clause += " AND c.id = $circuit_id"
            params["circuit_id"] = circuit_id
        
        query = f"""
        MATCH (c:Circuit)
        WHERE {where_clause}
        
        // Get components in each circuit
        OPTIONAL MATCH (comp:Component)-[:PART_OF]->(c)
        WHERE comp.vehicle_signature = $vehicle_signature
        
        // Calculate total load for the circuit
        WITH c, collect(comp) as components,
             reduce(total_load = 0.0, component in collect(comp) | 
                 total_load + coalesce(component.current_rating, 0.0)) as total_current_load
        
        // Calculate load percentage and safety margin
        WITH c, components, total_current_load,
             CASE 
                 WHEN c.max_current > 0 THEN (total_current_load / c.max_current) * 100
                 ELSE 0
             END as load_percentage,
             CASE
                 WHEN c.max_current > 0 THEN c.max_current - total_current_load
                 ELSE 0
             END as safety_margin
        
        RETURN 
            c.id as circuit_id,
            c.name as circuit_name,
            c.max_current as max_capacity,
            total_current_load,
            load_percentage,
            safety_margin,
            [comp in components | {{
                id: comp.id,
                name: comp.name,
                type: comp.type,
                current_rating: comp.current_rating,
                voltage_rating: comp.voltage_rating
            }}] as component_details
        ORDER BY load_percentage DESC
        """
        
        result = await self.neo4j.run(query, **params)
        
        analyses = []
        for record in result.records:
            # Generate recommendations based on load analysis
            recommendations = self._generate_load_recommendations(
                record["load_percentage"],
                record["safety_margin"],
                record["component_details"]
            )
            
            analyses.append(CircuitLoadAnalysis(
                circuit_id=record["circuit_id"],
                circuit_name=record["circuit_name"],
                max_capacity=record["max_capacity"],
                current_load=record["total_current_load"],
                load_percentage=record["load_percentage"],
                safety_margin=record["safety_margin"],
                components=record["component_details"],
                recommendations=recommendations
            ))
        
        return analyses
    
    def _generate_load_recommendations(
        self, 
        load_percentage: float, 
        safety_margin: float,
        components: List[Dict]
    ) -> List[str]:
        """Generate load-based recommendations"""
        
        recommendations = []
        
        if load_percentage > 100:
            recommendations.append("CRITICAL: Circuit is overloaded - immediate action required")
            recommendations.append("Consider splitting load across multiple circuits")
        elif load_percentage > 90:
            recommendations.append("WARNING: Circuit near capacity - monitor closely")
            recommendations.append("Plan for load redistribution")
        elif load_percentage > 80:
            recommendations.append("Circuit heavily loaded - consider capacity planning")
        
        if safety_margin < 5:
            recommendations.append("Insufficient safety margin - upgrade circuit capacity")
        
        # Check for high-draw components
        high_draw_components = [c for c in components if c.get('current_rating', 0) > 20]
        if high_draw_components:
            recommendations.append(f"Monitor high-draw components: {', '.join([c['name'] for c in high_draw_components])}")
        
        return recommendations
    
    # ==================== Fault Analysis ====================
    
    async def analyze_fault_tolerance(
        self, 
        vehicle_signature: str
    ) -> FaultAnalysisResult:
        """Analyze system fault tolerance and identify vulnerabilities"""
        
        # Find single points of failure
        spof_query = """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        
        // Count incoming and outgoing critical relationships
        OPTIONAL MATCH (c)-[:POWERED_BY]->(power_source:Component)
        OPTIONAL MATCH (dependent:Component)-[:POWERED_BY]->(c)
        
        WITH c, count(power_source) as power_sources, count(dependent) as dependents
        
        // Identify components with no redundancy
        WHERE power_sources <= 1 AND dependents > 0
        
        RETURN c.id as component_id, c.name as component_name, c.type as component_type,
               power_sources, dependents
        ORDER BY dependents DESC
        """
        
        spof_result = await self.neo4j.run(spof_query, vehicle_signature=vehicle_signature)
        single_points_of_failure = [record["component_id"] for record in spof_result.records]
        
        # Find critical components (safety-critical or high-dependency)
        critical_query = """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        
        // Count dependencies
        OPTIONAL MATCH (dependent:Component)-[:POWERED_BY|CONTROLS*1..3]->(c)
        WITH c, count(DISTINCT dependent) as dependency_count
        
        // Identify critical components
        WHERE dependency_count > 5 OR 
              c.type IN ['ecu', 'battery', 'alternator'] OR
              exists(c.safety_critical) AND c.safety_critical = true
        
        RETURN c.id as component_id, c.name as component_name, 
               c.type as component_type, dependency_count
        ORDER BY dependency_count DESC
        """
        
        critical_result = await self.neo4j.run(critical_query, vehicle_signature=vehicle_signature)
        critical_components = [record["component_id"] for record in critical_result.records]
        
        # Analyze redundancy gaps
        redundancy_query = """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        WHERE c.type IN ['ecu', 'sensor', 'actuator']
        
        // Check for functional redundancy
        MATCH (similar:Component {vehicle_signature: $vehicle_signature})
        WHERE similar.type = c.type AND similar.id <> c.id
        
        WITH c, count(similar) as redundant_count
        WHERE redundant_count = 0
        
        RETURN c.id as component_id, c.type as component_type
        """
        
        redundancy_result = await self.neo4j.run(redundancy_query, vehicle_signature=vehicle_signature)
        redundancy_gaps = [f"{record['component_type']}: {record['component_id']}" for record in redundancy_result.records]
        
        # Identify potential fault scenarios
        potential_faults = await self._identify_potential_faults(vehicle_signature)
        
        # Calculate overall reliability score
        reliability_score = await self._calculate_reliability_score(
            vehicle_signature, 
            len(single_points_of_failure),
            len(critical_components),
            len(redundancy_gaps)
        )
        
        return FaultAnalysisResult(
            vehicle_signature=vehicle_signature,
            potential_faults=potential_faults,
            single_points_of_failure=single_points_of_failure,
            redundancy_gaps=redundancy_gaps,
            reliability_score=reliability_score,
            critical_components=critical_components
        )
    
    async def _identify_potential_faults(self, vehicle_signature: str) -> List[Dict[str, Any]]:
        """Identify potential fault scenarios"""
        
        faults = []
        
        # Check for overloaded circuits
        overload_query = """
        MATCH (c:Circuit {vehicle_signature: $vehicle_signature})
        OPTIONAL MATCH (comp:Component)-[:PART_OF]->(c)
        
        WITH c, sum(coalesce(comp.current_rating, 0)) as total_load
        WHERE total_load > c.max_current
        
        RETURN c.id as circuit_id, c.name as circuit_name, 
               total_load, c.max_current as capacity
        """
        
        overload_result = await self.neo4j.run(overload_query, vehicle_signature=vehicle_signature)
        
        for record in overload_result.records:
            faults.append({
                "type": "circuit_overload",
                "severity": "high",
                "component_id": record["circuit_id"],
                "description": f"Circuit {record['circuit_name']} overloaded",
                "details": {
                    "current_load": record["total_load"],
                    "capacity": record["capacity"],
                    "overload_percentage": ((record["total_load"] / record["capacity"]) - 1) * 100
                }
            })
        
        # Check for missing protection devices
        protection_query = """
        MATCH (c:Circuit {vehicle_signature: $vehicle_signature})
        WHERE NOT exists(c.protection) OR c.protection = 'none'
        
        OPTIONAL MATCH (comp:Component)-[:PART_OF]->(c)
        WHERE comp.type NOT IN ['fuse', 'circuit_breaker']
        
        RETURN c.id as circuit_id, c.name as circuit_name, count(comp) as unprotected_components
        """
        
        protection_result = await self.neo4j.run(protection_query, vehicle_signature=vehicle_signature)
        
        for record in protection_result.records:
            if record["unprotected_components"] > 0:
                faults.append({
                    "type": "missing_protection",
                    "severity": "medium",
                    "component_id": record["circuit_id"],
                    "description": f"Circuit {record['circuit_name']} lacks protection",
                    "details": {
                        "unprotected_components": record["unprotected_components"]
                    }
                })
        
        return faults
    
    async def _calculate_reliability_score(
        self, 
        vehicle_signature: str,
        spof_count: int,
        critical_count: int,
        redundancy_gaps: int
    ) -> float:
        """Calculate overall system reliability score"""
        
        # Get total component count
        total_query = """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        RETURN count(c) as total_components
        """
        
        total_result = await self.neo4j.run(total_query, vehicle_signature=vehicle_signature)
        total_components = total_result.records[0]["total_components"] if total_result.records else 1
        
        # Base score
        base_score = 100.0
        
        # Penalties
        spof_penalty = (spof_count / total_components) * 30
        critical_penalty = (critical_count / total_components) * 15
        redundancy_penalty = (redundancy_gaps / total_components) * 20
        
        reliability_score = max(0, base_score - spof_penalty - critical_penalty - redundancy_penalty)
        
        return round(reliability_score, 2)
    
    # ==================== Power Efficiency Analysis ====================
    
    async def analyze_power_efficiency(
        self, 
        vehicle_signature: str
    ) -> PowerEfficiencyAnalysis:
        """Analyze power system efficiency and identify optimization opportunities"""
        
        # Analyze power distribution paths
        efficiency_query = """
        MATCH (source:Component {vehicle_signature: $vehicle_signature})
        WHERE source.type IN ['battery', 'alternator']
        
        // Find all loads powered by this source
        OPTIONAL MATCH path = (source)-[:POWERED_BY|CONNECTS_TO*1..5]->(load:Component)
        WHERE load.vehicle_signature = $vehicle_signature 
          AND load.type NOT IN ['battery', 'alternator']
        
        WITH source, collect(DISTINCT load) as loads, collect(DISTINCT path) as paths
        
        // Calculate efficiency for each path
        UNWIND paths as path
        WITH source, loads, path,
             reduce(resistance = 0.0, rel in relationships(path) |
                 resistance + coalesce(rel.resistance, 0.1)) as total_resistance,
             length(path) as path_length
        
        RETURN 
            source.id as source_id,
            source.type as source_type,
            avg(total_resistance) as avg_resistance,
            avg(path_length) as avg_path_length,
            count(DISTINCT path) as path_count
        """
        
        efficiency_result = await self.neo4j.run(efficiency_query, vehicle_signature=vehicle_signature)
        
        # Calculate overall efficiency
        total_resistance = 0.0
        total_paths = 0
        
        for record in efficiency_result.records:
            total_resistance += record["avg_resistance"] or 0
            total_paths += record["path_count"] or 0
        
        # Simplified efficiency calculation
        overall_efficiency = max(0.5, 1.0 - (total_resistance / 100.0)) if total_resistance > 0 else 0.9
        
        # Identify inefficient paths
        inefficient_paths = await self._find_inefficient_paths(vehicle_signature)
        
        # Calculate power losses
        power_losses = await self._calculate_power_losses(vehicle_signature)
        
        # Generate optimization opportunities
        optimization_opportunities = self._generate_efficiency_recommendations(
            overall_efficiency,
            inefficient_paths,
            power_losses
        )
        
        # Estimate potential savings
        estimated_savings = {
            "power_reduction_watts": len(inefficient_paths) * 5.0,
            "efficiency_improvement_percent": min(15.0, len(optimization_opportunities) * 2.0)
        }
        
        return PowerEfficiencyAnalysis(
            vehicle_signature=vehicle_signature,
            overall_efficiency=overall_efficiency,
            power_losses=power_losses,
            inefficient_paths=inefficient_paths,
            optimization_opportunities=optimization_opportunities,
            estimated_savings=estimated_savings
        )
    
    async def _find_inefficient_paths(self, vehicle_signature: str) -> List[Dict[str, Any]]:
        """Find inefficient power distribution paths"""
        
        query = """
        MATCH (source:Component {vehicle_signature: $vehicle_signature})
        WHERE source.type IN ['battery', 'alternator']
        
        MATCH path = (source)-[:CONNECTS_TO*3..]->(load:Component)
        WHERE load.vehicle_signature = $vehicle_signature
          AND length(path) > 4  // Paths longer than 4 hops
        
        RETURN 
            [node in nodes(path) | node.id] as path_nodes,
            length(path) as path_length,
            source.id as source_id,
            load.id as load_id
        ORDER BY path_length DESC
        LIMIT 10
        """
        
        result = await self.neo4j.run(query, vehicle_signature=vehicle_signature)
        
        inefficient_paths = []
        for record in result.records:
            inefficient_paths.append({
                "source_id": record["source_id"],
                "load_id": record["load_id"],
                "path_length": record["path_length"],
                "path_nodes": record["path_nodes"],
                "inefficiency_reason": "excessive_path_length"
            })
        
        return inefficient_paths
    
    async def _calculate_power_losses(self, vehicle_signature: str) -> Dict[str, float]:
        """Calculate estimated power losses in the system"""
        
        # Simplified power loss calculation
        query = """
        MATCH ()-[r:CONNECTS_TO]-()
        WHERE r.vehicle_signature = $vehicle_signature OR
              (startNode(r).vehicle_signature = $vehicle_signature AND 
               endNode(r).vehicle_signature = $vehicle_signature)
        
        RETURN count(r) as connection_count,
               avg(coalesce(r.resistance, 0.1)) as avg_resistance
        """
        
        result = await self.neo4j.run(query, vehicle_signature=vehicle_signature)
        
        if result.records:
            record = result.records[0]
            connection_count = record["connection_count"]
            avg_resistance = record["avg_resistance"]
            
            # Simplified loss calculations
            resistive_losses = connection_count * avg_resistance * 0.1  # Simplified
            conversion_losses = connection_count * 0.05  # 5% per conversion
            
            return {
                "resistive_losses_watts": resistive_losses,
                "conversion_losses_watts": conversion_losses,
                "total_losses_watts": resistive_losses + conversion_losses
            }
        
        return {"resistive_losses_watts": 0.0, "conversion_losses_watts": 0.0, "total_losses_watts": 0.0}
    
    def _generate_efficiency_recommendations(
        self,
        overall_efficiency: float,
        inefficient_paths: List[Dict],
        power_losses: Dict[str, float]
    ) -> List[str]:
        """Generate power efficiency recommendations"""
        
        recommendations = []
        
        if overall_efficiency < 0.8:
            recommendations.append("System efficiency below 80% - comprehensive optimization needed")
        
        if len(inefficient_paths) > 5:
            recommendations.append("Multiple inefficient power paths detected - consider routing optimization")
        
        if power_losses.get("total_losses_watts", 0) > 50:
            recommendations.append("High power losses detected - upgrade wire gauges and connections")
        
        if len(inefficient_paths) > 0:
            recommendations.append("Implement direct power routing for high-current loads")
        
        recommendations.append("Regular maintenance of connections to minimize resistance")
        
        return recommendations
    
    # ==================== Thermal Analysis ====================
    
    async def analyze_thermal_characteristics(
        self, 
        vehicle_signature: str
    ) -> ThermalAnalysisResult:
        """Analyze thermal characteristics and identify hot spots"""
        
        # Find high-power components (potential hot spots)
        hotspot_query = """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        WHERE c.voltage_rating IS NOT NULL AND c.current_rating IS NOT NULL
        
        WITH c, (c.voltage_rating * c.current_rating) as power_consumption
        WHERE power_consumption > 50  // Components over 50W
        
        OPTIONAL MATCH (c)-[:LOCATED_IN]->(z:Zone)
        
        RETURN c.id as component_id, c.name as component_name, c.type as component_type,
               power_consumption, c.zone as zone_name,
               z.environmental_conditions as zone_conditions
        ORDER BY power_consumption DESC
        """
        
        hotspot_result = await self.neo4j.run(hotspot_query, vehicle_signature=vehicle_signature)
        
        hot_spots = []
        for record in hotspot_result.records:
            hot_spots.append({
                "component_id": record["component_id"],
                "component_name": record["component_name"],
                "component_type": record["component_type"],
                "power_consumption": record["power_consumption"],
                "zone": record["zone_name"],
                "thermal_risk": "high" if record["power_consumption"] > 100 else "medium"
            })
        
        # Analyze thermal zones
        thermal_zones = await self._analyze_thermal_zones(vehicle_signature)
        
        # Generate cooling requirements
        cooling_requirements = self._generate_cooling_requirements(hot_spots, thermal_zones)
        
        # Identify thermally stressed components
        thermal_stress_components = [hs["component_id"] for hs in hot_spots if hs["thermal_risk"] == "high"]
        
        return ThermalAnalysisResult(
            vehicle_signature=vehicle_signature,
            hot_spots=hot_spots,
            thermal_zones=thermal_zones,
            cooling_requirements=cooling_requirements,
            thermal_stress_components=thermal_stress_components
        )
    
    async def _analyze_thermal_zones(self, vehicle_signature: str) -> Dict[str, Dict[str, Any]]:
        """Analyze thermal characteristics by zone"""
        
        query = """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})-[:LOCATED_IN]->(z:Zone)
        WHERE c.voltage_rating IS NOT NULL AND c.current_rating IS NOT NULL
        
        WITH z, collect(c) as components,
             sum(c.voltage_rating * c.current_rating) as total_zone_power
        
        RETURN z.id as zone_id, z.name as zone_name,
               z.environmental_conditions as conditions,
               total_zone_power,
               size(components) as component_count
        ORDER BY total_zone_power DESC
        """
        
        result = await self.neo4j.run(query, vehicle_signature=vehicle_signature)
        
        thermal_zones = {}
        for record in result.records:
            zone_id = record["zone_id"]
            thermal_zones[zone_id] = {
                "name": record["zone_name"],
                "total_power": record["total_zone_power"],
                "component_count": record["component_count"],
                "conditions": record["conditions"],
                "thermal_load": "high" if record["total_zone_power"] > 200 else "moderate" if record["total_zone_power"] > 100 else "low"
            }
        
        return thermal_zones
    
    def _generate_cooling_requirements(
        self, 
        hot_spots: List[Dict], 
        thermal_zones: Dict[str, Dict]
    ) -> List[str]:
        """Generate cooling requirements based on thermal analysis"""
        
        requirements = []
        
        high_power_components = [hs for hs in hot_spots if hs["power_consumption"] > 100]
        if high_power_components:
            requirements.append(f"Active cooling required for {len(high_power_components)} high-power components")
        
        high_thermal_zones = [zone for zone, data in thermal_zones.items() if data["thermal_load"] == "high"]
        if high_thermal_zones:
            requirements.append(f"Zone-level thermal management needed for: {', '.join(high_thermal_zones)}")
        
        requirements.append("Ensure adequate ventilation in equipment compartments")
        requirements.append("Monitor component temperatures during operation")
        
        return requirements
    
    # ==================== Comprehensive System Analysis ====================
    
    async def perform_comprehensive_analysis(
        self, 
        vehicle_signature: str
    ) -> Dict[str, Any]:
        """Perform comprehensive electrical system analysis"""
        
        logger.info(f"Starting comprehensive analysis for vehicle: {vehicle_signature}")
        
        # Run all analyses concurrently
        circuit_analysis_task = self.analyze_circuit_loads(vehicle_signature)
        fault_analysis_task = self.analyze_fault_tolerance(vehicle_signature)
        efficiency_analysis_task = self.analyze_power_efficiency(vehicle_signature)
        thermal_analysis_task = self.analyze_thermal_characteristics(vehicle_signature)
        
        # Wait for all analyses to complete
        circuit_analysis, fault_analysis, efficiency_analysis, thermal_analysis = await asyncio.gather(
            circuit_analysis_task,
            fault_analysis_task,
            efficiency_analysis_task,
            thermal_analysis_task
        )
        
        # Calculate overall system health score
        health_score = self._calculate_overall_health_score(
            circuit_analysis,
            fault_analysis,
            efficiency_analysis,
            thermal_analysis
        )
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            health_score,
            circuit_analysis,
            fault_analysis,
            efficiency_analysis,
            thermal_analysis
        )
        
        return {
            "vehicle_signature": vehicle_signature,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "overall_health_score": health_score,
            "executive_summary": executive_summary,
            "detailed_analysis": {
                "circuit_loads": [analysis.__dict__ for analysis in circuit_analysis],
                "fault_tolerance": fault_analysis.__dict__,
                "power_efficiency": efficiency_analysis.__dict__,
                "thermal_characteristics": thermal_analysis.__dict__
            }
        }
    
    def _calculate_overall_health_score(
        self,
        circuit_analysis: List[CircuitLoadAnalysis],
        fault_analysis: FaultAnalysisResult,
        efficiency_analysis: PowerEfficiencyAnalysis,
        thermal_analysis: ThermalAnalysisResult
    ) -> float:
        """Calculate overall system health score"""
        
        # Circuit health (25% weight)
        overloaded_circuits = sum(1 for ca in circuit_analysis if ca.load_percentage > 100)
        circuit_score = max(0, 100 - (overloaded_circuits * 20))
        
        # Fault tolerance (30% weight)
        fault_score = fault_analysis.reliability_score
        
        # Efficiency (25% weight)
        efficiency_score = efficiency_analysis.overall_efficiency * 100
        
        # Thermal (20% weight)
        high_risk_thermal = sum(1 for hs in thermal_analysis.hot_spots if hs["thermal_risk"] == "high")
        thermal_score = max(0, 100 - (high_risk_thermal * 15))
        
        overall_score = (
            circuit_score * 0.25 +
            fault_score * 0.30 +
            efficiency_score * 0.25 +
            thermal_score * 0.20
        )
        
        return round(overall_score, 1)
    
    def _generate_executive_summary(
        self,
        health_score: float,
        circuit_analysis: List[CircuitLoadAnalysis],
        fault_analysis: FaultAnalysisResult,
        efficiency_analysis: PowerEfficiencyAnalysis,
        thermal_analysis: ThermalAnalysisResult
    ) -> Dict[str, Any]:
        """Generate executive summary of analysis results"""
        
        # Determine overall status
        if health_score >= 85:
            status = "excellent"
        elif health_score >= 70:
            status = "good"
        elif health_score >= 50:
            status = "fair"
        else:
            status = "poor"
        
        # Key findings
        key_findings = []
        
        overloaded_circuits = sum(1 for ca in circuit_analysis if ca.load_percentage > 100)
        if overloaded_circuits > 0:
            key_findings.append(f"{overloaded_circuits} circuit(s) are overloaded")
        
        if len(fault_analysis.single_points_of_failure) > 0:
            key_findings.append(f"{len(fault_analysis.single_points_of_failure)} single points of failure identified")
        
        if efficiency_analysis.overall_efficiency < 0.8:
            key_findings.append("Power efficiency below optimal level")
        
        if len(thermal_analysis.hot_spots) > 0:
            key_findings.append(f"{len(thermal_analysis.hot_spots)} thermal hot spots detected")
        
        # Priority actions
        priority_actions = []
        
        if overloaded_circuits > 0:
            priority_actions.append("Address circuit overloading immediately")
        
        if len(fault_analysis.single_points_of_failure) > 5:
            priority_actions.append("Implement redundancy for critical components")
        
        if len(thermal_analysis.thermal_stress_components) > 0:
            priority_actions.append("Implement thermal management for high-power components")
        
        return {
            "overall_status": status,
            "health_score": health_score,
            "key_findings": key_findings,
            "priority_actions": priority_actions,
            "metrics": {
                "total_circuits_analyzed": len(circuit_analysis),
                "overloaded_circuits": overloaded_circuits,
                "single_points_of_failure": len(fault_analysis.single_points_of_failure),
                "efficiency_percentage": round(efficiency_analysis.overall_efficiency * 100, 1),
                "thermal_hot_spots": len(thermal_analysis.hot_spots)
            }
        }