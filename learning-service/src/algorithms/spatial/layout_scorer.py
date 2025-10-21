"""
Multi-objective optimization scoring for electrical system layouts.
Evaluates layouts across multiple criteria: safety, efficiency, maintainability, cost.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class ScoreCategory(Enum):
    SAFETY = "safety"
    EFFICIENCY = "efficiency"
    MAINTAINABILITY = "maintainability"
    COST = "cost"
    RELIABILITY = "reliability"
    MANUFACTURABILITY = "manufacturability"

@dataclass
class ComponentInfo:
    """Component information for scoring."""
    id: str
    type: str
    position: Tuple[float, float, float]
    dimensions: Tuple[float, float, float]
    electrical_properties: Dict[str, Any]
    mechanical_properties: Dict[str, Any]
    cost: float
    reliability_rating: float

@dataclass
class WireInfo:
    """Wire information for scoring."""
    id: str
    path: List[Tuple[float, float, float]]
    gauge: float
    current_rating: float
    length: float
    cost: float
    bend_count: int
    clearances: List[float]

@dataclass
class LayoutScore:
    """Complete layout scoring result."""
    total_score: float
    category_scores: Dict[ScoreCategory, float]
    detailed_metrics: Dict[str, float]
    violations: List[str]
    recommendations: List[str]
    confidence: float

class BaseScorer(ABC):
    """Abstract base class for layout scorers."""
    
    @abstractmethod
    def calculate_score(self, components: List[ComponentInfo], 
                       wires: List[WireInfo]) -> float:
        """Calculate score for this category."""
        pass
    
    @abstractmethod
    def get_violations(self, components: List[ComponentInfo], 
                      wires: List[WireInfo]) -> List[str]:
        """Get violations for this category."""
        pass

class SafetyScorer(BaseScorer):
    """Evaluates layout safety aspects."""
    
    def __init__(self):
        self.min_high_voltage_clearance = 50.0  # mm
        self.min_heat_clearance = 30.0  # mm
        self.max_wire_current_density = 5.0  # A/mm²
        
    def calculate_score(self, components: List[ComponentInfo], 
                       wires: List[WireInfo]) -> float:
        """Calculate safety score (0-1, higher is better)."""
        scores = []
        
        # 1. Electrical clearance safety
        clearance_score = self._evaluate_clearances(components, wires)
        scores.append(clearance_score * 0.3)
        
        # 2. Current density safety
        current_density_score = self._evaluate_current_densities(wires)
        scores.append(current_density_score * 0.25)
        
        # 3. Heat dissipation safety
        heat_score = self._evaluate_heat_management(components)
        scores.append(heat_score * 0.2)
        
        # 4. Arc fault prevention
        arc_fault_score = self._evaluate_arc_fault_prevention(components, wires)
        scores.append(arc_fault_score * 0.15)
        
        # 5. Accessibility for maintenance
        accessibility_score = self._evaluate_maintenance_accessibility(components)
        scores.append(accessibility_score * 0.1)
        
        return sum(scores)
    
    def get_violations(self, components: List[ComponentInfo], 
                      wires: List[WireInfo]) -> List[str]:
        """Get safety violations."""
        violations = []
        
        # Check clearances
        for i, comp1 in enumerate(components):
            for j, comp2 in enumerate(components[i+1:], i+1):
                distance = np.linalg.norm(
                    np.array(comp1.position) - np.array(comp2.position))
                
                # High voltage component clearance
                if (comp1.electrical_properties.get('voltage', 0) > 60 or 
                    comp2.electrical_properties.get('voltage', 0) > 60):
                    if distance < self.min_high_voltage_clearance:
                        violations.append(
                            f"Insufficient clearance between high-voltage components "
                            f"{comp1.id} and {comp2.id}: {distance:.1f}mm < {self.min_high_voltage_clearance}mm")
        
        # Check wire current densities
        for wire in wires:
            current_density = wire.current_rating / (np.pi * (wire.gauge/2)**2)
            if current_density > self.max_wire_current_density:
                violations.append(
                    f"Wire {wire.id} current density {current_density:.2f} A/mm² "
                    f"exceeds maximum {self.max_wire_current_density} A/mm²")
        
        return violations
    
    def _evaluate_clearances(self, components: List[ComponentInfo], 
                           wires: List[WireInfo]) -> float:
        """Evaluate electrical clearances."""
        clearance_scores = []
        
        for i, comp1 in enumerate(components):
            for j, comp2 in enumerate(components[i+1:], i+1):
                distance = np.linalg.norm(
                    np.array(comp1.position) - np.array(comp2.position))
                
                # Required clearance based on voltage levels
                max_voltage = max(
                    comp1.electrical_properties.get('voltage', 0),
                    comp2.electrical_properties.get('voltage', 0)
                )
                
                required_clearance = self._calculate_required_clearance(max_voltage)
                clearance_ratio = min(distance / required_clearance, 1.0)
                clearance_scores.append(clearance_ratio)
        
        return np.mean(clearance_scores) if clearance_scores else 1.0
    
    def _calculate_required_clearance(self, voltage: float) -> float:
        """Calculate required clearance based on voltage."""
        if voltage <= 12:
            return 2.0  # mm
        elif voltage <= 60:
            return 5.0  # mm
        elif voltage <= 300:
            return 15.0  # mm
        else:
            return 50.0  # mm
    
    def _evaluate_current_densities(self, wires: List[WireInfo]) -> float:
        """Evaluate wire current densities."""
        density_scores = []
        
        for wire in wires:
            area = np.pi * (wire.gauge / 2) ** 2
            current_density = wire.current_rating / area
            
            # Score based on percentage of maximum safe density
            density_ratio = current_density / self.max_wire_current_density
            density_score = max(0, 1.0 - density_ratio)
            density_scores.append(density_score)
        
        return np.mean(density_scores) if density_scores else 1.0
    
    def _evaluate_heat_management(self, components: List[ComponentInfo]) -> float:
        """Evaluate heat management and dissipation."""
        heat_scores = []
        
        for component in components:
            heat_generation = component.electrical_properties.get('power_dissipation', 0)
            
            if heat_generation > 0:
                # Find nearest heat-sensitive components
                min_distance = float('inf')
                for other_comp in components:
                    if other_comp.id != component.id:
                        distance = np.linalg.norm(
                            np.array(component.position) - np.array(other_comp.position))
                        min_distance = min(min_distance, distance)
                
                # Score based on distance to other components
                heat_score = min(min_distance / self.min_heat_clearance, 1.0)
                heat_scores.append(heat_score)
        
        return np.mean(heat_scores) if heat_scores else 1.0
    
    def _evaluate_arc_fault_prevention(self, components: List[ComponentInfo], 
                                     wires: List[WireInfo]) -> float:
        """Evaluate arc fault prevention measures."""
        # Simplified arc fault evaluation
        # In reality, this would consider insulation ratings, environmental factors, etc.
        return 0.8  # Placeholder
    
    def _evaluate_maintenance_accessibility(self, components: List[ComponentInfo]) -> float:
        """Evaluate accessibility for maintenance."""
        accessibility_scores = []
        
        for component in components:
            # Components that need regular maintenance should be accessible
            maintenance_requirement = component.mechanical_properties.get(
                'maintenance_frequency', 0)
            
            if maintenance_requirement > 0:
                # Check if component is near edges for accessibility
                position = np.array(component.position)
                # Simplified: components near workspace edges are more accessible
                edge_distances = [
                    position[0], position[1], position[2],  # Distance from origin
                    1000 - position[0], 1000 - position[1], 1000 - position[2]  # Distance from far edges
                ]
                min_edge_distance = min(edge_distances)
                accessibility_score = min(min_edge_distance / 100, 1.0)
                accessibility_scores.append(accessibility_score)
        
        return np.mean(accessibility_scores) if accessibility_scores else 1.0

class EfficiencyScorer(BaseScorer):
    """Evaluates layout efficiency aspects."""
    
    def calculate_score(self, components: List[ComponentInfo], 
                       wires: List[WireInfo]) -> float:
        """Calculate efficiency score."""
        scores = []
        
        # 1. Wire length efficiency (shorter is better)
        wire_length_score = self._evaluate_wire_lengths(wires)
        scores.append(wire_length_score * 0.4)
        
        # 2. Space utilization efficiency
        space_score = self._evaluate_space_utilization(components)
        scores.append(space_score * 0.3)
        
        # 3. Power loss minimization
        power_loss_score = self._evaluate_power_losses(wires)
        scores.append(power_loss_score * 0.2)
        
        # 4. Assembly efficiency
        assembly_score = self._evaluate_assembly_efficiency(components, wires)
        scores.append(assembly_score * 0.1)
        
        return sum(scores)
    
    def get_violations(self, components: List[ComponentInfo], 
                      wires: List[WireInfo]) -> List[str]:
        """Get efficiency violations."""
        violations = []
        
        # Check for excessively long wires
        total_wire_length = sum(wire.length for wire in wires)
        component_count = len(components)
        
        if component_count > 0:
            avg_wire_length = total_wire_length / len(wires) if wires else 0
            if avg_wire_length > 500:  # mm
                violations.append(
                    f"Average wire length {avg_wire_length:.1f}mm is excessive")
        
        return violations
    
    def _evaluate_wire_lengths(self, wires: List[WireInfo]) -> float:
        """Evaluate wire length efficiency."""
        if not wires:
            return 1.0
        
        total_length = sum(wire.length for wire in wires)
        
        # Estimate ideal minimum length (straight-line distances)
        # This would require start/end points, simplified here
        estimated_minimum = sum(wire.length * 0.7 for wire in wires)  # Assume 70% is ideal
        
        efficiency = min(estimated_minimum / total_length, 1.0)
        return efficiency
    
    def _evaluate_space_utilization(self, components: List[ComponentInfo]) -> float:
        """Evaluate how efficiently space is utilized."""
        if not components:
            return 1.0
        
        # Calculate bounding box of all components
        positions = np.array([comp.position for comp in components])
        min_bounds = np.min(positions, axis=0)
        max_bounds = np.max(positions, axis=0)
        
        used_volume = np.prod(max_bounds - min_bounds)
        
        # Calculate total component volume
        component_volume = sum(np.prod(comp.dimensions) for comp in components)
        
        # Space utilization ratio
        utilization = component_volume / used_volume if used_volume > 0 else 0
        
        # Optimal utilization is around 0.6 (60%)
        optimal_utilization = 0.6
        score = 1.0 - abs(utilization - optimal_utilization) / optimal_utilization
        
        return max(0, score)
    
    def _evaluate_power_losses(self, wires: List[WireInfo]) -> float:
        """Evaluate power losses in wires."""
        total_loss_score = 0.0
        
        for wire in wires:
            # Power loss = I²R, where R = ρL/A
            resistance_per_meter = 0.0175 / (np.pi * (wire.gauge/2)**2)  # Copper resistivity
            resistance = resistance_per_meter * (wire.length / 1000)  # Convert mm to m
            power_loss = (wire.current_rating ** 2) * resistance
            
            # Normalize and invert (lower loss = higher score)
            max_acceptable_loss = 10.0  # watts
            loss_score = max(0, 1.0 - power_loss / max_acceptable_loss)
            total_loss_score += loss_score
        
        return total_loss_score / len(wires) if wires else 1.0
    
    def _evaluate_assembly_efficiency(self, components: List[ComponentInfo], 
                                    wires: List[WireInfo]) -> float:
        """Evaluate how easy the layout is to assemble."""
        # Factor in wire bend count, component accessibility, etc.
        assembly_factors = []
        
        # Wire complexity
        for wire in wires:
            bend_penalty = wire.bend_count * 0.1
            wire_complexity_score = max(0, 1.0 - bend_penalty)
            assembly_factors.append(wire_complexity_score)
        
        return np.mean(assembly_factors) if assembly_factors else 1.0

class CostScorer(BaseScorer):
    """Evaluates layout cost aspects."""
    
    def calculate_score(self, components: List[ComponentInfo], 
                       wires: List[WireInfo]) -> float:
        """Calculate cost efficiency score."""
        # Calculate total cost
        component_cost = sum(comp.cost for comp in components)
        wire_cost = sum(wire.cost for wire in wires)
        total_cost = component_cost + wire_cost
        
        # Estimate manufacturing and assembly costs
        manufacturing_cost = self._estimate_manufacturing_cost(components, wires)
        
        # Total system cost
        system_cost = total_cost + manufacturing_cost
        
        # Score based on cost efficiency (lower cost = higher score)
        # This would typically compare against a baseline or target cost
        baseline_cost = len(components) * 100 + len(wires) * 20  # Simplified baseline
        
        cost_efficiency = min(baseline_cost / system_cost, 1.0) if system_cost > 0 else 1.0
        
        return cost_efficiency
    
    def get_violations(self, components: List[ComponentInfo], 
                      wires: List[WireInfo]) -> List[str]:
        """Get cost-related violations."""
        violations = []
        
        # Check for cost overruns
        total_cost = sum(comp.cost for comp in components) + sum(wire.cost for wire in wires)
        
        # Simplified cost check
        if total_cost > 10000:  # Arbitrary threshold
            violations.append(f"Total cost ${total_cost:.2f} exceeds budget")
        
        return violations
    
    def _estimate_manufacturing_cost(self, components: List[ComponentInfo], 
                                   wires: List[WireInfo]) -> float:
        """Estimate manufacturing and assembly costs."""
        # Assembly time based on complexity
        assembly_time = len(components) * 2 + len(wires) * 1  # minutes
        
        # Wire routing complexity adds time
        routing_complexity = sum(wire.bend_count for wire in wires)
        assembly_time += routing_complexity * 0.5
        
        # Labor cost (simplified)
        labor_rate = 50  # $/hour
        labor_cost = (assembly_time / 60) * labor_rate
        
        return labor_cost

class MultiObjectiveLayoutScorer:
    """Comprehensive layout scorer using multiple objectives."""
    
    def __init__(self, weights: Optional[Dict[ScoreCategory, float]] = None):
        """Initialize with category weights."""
        self.weights = weights or {
            ScoreCategory.SAFETY: 0.35,
            ScoreCategory.EFFICIENCY: 0.25,
            ScoreCategory.COST: 0.20,
            ScoreCategory.RELIABILITY: 0.10,
            ScoreCategory.MAINTAINABILITY: 0.10
        }
        
        # Validate weights sum to 1.0
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.001:
            logger.warning(f"Weights sum to {total_weight}, normalizing to 1.0")
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        # Initialize scorers
        self.scorers = {
            ScoreCategory.SAFETY: SafetyScorer(),
            ScoreCategory.EFFICIENCY: EfficiencyScorer(),
            ScoreCategory.COST: CostScorer(),
            # Add more scorers as needed
        }
    
    def score_layout(self, components: List[ComponentInfo], 
                    wires: List[WireInfo]) -> LayoutScore:
        """Score a complete layout across all categories."""
        logger.info(f"Scoring layout with {len(components)} components and {len(wires)} wires")
        
        category_scores = {}
        all_violations = []
        detailed_metrics = {}
        
        # Score each category
        for category, weight in self.weights.items():
            if category in self.scorers:
                scorer = self.scorers[category]
                score = scorer.calculate_score(components, wires)
                violations = scorer.get_violations(components, wires)
                
                category_scores[category] = score
                all_violations.extend(violations)
                detailed_metrics[f"{category.value}_score"] = score
        
        # Calculate weighted total score
        total_score = sum(
            category_scores.get(category, 0) * weight 
            for category, weight in self.weights.items()
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            category_scores, all_violations, components, wires)
        
        # Calculate confidence based on score distribution
        confidence = self._calculate_confidence(category_scores)
        
        return LayoutScore(
            total_score=total_score,
            category_scores=category_scores,
            detailed_metrics=detailed_metrics,
            violations=all_violations,
            recommendations=recommendations,
            confidence=confidence
        )
    
    def compare_layouts(self, layout_scores: List[LayoutScore]) -> Dict[str, Any]:
        """Compare multiple layout scores and provide analysis."""
        if not layout_scores:
            return {"error": "No layouts to compare"}
        
        comparison = {
            "best_overall": max(layout_scores, key=lambda x: x.total_score),
            "category_leaders": {},
            "violation_analysis": self._analyze_violations(layout_scores),
            "improvement_opportunities": []
        }
        
        # Find best in each category
        for category in ScoreCategory:
            best_layout = max(
                layout_scores, 
                key=lambda x: x.category_scores.get(category, 0)
            )
            comparison["category_leaders"][category.value] = {
                "score": best_layout.category_scores.get(category, 0),
                "layout_index": layout_scores.index(best_layout)
            }
        
        return comparison
    
    def _generate_recommendations(self, category_scores: Dict[ScoreCategory, float],
                                violations: List[str], components: List[ComponentInfo],
                                wires: List[WireInfo]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Identify lowest scoring categories
        sorted_categories = sorted(
            category_scores.items(), key=lambda x: x[1]
        )
        
        for category, score in sorted_categories[:2]:  # Top 2 areas for improvement
            if score < 0.7:
                if category == ScoreCategory.SAFETY:
                    recommendations.append(
                        "Increase clearances between high-voltage components")
                elif category == ScoreCategory.EFFICIENCY:
                    recommendations.append(
                        "Optimize wire routing to reduce total length")
                elif category == ScoreCategory.COST:
                    recommendations.append(
                        "Consider lower-cost component alternatives")
        
        # Add violation-specific recommendations
        if violations:
            recommendations.append(
                f"Address {len(violations)} safety/code violations")
        
        return recommendations
    
    def _calculate_confidence(self, category_scores: Dict[ScoreCategory, float]) -> float:
        """Calculate confidence in the scoring based on score distribution."""
        scores = list(category_scores.values())
        if not scores:
            return 0.0
        
        # High confidence when scores are consistent (low variance)
        mean_score = np.mean(scores)
        variance = np.var(scores)
        
        # Normalize confidence (lower variance = higher confidence)
        max_variance = 0.25  # Maximum expected variance
        confidence = max(0, 1.0 - variance / max_variance)
        
        return confidence
    
    def _analyze_violations(self, layout_scores: List[LayoutScore]) -> Dict[str, Any]:
        """Analyze violations across multiple layouts."""
        all_violations = []
        for layout in layout_scores:
            all_violations.extend(layout.violations)
        
        # Count violation types
        violation_counts = {}
        for violation in all_violations:
            violation_type = violation.split(':')[0] if ':' in violation else violation
            violation_counts[violation_type] = violation_counts.get(violation_type, 0) + 1
        
        return {
            "total_violations": len(all_violations),
            "violation_types": violation_counts,
            "most_common": max(violation_counts.items(), key=lambda x: x[1]) if violation_counts else None
        }