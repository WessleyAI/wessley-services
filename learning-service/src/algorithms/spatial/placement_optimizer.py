"""
Genetic algorithms for optimal component placement in 3D electrical systems.
Optimizes for minimal interference, accessibility, and wire length.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

@dataclass
class Component:
    """Electrical component with physical and electrical properties."""
    id: str
    type: str
    dimensions: Tuple[float, float, float]  # width, height, depth in mm
    weight: float  # kg
    heat_generation: float  # watts
    accessibility_requirement: float  # 0-1, higher means needs more access
    vibration_sensitivity: float  # 0-1
    electromagnetic_interference: float  # 0-1
    connections: List[str]  # IDs of connected components

@dataclass
class PlacementSolution:
    """A placement solution with position and orientation for each component."""
    positions: Dict[str, Tuple[float, float, float]]  # component_id -> (x, y, z)
    orientations: Dict[str, Tuple[float, float, float]]  # component_id -> (rx, ry, rz)
    fitness: float
    violations: List[str]

class GeneticPlacementOptimizer:
    """Genetic algorithm for optimal 3D component placement."""
    
    def __init__(self, 
                 workspace_bounds: Tuple[float, float, float],
                 population_size: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 max_generations: int = 500):
        self.workspace_bounds = workspace_bounds
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.components: List[Component] = []
        
    def add_component(self, component: Component):
        """Add a component to be placed."""
        self.components.append(component)
        
    def optimize(self) -> PlacementSolution:
        """Run genetic optimization to find optimal placement."""
        logger.info(f"Starting placement optimization for {len(self.components)} components")
        
        # Initialize population
        population = self._initialize_population()
        
        best_solution = None
        best_fitness = float('-inf')
        
        for generation in range(self.max_generations):
            # Evaluate fitness for all solutions
            fitness_scores = self._evaluate_population(population)
            
            # Track best solution
            max_idx = np.argmax(fitness_scores)
            if fitness_scores[max_idx] > best_fitness:
                best_fitness = fitness_scores[max_idx]
                best_solution = population[max_idx]
                
            logger.debug(f"Generation {generation}: Best fitness = {best_fitness:.3f}")
            
            # Selection, crossover, mutation
            population = self._evolve_population(population, fitness_scores)
            
            # Early stopping if solution is good enough
            if best_fitness > 0.95:
                logger.info(f"Early stopping at generation {generation}")
                break
                
        return self._create_solution(best_solution, best_fitness)
    
    def _initialize_population(self) -> List[Dict]:
        """Create initial random population."""
        population = []
        
        for _ in range(self.population_size):
            individual = {}
            for component in self.components:
                # Random position within workspace
                x = random.uniform(0, self.workspace_bounds[0])
                y = random.uniform(0, self.workspace_bounds[1])
                z = random.uniform(0, self.workspace_bounds[2])
                
                # Random orientation
                rx = random.uniform(0, 2 * np.pi)
                ry = random.uniform(0, 2 * np.pi)
                rz = random.uniform(0, 2 * np.pi)
                
                individual[component.id] = {
                    'position': (x, y, z),
                    'orientation': (rx, ry, rz)
                }
            population.append(individual)
            
        return population
    
    def _evaluate_population(self, population: List[Dict]) -> List[float]:
        """Evaluate fitness for entire population using parallel processing."""
        with ThreadPoolExecutor(max_workers=8) as executor:
            fitness_scores = list(executor.map(self._evaluate_individual, population))
        return fitness_scores
    
    def _evaluate_individual(self, individual: Dict) -> float:
        """Evaluate fitness of a single placement solution."""
        fitness = 0.0
        penalties = 0.0
        
        # 1. Collision detection penalty
        collisions = self._detect_collisions(individual)
        penalties += len(collisions) * 100
        
        # 2. Wire length optimization (minimize total wire length)
        wire_length_score = self._calculate_wire_length_score(individual)
        fitness += wire_length_score * 0.3
        
        # 3. Accessibility score
        accessibility_score = self._calculate_accessibility_score(individual)
        fitness += accessibility_score * 0.2
        
        # 4. Heat management score
        heat_score = self._calculate_heat_management_score(individual)
        fitness += heat_score * 0.2
        
        # 5. Vibration isolation score
        vibration_score = self._calculate_vibration_score(individual)
        fitness += vibration_score * 0.1
        
        # 6. EMI minimization score
        emi_score = self._calculate_emi_score(individual)
        fitness += emi_score * 0.2
        
        return max(0, fitness - penalties)
    
    def _detect_collisions(self, individual: Dict) -> List[Tuple[str, str]]:
        """Detect collisions between components."""
        collisions = []
        components_list = list(self.components)
        
        for i in range(len(components_list)):
            for j in range(i + 1, len(components_list)):
                comp1, comp2 = components_list[i], components_list[j]
                pos1 = individual[comp1.id]['position']
                pos2 = individual[comp2.id]['position']
                
                # Simple bounding box collision detection
                if self._boxes_intersect(pos1, comp1.dimensions, pos2, comp2.dimensions):
                    collisions.append((comp1.id, comp2.id))
                    
        return collisions
    
    def _boxes_intersect(self, pos1: Tuple[float, float, float], dim1: Tuple[float, float, float],
                        pos2: Tuple[float, float, float], dim2: Tuple[float, float, float]) -> bool:
        """Check if two 3D bounding boxes intersect."""
        for i in range(3):
            if (pos1[i] + dim1[i]/2 < pos2[i] - dim2[i]/2 or 
                pos1[i] - dim1[i]/2 > pos2[i] + dim2[i]/2):
                return False
        return True
    
    def _calculate_wire_length_score(self, individual: Dict) -> float:
        """Calculate score based on total wire length (lower is better)."""
        total_length = 0.0
        connections_made = set()
        
        for component in self.components:
            pos1 = individual[component.id]['position']
            
            for connected_id in component.connections:
                connection_key = tuple(sorted([component.id, connected_id]))
                if connection_key not in connections_made:
                    connections_made.add(connection_key)
                    
                    # Find connected component
                    connected_comp = next((c for c in self.components if c.id == connected_id), None)
                    if connected_comp:
                        pos2 = individual[connected_id]['position']
                        distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
                        total_length += distance
        
        # Normalize and invert (shorter wires = higher score)
        max_possible_length = len(self.components) * np.linalg.norm(self.workspace_bounds)
        return 1.0 - min(total_length / max_possible_length, 1.0)
    
    def _calculate_accessibility_score(self, individual: Dict) -> float:
        """Calculate accessibility score based on clearance around components."""
        total_score = 0.0
        
        for component in self.components:
            if component.accessibility_requirement > 0:
                pos = individual[component.id]['position']
                clearance = self._calculate_clearance(pos, component, individual)
                required_clearance = component.accessibility_requirement * 100  # mm
                
                score = min(clearance / required_clearance, 1.0)
                total_score += score * component.accessibility_requirement
                
        return total_score / len(self.components) if self.components else 0.0
    
    def _calculate_clearance(self, pos: Tuple[float, float, float], 
                           component: Component, individual: Dict) -> float:
        """Calculate minimum clearance around a component."""
        min_clearance = float('inf')
        
        for other_comp in self.components:
            if other_comp.id != component.id:
                other_pos = individual[other_comp.id]['position']
                distance = np.linalg.norm(np.array(pos) - np.array(other_pos))
                clearance = distance - (np.linalg.norm(component.dimensions) + 
                                      np.linalg.norm(other_comp.dimensions)) / 2
                min_clearance = min(min_clearance, clearance)
                
        return max(0, min_clearance)
    
    def _calculate_heat_management_score(self, individual: Dict) -> float:
        """Calculate heat management score."""
        total_score = 0.0
        
        for component in self.components:
            if component.heat_generation > 0:
                pos = individual[component.id]['position']
                
                # Heat dissipation is better at edges and with airflow
                edge_score = self._calculate_edge_proximity_score(pos)
                airflow_score = self._calculate_airflow_score(pos, individual)
                
                heat_score = (edge_score + airflow_score) / 2
                total_score += heat_score * (component.heat_generation / 100)  # Normalize
                
        return total_score / len(self.components) if self.components else 0.0
    
    def _calculate_edge_proximity_score(self, pos: Tuple[float, float, float]) -> float:
        """Calculate how close component is to workspace edges for heat dissipation."""
        distances_to_edges = [
            pos[0], self.workspace_bounds[0] - pos[0],
            pos[1], self.workspace_bounds[1] - pos[1],
            pos[2], self.workspace_bounds[2] - pos[2]
        ]
        min_distance = min(distances_to_edges)
        max_distance = max(self.workspace_bounds) / 2
        
        return min(min_distance / max_distance, 1.0)
    
    def _calculate_airflow_score(self, pos: Tuple[float, float, float], individual: Dict) -> float:
        """Calculate airflow availability score."""
        # Simplified: components higher up have better airflow
        return pos[2] / self.workspace_bounds[2]
    
    def _calculate_vibration_score(self, individual: Dict) -> float:
        """Calculate vibration isolation score."""
        total_score = 0.0
        
        for component in self.components:
            if component.vibration_sensitivity > 0:
                pos = individual[component.id]['position']
                
                # Lower positions and away from moving parts have less vibration
                height_score = 1.0 - (pos[2] / self.workspace_bounds[2])
                isolation_score = self._calculate_isolation_from_vibration_sources(pos, individual)
                
                vibration_score = (height_score + isolation_score) / 2
                total_score += vibration_score * component.vibration_sensitivity
                
        return total_score / len(self.components) if self.components else 0.0
    
    def _calculate_isolation_from_vibration_sources(self, pos: Tuple[float, float, float], 
                                                  individual: Dict) -> float:
        """Calculate isolation from vibration sources like motors."""
        # Simplified: assume vibration sources are components with high weight
        min_distance = float('inf')
        
        for component in self.components:
            if component.weight > 5.0:  # Heavy components likely cause vibration
                other_pos = individual[component.id]['position']
                distance = np.linalg.norm(np.array(pos) - np.array(other_pos))
                min_distance = min(min_distance, distance)
                
        max_distance = np.linalg.norm(self.workspace_bounds)
        return min(min_distance / max_distance, 1.0) if min_distance != float('inf') else 1.0
    
    def _calculate_emi_score(self, individual: Dict) -> float:
        """Calculate electromagnetic interference minimization score."""
        total_interference = 0.0
        
        for i, comp1 in enumerate(self.components):
            pos1 = individual[comp1.id]['position']
            
            for j, comp2 in enumerate(self.components[i+1:], i+1):
                pos2 = individual[comp2.id]['position']
                distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
                
                # EMI decreases with distance squared
                interference = (comp1.electromagnetic_interference * 
                              comp2.electromagnetic_interference) / (distance**2 + 1)
                total_interference += interference
                
        # Normalize and invert (less interference = higher score)
        max_possible_interference = len(self.components)**2
        return 1.0 - min(total_interference / max_possible_interference, 1.0)
    
    def _evolve_population(self, population: List[Dict], fitness_scores: List[float]) -> List[Dict]:
        """Evolve population through selection, crossover, and mutation."""
        new_population = []
        
        # Elite selection (keep best 10%)
        elite_count = max(1, self.population_size // 10)
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # Generate rest through crossover and mutation
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy()
            
            # Mutation
            if random.random() < self.mutation_rate:
                child = self._mutate(child)
                
            new_population.append(child)
            
        return new_population[:self.population_size]
    
    def _tournament_selection(self, population: List[Dict], fitness_scores: List[float], 
                            tournament_size: int = 3) -> Dict:
        """Select individual using tournament selection."""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx]
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Create child through crossover of two parents."""
        child = {}
        
        for component_id in parent1.keys():
            # Randomly choose genetic material from either parent
            if random.random() < 0.5:
                child[component_id] = parent1[component_id].copy()
            else:
                child[component_id] = parent2[component_id].copy()
                
        return child
    
    def _mutate(self, individual: Dict) -> Dict:
        """Mutate individual by randomly adjusting positions/orientations."""
        mutated = {}
        
        for component_id, genes in individual.items():
            pos = genes['position']
            orientation = genes['orientation']
            
            # Small random adjustments
            new_pos = (
                max(0, min(self.workspace_bounds[0], pos[0] + random.gauss(0, 10))),
                max(0, min(self.workspace_bounds[1], pos[1] + random.gauss(0, 10))),
                max(0, min(self.workspace_bounds[2], pos[2] + random.gauss(0, 10)))
            )
            
            new_orientation = (
                orientation[0] + random.gauss(0, 0.1),
                orientation[1] + random.gauss(0, 0.1),
                orientation[2] + random.gauss(0, 0.1)
            )
            
            mutated[component_id] = {
                'position': new_pos,
                'orientation': new_orientation
            }
            
        return mutated
    
    def _create_solution(self, best_individual: Dict, fitness: float) -> PlacementSolution:
        """Create final solution object."""
        positions = {}
        orientations = {}
        violations = []
        
        for component_id, genes in best_individual.items():
            positions[component_id] = genes['position']
            orientations[component_id] = genes['orientation']
        
        # Check for violations
        collisions = self._detect_collisions(best_individual)
        if collisions:
            violations.extend([f"Collision between {c1} and {c2}" for c1, c2 in collisions])
            
        return PlacementSolution(
            positions=positions,
            orientations=orientations,
            fitness=fitness,
            violations=violations
        )