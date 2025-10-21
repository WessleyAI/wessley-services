"""
Intelligent component recommendation service.
Provides suggestions based on similarity, compatibility, and usage patterns.
"""

import asyncio
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
import math

from ..config.settings import get_settings
from ..core.logging import get_logger
from ..models.component_models import (
    ElectricalComponent, ComponentRecommendation, ComponentRecommendationResponse,
    ComponentType, AnchorZone
)
from ..models.search_models import CollectionName, SearchFilter
from .vector_store import VectorStoreService
from .embedding import EmbeddingService

settings = get_settings()
logger = get_logger(__name__)


class RecommendationService:
    """Service for generating intelligent component recommendations."""
    
    def __init__(self):
        self.vector_store = None
        self.embedding_service = None
        self.similarity_weights = {
            "semantic": 0.4,
            "electrical": 0.3,
            "functional": 0.2,
            "spatial": 0.1
        }
        
    async def initialize(self) -> None:
        """Initialize recommendation service dependencies."""
        logger.logger.info("Initializing recommendation service")
        
        self.vector_store = VectorStoreService()
        await self.vector_store.initialize()
        
        self.embedding_service = EmbeddingService()
        await self.embedding_service.initialize()
        
        logger.logger.info("Recommendation service initialized")
    
    async def close(self) -> None:
        """Close service connections."""
        if self.vector_store:
            await self.vector_store.close()
        if self.embedding_service:
            await self.embedding_service.close()
    
    async def get_component_recommendations(self, 
                                         component_id: str,
                                         limit: int = 10,
                                         similarity_threshold: float = 0.6,
                                         recommendation_types: List[str] = None) -> ComponentRecommendationResponse:
        """
        Generate comprehensive component recommendations.
        """
        start_time = datetime.utcnow()
        
        try:
            # Get source component
            source_component = await self._get_component_by_id(component_id)
            if not source_component:
                raise ValueError(f"Component not found: {component_id}")
            
            # Default recommendation types
            if not recommendation_types:
                recommendation_types = ["similar", "compatible", "alternative", "related"]
            
            # Generate different types of recommendations
            all_recommendations = []
            
            for rec_type in recommendation_types:
                if rec_type == "similar":
                    similar = await self._get_similar_components(
                        source_component, limit, similarity_threshold
                    )
                    all_recommendations.extend(similar)
                elif rec_type == "compatible":
                    compatible = await self._get_compatible_components(
                        source_component, limit, similarity_threshold
                    )
                    all_recommendations.extend(compatible)
                elif rec_type == "alternative":
                    alternatives = await self._get_alternative_components(
                        source_component, limit, similarity_threshold
                    )
                    all_recommendations.extend(alternatives)
                elif rec_type == "related":
                    related = await self._get_related_components(
                        source_component, limit, similarity_threshold
                    )
                    all_recommendations.extend(related)
            
            # Remove duplicates and rank
            unique_recommendations = self._deduplicate_recommendations(all_recommendations)
            ranked_recommendations = self._rank_recommendations(
                source_component, unique_recommendations
            )
            
            # Apply final limit
            final_recommendations = ranked_recommendations[:limit]
            
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            logger.logger.info(
                "Recommendations generated",
                component_id=component_id,
                recommendations_count=len(final_recommendations),
                response_time_ms=response_time
            )
            
            return ComponentRecommendationResponse(
                source_component_id=component_id,
                recommendations=final_recommendations,
                recommendation_time_ms=response_time
            )
            
        except Exception as e:
            logger.logger.error(
                "Recommendation generation failed",
                component_id=component_id,
                error=str(e)
            )
            raise
    
    async def _get_component_by_id(self, component_id: str) -> Optional[ElectricalComponent]:
        """Retrieve component by ID from vector store."""
        try:
            result = await self.vector_store.get_point_by_id(
                CollectionName.COMPONENTS, component_id
            )
            
            if result and result.get("payload"):
                # Convert payload back to ElectricalComponent
                return self._payload_to_component(result["payload"])
            
        except Exception as e:
            logger.logger.error(
                "Failed to retrieve component",
                component_id=component_id,
                error=str(e)
            )
        
        return None
    
    async def _get_similar_components(self, 
                                    source_component: ElectricalComponent,
                                    limit: int,
                                    threshold: float) -> List[ComponentRecommendation]:
        """Find components similar by description and function."""
        try:
            # Generate embedding for source component
            component_text = self.embedding_service.build_component_text(source_component)
            query_vector = await self.embedding_service.generate_embedding(component_text)
            
            # Search for similar components
            search_results = await self.vector_store.search_vectors(
                collection_name=CollectionName.COMPONENTS,
                query_vector=query_vector,
                limit=limit + 1,  # +1 to exclude source component
                score_threshold=threshold
            )
            
            recommendations = []
            for result in search_results:
                # Skip source component
                if result["id"] == source_component.component_id:
                    continue
                
                component = self._payload_to_component(result["payload"])
                if component:
                    recommendation = ComponentRecommendation(
                        component=component,
                        recommendation_score=result["score"],
                        reason=f"Similar description and functionality to {source_component.canonical_id}",
                        similarity_type="semantic_similarity"
                    )
                    recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.logger.error("Failed to get similar components", error=str(e))
            return []
    
    async def _get_compatible_components(self, 
                                       source_component: ElectricalComponent,
                                       limit: int,
                                       threshold: float) -> List[ComponentRecommendation]:
        """Find electrically compatible components."""
        try:
            filters = []
            
            # Filter by electrical compatibility
            specs = source_component.specifications
            if specs.voltage_rating:
                # Components that work with same voltage ±20%
                voltage_tolerance = specs.voltage_rating * 0.2
                filters.extend([
                    SearchFilter(
                        field="voltage_rating",
                        operator="gte",
                        value=specs.voltage_rating - voltage_tolerance
                    ),
                    SearchFilter(
                        field="voltage_rating",
                        operator="lte", 
                        value=specs.voltage_rating + voltage_tolerance
                    )
                ])
            
            # Same vehicle signature for compatibility
            if source_component.vehicle_signature:
                filters.append(SearchFilter(
                    field="vehicle_signature",
                    operator="eq",
                    value=source_component.vehicle_signature
                ))
            
            # Search with compatibility filters
            component_text = self.embedding_service.build_component_text(source_component)
            query_vector = await self.embedding_service.generate_embedding(component_text)
            
            search_results = await self.vector_store.search_vectors(
                collection_name=CollectionName.COMPONENTS,
                query_vector=query_vector,
                limit=limit + 1,
                score_threshold=threshold * 0.8,  # Lower threshold for compatibility
                filters=filters
            )
            
            recommendations = []
            for result in search_results:
                if result["id"] == source_component.component_id:
                    continue
                
                component = self._payload_to_component(result["payload"])
                if component:
                    compatibility_score = self._calculate_electrical_compatibility(
                        source_component, component
                    )
                    
                    recommendation = ComponentRecommendation(
                        component=component,
                        recommendation_score=compatibility_score,
                        reason=f"Electrically compatible with {source_component.canonical_id}",
                        similarity_type="electrical_compatibility"
                    )
                    recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.logger.error("Failed to get compatible components", error=str(e))
            return []
    
    async def _get_alternative_components(self, 
                                        source_component: ElectricalComponent,
                                        limit: int,
                                        threshold: float) -> List[ComponentRecommendation]:
        """Find alternative components that can serve the same function."""
        try:
            filters = []
            
            # Same component type
            filters.append(SearchFilter(
                field="component_type",
                operator="eq",
                value=source_component.component_type.value
            ))
            
            # Same anchor zone (physical location)
            filters.append(SearchFilter(
                field="anchor_zone",
                operator="eq", 
                value=source_component.anchor_zone.value
            ))
            
            # Different manufacturer for alternatives
            if source_component.manufacturer:
                filters.append(SearchFilter(
                    field="manufacturer",
                    operator="ne",
                    value=source_component.manufacturer
                ))
            
            # Search for alternatives
            function_text = f"{source_component.function} {source_component.purpose}"
            query_vector = await self.embedding_service.generate_embedding(function_text)
            
            search_results = await self.vector_store.search_vectors(
                collection_name=CollectionName.COMPONENTS,
                query_vector=query_vector,
                limit=limit + 1,
                score_threshold=threshold * 0.7,
                filters=filters
            )
            
            recommendations = []
            for result in search_results:
                if result["id"] == source_component.component_id:
                    continue
                
                component = self._payload_to_component(result["payload"])
                if component:
                    functional_score = self._calculate_functional_similarity(
                        source_component, component
                    )
                    
                    recommendation = ComponentRecommendation(
                        component=component,
                        recommendation_score=functional_score,
                        reason=f"Alternative to {source_component.canonical_id} with same function",
                        similarity_type="functional_alternative"
                    )
                    recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.logger.error("Failed to get alternative components", error=str(e))
            return []
    
    async def _get_related_components(self, 
                                    source_component: ElectricalComponent,
                                    limit: int,
                                    threshold: float) -> List[ComponentRecommendation]:
        """Find components commonly used with the source component."""
        try:
            # Find components in same circuit or subsystem
            filters = []
            
            if source_component.part_of_circuit:
                filters.append(SearchFilter(
                    field="part_of_circuit",
                    operator="eq",
                    value=source_component.part_of_circuit
                ))
            
            if source_component.subsystem:
                filters.append(SearchFilter(
                    field="subsystem",
                    operator="eq",
                    value=source_component.subsystem
                ))
            
            # Search within same vehicle
            if source_component.vehicle_signature:
                filters.append(SearchFilter(
                    field="vehicle_signature",
                    operator="eq",
                    value=source_component.vehicle_signature
                ))
            
            # Use categories and tags for related search
            search_terms = source_component.categories + source_component.tags
            search_text = " ".join(search_terms)
            
            if search_text:
                query_vector = await self.embedding_service.generate_embedding(search_text)
                
                search_results = await self.vector_store.search_vectors(
                    collection_name=CollectionName.COMPONENTS,
                    query_vector=query_vector,
                    limit=limit + 1,
                    score_threshold=threshold * 0.6,
                    filters=filters
                )
                
                recommendations = []
                for result in search_results:
                    if result["id"] == source_component.component_id:
                        continue
                    
                    component = self._payload_to_component(result["payload"])
                    if component:
                        relation_score = self._calculate_relationship_strength(
                            source_component, component
                        )
                        
                        recommendation = ComponentRecommendation(
                            component=component,
                            recommendation_score=relation_score,
                            reason=f"Commonly used with {source_component.canonical_id}",
                            similarity_type="system_relationship"
                        )
                        recommendations.append(recommendation)
                
                return recommendations
            
        except Exception as e:
            logger.logger.error("Failed to get related components", error=str(e))
        
        return []
    
    def _calculate_electrical_compatibility(self, 
                                          comp1: ElectricalComponent,
                                          comp2: ElectricalComponent) -> float:
        """Calculate electrical compatibility score between components."""
        score = 0.0
        factors = 0
        
        specs1 = comp1.specifications
        specs2 = comp2.specifications
        
        # Voltage compatibility
        if specs1.voltage_rating and specs2.voltage_rating:
            voltage_diff = abs(specs1.voltage_rating - specs2.voltage_rating)
            max_voltage = max(specs1.voltage_rating, specs2.voltage_rating)
            voltage_score = max(0, 1 - (voltage_diff / max_voltage))
            score += voltage_score
            factors += 1
        
        # Current compatibility
        if specs1.current_rating and specs2.current_rating:
            current_ratio = min(specs1.current_rating, specs2.current_rating) / max(specs1.current_rating, specs2.current_rating)
            score += current_ratio
            factors += 1
        
        # Same component type bonus
        if comp1.component_type == comp2.component_type:
            score += 0.5
            factors += 1
        
        return score / factors if factors > 0 else 0.0
    
    def _calculate_functional_similarity(self, 
                                       comp1: ElectricalComponent,
                                       comp2: ElectricalComponent) -> float:
        """Calculate functional similarity score."""
        score = 0.0
        factors = 0
        
        # Same component type
        if comp1.component_type == comp2.component_type:
            score += 0.4
            factors += 1
        
        # Same anchor zone
        if comp1.anchor_zone == comp2.anchor_zone:
            score += 0.3
            factors += 1
        
        # Function similarity (basic text overlap)
        if comp1.function and comp2.function:
            func_words1 = set(comp1.function.lower().split())
            func_words2 = set(comp2.function.lower().split())
            if func_words1 and func_words2:
                overlap = len(func_words1.intersection(func_words2))
                union = len(func_words1.union(func_words2))
                func_score = overlap / union if union > 0 else 0
                score += func_score * 0.3
                factors += 1
        
        return score / factors if factors > 0 else 0.0
    
    def _calculate_relationship_strength(self, 
                                       comp1: ElectricalComponent,
                                       comp2: ElectricalComponent) -> float:
        """Calculate relationship strength between components."""
        score = 0.0
        factors = 0
        
        # Same circuit
        if comp1.part_of_circuit and comp2.part_of_circuit:
            if comp1.part_of_circuit == comp2.part_of_circuit:
                score += 0.5
            factors += 1
        
        # Same subsystem
        if comp1.subsystem and comp2.subsystem:
            if comp1.subsystem == comp2.subsystem:
                score += 0.3
            factors += 1
        
        # Category overlap
        if comp1.categories and comp2.categories:
            cat_overlap = len(set(comp1.categories).intersection(set(comp2.categories)))
            cat_union = len(set(comp1.categories).union(set(comp2.categories)))
            if cat_union > 0:
                score += (cat_overlap / cat_union) * 0.2
                factors += 1
        
        return score / factors if factors > 0 else 0.0
    
    def _deduplicate_recommendations(self, 
                                   recommendations: List[ComponentRecommendation]) -> List[ComponentRecommendation]:
        """Remove duplicate recommendations, keeping highest scored ones."""
        seen_ids = set()
        unique_recommendations = []
        
        # Sort by score descending
        sorted_recs = sorted(recommendations, key=lambda r: r.recommendation_score, reverse=True)
        
        for rec in sorted_recs:
            if rec.component.component_id not in seen_ids:
                seen_ids.add(rec.component.component_id)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _rank_recommendations(self, 
                            source_component: ElectricalComponent,
                            recommendations: List[ComponentRecommendation]) -> List[ComponentRecommendation]:
        """Apply final ranking to recommendations."""
        for rec in recommendations:
            # Apply similarity type weights
            type_weight = {
                "semantic_similarity": 1.0,
                "electrical_compatibility": 0.9,
                "functional_alternative": 0.8,
                "system_relationship": 0.7
            }.get(rec.similarity_type, 0.5)
            
            # Boost score for same manufacturer (brand consistency)
            if (source_component.manufacturer and rec.component.manufacturer and
                source_component.manufacturer == rec.component.manufacturer):
                type_weight *= 1.1
            
            # Apply final weighted score
            rec.recommendation_score = min(1.0, rec.recommendation_score * type_weight)
        
        # Sort by final score
        return sorted(recommendations, key=lambda r: r.recommendation_score, reverse=True)
    
    def _payload_to_component(self, payload: Dict[str, Any]) -> Optional[ElectricalComponent]:
        """Convert Qdrant payload back to ElectricalComponent model."""
        try:
            # This is a simplified conversion - in practice you'd want more robust parsing
            component_data = {
                "component_id": payload.get("component_id"),
                "vehicle_signature": payload.get("vehicle_signature"),
                "canonical_id": payload.get("canonical_id"),
                "code_id": payload.get("code_id"),
                "component_type": payload.get("component_type"),
                "node_type": payload.get("node_type"),
                "anchor_zone": payload.get("anchor_zone"),
                "description": payload.get("description"),
                "function": payload.get("function"),
                "purpose": payload.get("purpose"),
                "manufacturer": payload.get("manufacturer"),
                "part_number": payload.get("part_number"),
                "model_number": payload.get("model_number"),
                "categories": payload.get("categories", []),
                "tags": payload.get("tags", []),
                "keywords": payload.get("keywords", []),
                "position_3d": payload.get("position_3d"),
                "specifications": {
                    "voltage_rating": payload.get("voltage_rating"),
                    "current_rating": payload.get("current_rating"),
                    "power_rating": payload.get("power_rating")
                }
            }
            
            return ElectricalComponent(**component_data)
            
        except Exception as e:
            logger.logger.warning("Failed to convert payload to component", error=str(e))
            return None