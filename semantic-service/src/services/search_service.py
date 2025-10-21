"""
Core search service orchestrating natural language queries across vector and hybrid search.
Handles query processing, result ranking, and context enhancement for chat interactions.
"""

import asyncio
import time
import re
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import spacy
from collections import defaultdict

from ..config.settings import get_settings
from ..core.logging import get_logger
from ..models.search_models import (
    UniversalSearchQuery, SearchResponse, SearchResult, SearchType,
    CollectionName, QueryIntent, ChatEnhancementRequest, ChatEnhancementResponse
)
from ..models.component_models import ComponentSearchQuery, ComponentSearchResponse
from ..services.embedding import EmbeddingService
from ..services.vector_store import VectorStoreService

settings = get_settings()
logger = get_logger(__name__)


class QueryProcessor:
    """Processes and enhances natural language queries."""
    
    def __init__(self):
        self.nlp = None
        self.technical_terms = self._load_technical_terms()
        self.component_synonyms = self._load_component_synonyms()
    
    async def initialize(self) -> None:
        """Initialize NLP models."""
        try:
            # Load spaCy model in executor to avoid blocking
            loop = asyncio.get_event_loop()
            self.nlp = await loop.run_in_executor(
                None, spacy.load, "en_core_web_sm"
            )
            logger.logger.info("NLP model loaded successfully")
        except Exception as e:
            logger.logger.warning("Failed to load NLP model", error=str(e))
            self.nlp = None
    
    def parse_technical_query(self, query: str) -> Dict[str, Any]:
        """Parse technical specifications and entities from query."""
        parsed = {
            "original_query": query,
            "component_types": [],
            "specifications": {},
            "actions": [],
            "locations": [],
            "intent": QueryIntent.GENERAL_QUESTION,
            "expanded_terms": []
        }
        
        query_lower = query.lower()
        
        # Extract component types
        for term, synonyms in self.component_synonyms.items():
            if any(syn in query_lower for syn in synonyms):
                parsed["component_types"].append(term)
        
        # Extract electrical specifications
        parsed["specifications"] = self._extract_specifications(query)
        
        # Extract action verbs
        parsed["actions"] = self._extract_actions(query_lower)
        
        # Extract location references
        parsed["locations"] = self._extract_locations(query_lower)
        
        # Detect intent
        parsed["intent"] = self._detect_intent(query_lower, parsed)
        
        # Expand technical terms
        parsed["expanded_terms"] = self._expand_technical_terms(query_lower)
        
        return parsed
    
    def _extract_specifications(self, query: str) -> Dict[str, Any]:
        """Extract electrical specifications from query text."""
        specs = {}
        
        # Voltage patterns
        voltage_patterns = [
            r'(\d+(?:\.\d+)?)\s*v(?:olt)?s?',
            r'(\d+(?:\.\d+)?)\s*volt(?:age)?'
        ]
        for pattern in voltage_patterns:
            matches = re.findall(pattern, query.lower())
            if matches:
                specs["voltage"] = float(matches[0])
        
        # Current patterns  
        current_patterns = [
            r'(\d+(?:\.\d+)?)\s*a(?:mp)?s?',
            r'(\d+(?:\.\d+)?)\s*amp(?:ere)?s?'
        ]
        for pattern in current_patterns:
            matches = re.findall(pattern, query.lower())
            if matches:
                specs["current"] = float(matches[0])
        
        # Power patterns
        power_patterns = [
            r'(\d+(?:\.\d+)?)\s*w(?:att)?s?',
            r'(\d+(?:\.\d+)?)\s*watt(?:age)?'
        ]
        for pattern in power_patterns:
            matches = re.findall(pattern, query.lower())
            if matches:
                specs["power"] = float(matches[0])
        
        return specs
    
    def _extract_actions(self, query: str) -> List[str]:
        """Extract action verbs from query."""
        action_keywords = {
            "find": ["find", "locate", "search", "get"],
            "troubleshoot": ["troubleshoot", "diagnose", "fix", "repair", "problem"],
            "compare": ["compare", "difference", "versus", "vs"],
            "install": ["install", "mount", "connect", "wire"],
            "replace": ["replace", "substitute", "swap", "change"]
        }
        
        actions = []
        for action, keywords in action_keywords.items():
            if any(keyword in query for keyword in keywords):
                actions.append(action)
        
        return actions
    
    def _extract_locations(self, query: str) -> List[str]:
        """Extract location references from query."""
        location_keywords = {
            "engine": ["engine", "motor", "bay", "compartment"],
            "dashboard": ["dashboard", "dash", "panel", "instrument"],
            "cabin": ["cabin", "interior", "inside"],
            "trunk": ["trunk", "rear", "cargo", "back"],
            "chassis": ["chassis", "frame", "ground"],
            "fuse_box": ["fuse box", "fusebox", "electrical box"]
        }
        
        locations = []
        for location, keywords in location_keywords.items():
            if any(keyword in query for keyword in keywords):
                locations.append(location)
        
        return locations
    
    def _detect_intent(self, query: str, parsed_data: Dict[str, Any]) -> QueryIntent:
        """Detect user intent from query and parsed data."""
        # Location-based queries
        if any(word in query for word in ["where", "location", "located", "position"]):
            return QueryIntent.LOCATE_COMPONENT
        
        # Troubleshooting queries
        if any(word in query for word in ["problem", "issue", "not working", "broken", "fault"]):
            return QueryIntent.TROUBLESHOOT
        
        # Comparison queries
        if any(word in query for word in ["compare", "difference", "better", "versus", "vs"]):
            return QueryIntent.COMPARE_COMPONENTS
        
        # Specification queries
        if any(word in query for word in ["specs", "specification", "rating", "capacity"]):
            return QueryIntent.GET_SPECIFICATIONS
        
        # Similar component queries
        if any(word in query for word in ["similar", "alternative", "equivalent", "replacement"]):
            return QueryIntent.FIND_SIMILAR
        
        # Default to component search
        if parsed_data["component_types"]:
            return QueryIntent.FIND_COMPONENT
        
        return QueryIntent.GENERAL_QUESTION
    
    def _expand_technical_terms(self, query: str) -> List[str]:
        """Expand technical terms with synonyms and related concepts."""
        expanded = []
        
        for term, expansions in self.technical_terms.items():
            if term in query:
                expanded.extend(expansions)
        
        return list(set(expanded))
    
    def _load_technical_terms(self) -> Dict[str, List[str]]:
        """Load technical term expansions."""
        return {
            "relay": ["switch", "contactor", "solenoid"],
            "fuse": ["breaker", "protection", "safety"],
            "ecu": ["computer", "module", "controller", "brain"],
            "sensor": ["detector", "monitor", "probe"],
            "battery": ["power", "electrical", "12v", "energy"],
            "alternator": ["generator", "charging", "dynamo"],
            "starter": ["cranking", "ignition", "starting"],
            "ground": ["earth", "negative", "chassis"],
            "wire": ["cable", "harness", "conductor"],
            "connector": ["plug", "socket", "terminal"]
        }
    
    def _load_component_synonyms(self) -> Dict[str, List[str]]:
        """Load component type synonyms."""
        return {
            "relay": ["relay", "relays"],
            "fuse": ["fuse", "fuses", "breaker", "breakers"],
            "battery": ["battery", "batteries"],
            "ecu": ["ecu", "ecm", "pcm", "computer", "module"],
            "sensor": ["sensor", "sensors"],
            "alternator": ["alternator", "generator"],
            "starter": ["starter", "starter motor"],
            "ground": ["ground", "earth", "chassis ground"],
            "wire": ["wire", "wires", "cable", "cables", "harness"],
            "connector": ["connector", "connectors", "plug", "plugs"],
            "switch": ["switch", "switches"],
            "motor": ["motor", "motors"],
            "pump": ["pump", "pumps"],
            "injector": ["injector", "injectors"],
            "coil": ["coil", "coils", "ignition coil"]
        }


class SearchService:
    """Core search service orchestrating all search operations."""
    
    def __init__(self, embedding_service: EmbeddingService, 
                 vector_store_service: VectorStoreService):
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.query_processor = QueryProcessor()
        
    async def initialize(self) -> None:
        """Initialize search service components."""
        await self.query_processor.initialize()
        logger.logger.info("Search service initialized")
    
    async def search(self, query: UniversalSearchQuery) -> SearchResponse:
        """Perform universal search across specified collections."""
        start_time = time.time()
        
        # Process and enhance query
        processed_query = self.query_processor.parse_technical_query(query.query)
        
        # Generate embedding for vector search
        embedding_start = time.time()
        if query.search_type in [SearchType.VECTOR, SearchType.HYBRID, SearchType.SEMANTIC]:
            query_embedding = await self.embedding_service.generate_embedding(
                query.query,
                use_cache=True
            )
        else:
            query_embedding = None
        embedding_time_ms = (time.time() - embedding_start) * 1000
        
        # Perform search across collections
        search_start = time.time()
        collection_results = await self._search_collections(
            query, query_embedding, processed_query
        )
        search_time_ms = (time.time() - search_start) * 1000
        
        # Combine and rank results
        rerank_start = time.time()
        combined_results = await self._combine_and_rank_results(
            collection_results, query, processed_query
        )
        rerank_time_ms = (time.time() - rerank_start) * 1000
        
        # Generate suggestions and enhancements
        suggestions = await self._generate_suggestions(query.query, processed_query)
        related_queries = await self._generate_related_queries(processed_query)
        
        total_time_ms = (time.time() - start_time) * 1000
        
        # Build response
        response = SearchResponse(
            query=query.query,
            search_type=query.search_type,
            processed_query=processed_query.get("original_query", query.query),
            detected_intent=processed_query.get("intent"),
            results=combined_results,
            total_found=len(combined_results),
            results_per_collection={
                collection: len(results) 
                for collection, results in collection_results.items()
            },
            search_time_ms=total_time_ms,
            embedding_time_ms=embedding_time_ms,
            vector_search_time_ms=search_time_ms,
            rerank_time_ms=rerank_time_ms,
            suggestions=suggestions,
            related_queries=related_queries,
            filters_applied=query.filters,
            search_context={
                "vehicle_signature": query.vehicle_signature,
                "collections_searched": [c.value for c in query.collections],
                "search_type": query.search_type.value
            }
        )
        
        # Log search operation
        logger.log_search_operation(
            query=query.query,
            collection="multi_collection",
            results_count=len(combined_results),
            response_time_ms=total_time_ms
        )
        
        return response
    
    async def _search_collections(self, query: UniversalSearchQuery,
                                query_embedding: Optional[List[float]],
                                processed_query: Dict[str, Any]) -> Dict[str, List[SearchResult]]:
        """Search across specified collections."""
        collection_results = {}
        
        if query_embedding:
            # Vector search across collections
            vector_results = await self.vector_store_service.search_multiple_collections(
                collections=query.collections,
                query_vector=query_embedding,
                limit_per_collection=query.limit,
                score_threshold=query.similarity_threshold,
                filters=query.filters
            )
            
            # Convert to SearchResult objects
            for collection_name, results in vector_results.items():
                search_results = []
                for i, result in enumerate(results):
                    search_result = SearchResult(
                        id=result["id"],
                        collection=CollectionName(collection_name),
                        content=result["payload"],
                        score=result["score"],
                        rank=i + 1,
                        match_type="semantic",
                        matched_fields=self._identify_matched_fields(
                            result["payload"], processed_query
                        ),
                        highlights=self._generate_highlights(
                            result["payload"], query.query
                        ),
                        explanation=self._generate_explanation(
                            result, processed_query
                        ),
                        confidence=result["score"]
                    )
                    search_results.append(search_result)
                
                collection_results[collection_name] = search_results
        
        return collection_results
    
    def _identify_matched_fields(self, payload: Dict[str, Any], 
                               processed_query: Dict[str, Any]) -> List[str]:
        """Identify which fields contributed to the match."""
        matched_fields = []
        query_terms = processed_query["original_query"].lower().split()
        
        for field, value in payload.items():
            if isinstance(value, str) and value:
                if any(term in value.lower() for term in query_terms):
                    matched_fields.append(field)
        
        return matched_fields
    
    def _generate_highlights(self, payload: Dict[str, Any], 
                           query: str) -> Dict[str, List[str]]:
        """Generate highlighted text snippets."""
        highlights = {}
        query_terms = query.lower().split()
        
        for field, value in payload.items():
            if isinstance(value, str) and value:
                field_highlights = []
                value_lower = value.lower()
                
                for term in query_terms:
                    if term in value_lower:
                        # Simple highlighting - in production, use more sophisticated methods
                        highlighted = value.replace(
                            term, f"<mark>{term}</mark>", 
                            # Case-insensitive replacement would be better
                        )
                        field_highlights.append(highlighted)
                
                if field_highlights:
                    highlights[field] = field_highlights
        
        return highlights
    
    def _generate_explanation(self, result: Dict[str, Any], 
                            processed_query: Dict[str, Any]) -> str:
        """Generate explanation for why this result was returned."""
        score = result["score"]
        payload = result["payload"]
        
        explanations = []
        
        if score > 0.9:
            explanations.append("Exact semantic match")
        elif score > 0.8:
            explanations.append("Strong semantic similarity")
        elif score > 0.7:
            explanations.append("Good semantic match")
        else:
            explanations.append("Relevant match")
        
        # Add specific matching criteria
        if processed_query.get("component_types"):
            comp_type = payload.get("component_type", "")
            if comp_type in processed_query["component_types"]:
                explanations.append(f"Matches component type: {comp_type}")
        
        if processed_query.get("specifications"):
            specs = processed_query["specifications"]
            if "voltage" in specs and payload.get("voltage_rating"):
                explanations.append(f"Voltage specification match")
        
        return "; ".join(explanations)
    
    async def _combine_and_rank_results(self, collection_results: Dict[str, List[SearchResult]],
                                      query: UniversalSearchQuery,
                                      processed_query: Dict[str, Any]) -> List[SearchResult]:
        """Combine results from multiple collections and re-rank."""
        all_results = []
        
        # Collect all results
        for collection_name, results in collection_results.items():
            all_results.extend(results)
        
        # Apply query-specific ranking boost
        for result in all_results:
            # Boost exact component type matches
            if processed_query.get("component_types"):
                comp_type = result.content.get("component_type", "")
                if comp_type in processed_query["component_types"]:
                    result.score *= query.boost_factors.get("exact_match", 1.0)
            
            # Boost vehicle-specific matches
            if query.vehicle_signature:
                result_vehicle = result.content.get("vehicle_signature", "")
                if result_vehicle == query.vehicle_signature:
                    result.score *= 1.2
        
        # Sort by score (descending)
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(all_results):
            result.rank = i + 1
        
        # Apply limit
        return all_results[:query.limit]
    
    async def _generate_suggestions(self, query: str, 
                                  processed_query: Dict[str, Any]) -> List[str]:
        """Generate query suggestions based on analysis."""
        suggestions = []
        
        # Suggest more specific queries
        if not processed_query.get("component_types"):
            suggestions.append("Try specifying a component type (e.g., 'relay', 'fuse')")
        
        if not processed_query.get("specifications"):
            suggestions.append("Add electrical specifications (e.g., '12V', '30A')")
        
        # Suggest related searches based on intent
        intent = processed_query.get("intent")
        if intent == QueryIntent.FIND_COMPONENT:
            suggestions.extend([
                "Add location context (e.g., 'in engine bay')",
                "Specify vehicle model for better results"
            ])
        elif intent == QueryIntent.LOCATE_COMPONENT:
            suggestions.append("Try searching for the component first, then ask for location")
        
        return suggestions[:3]  # Limit to top 3 suggestions
    
    async def _generate_related_queries(self, processed_query: Dict[str, Any]) -> List[str]:
        """Generate related query suggestions."""
        related = []
        
        component_types = processed_query.get("component_types", [])
        for comp_type in component_types[:2]:  # Limit to 2 types
            related.extend([
                f"How to install {comp_type}",
                f"Troubleshoot {comp_type} problems",
                f"Replace {comp_type}"
            ])
        
        return related[:5]  # Limit to 5 related queries
    
    async def enhance_chat_context(self, request: ChatEnhancementRequest) -> ChatEnhancementResponse:
        """Enhance chat context with relevant search results."""
        start_time = time.time()
        
        # Process user query
        processed_query = self.query_processor.parse_technical_query(request.user_query)
        
        # Create search query for components
        component_query = UniversalSearchQuery(
            query=request.user_query,
            search_type=SearchType.SEMANTIC,
            collections=[CollectionName.COMPONENTS],
            vehicle_signature=request.vehicle_signature,
            limit=request.max_components,
            similarity_threshold=0.6
        )
        
        # Create search query for documentation
        doc_query = UniversalSearchQuery(
            query=request.user_query,
            search_type=SearchType.SEMANTIC,
            collections=[CollectionName.DOCUMENTATION],
            vehicle_signature=request.vehicle_signature,
            limit=request.max_documentation,
            similarity_threshold=0.6
        )
        
        # Execute searches in parallel
        component_task = self.search(component_query)
        doc_task = self.search(doc_query)
        
        component_results, doc_results = await asyncio.gather(
            component_task, doc_task, return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(component_results, Exception):
            logger.logger.error("Component search failed", error=str(component_results))
            component_results = SearchResponse(
                query=request.user_query, search_type=SearchType.SEMANTIC,
                processed_query=request.user_query, results=[], total_found=0,
                search_time_ms=0, results_per_collection={}
            )
        
        if isinstance(doc_results, Exception):
            logger.logger.error("Documentation search failed", error=str(doc_results))
            doc_results = SearchResponse(
                query=request.user_query, search_type=SearchType.SEMANTIC,
                processed_query=request.user_query, results=[], total_found=0,
                search_time_ms=0, results_per_collection={}
            )
        
        # Generate spatial context if requested
        spatial_context = None
        if request.include_spatial and request.model_metadata:
            spatial_context = await self._generate_spatial_context(
                component_results.results, request.model_metadata
            )
        
        # Generate relationship context if requested
        relationship_context = None
        if request.include_relationships:
            relationship_context = await self._generate_relationship_context(
                component_results.results
            )
        
        # Generate suggestions
        suggested_questions = await self._generate_suggested_questions(
            processed_query, component_results.results
        )
        
        enhancement_time_ms = (time.time() - start_time) * 1000
        
        return ChatEnhancementResponse(
            user_query=request.user_query,
            enhanced_query=processed_query.get("original_query", request.user_query),
            detected_intent=processed_query.get("intent"),
            relevant_components=component_results.results,
            relevant_documentation=doc_results.results,
            spatial_context=spatial_context,
            relationship_context=relationship_context,
            suggested_questions=suggested_questions,
            related_topics=processed_query.get("expanded_terms", []),
            enhancement_time_ms=enhancement_time_ms
        )
    
    async def _generate_spatial_context(self, component_results: List[SearchResult],
                                      model_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate spatial context from 3D model metadata."""
        spatial_context = {
            "model_id": model_metadata.get("model_id"),
            "vehicle_signature": model_metadata.get("vehicle_signature"),
            "component_locations": []
        }
        
        for result in component_results:
            position_3d = result.content.get("position_3d")
            if position_3d:
                spatial_context["component_locations"].append({
                    "component_id": result.id,
                    "name": result.content.get("canonical_id"),
                    "position": position_3d,
                    "anchor_zone": result.content.get("anchor_zone")
                })
        
        return spatial_context
    
    async def _generate_relationship_context(self, 
                                           component_results: List[SearchResult]) -> Dict[str, Any]:
        """Generate component relationship context."""
        # This would integrate with Neo4j to get relationships
        # For now, return placeholder structure
        relationship_context = {
            "connected_components": [],
            "circuit_context": None,
            "subsystem_context": None
        }
        
        return relationship_context
    
    async def _generate_suggested_questions(self, processed_query: Dict[str, Any],
                                          component_results: List[SearchResult]) -> List[str]:
        """Generate contextual follow-up questions."""
        suggestions = []
        
        intent = processed_query.get("intent")
        
        if intent == QueryIntent.FIND_COMPONENT and component_results:
            comp_name = component_results[0].content.get("canonical_id", "component")
            suggestions.extend([
                f"Where is the {comp_name} located?",
                f"How to test the {comp_name}?",
                f"What is connected to the {comp_name}?"
            ])
        
        elif intent == QueryIntent.LOCATE_COMPONENT:
            suggestions.extend([
                "How do I access this component?",
                "What tools do I need?",
                "Are there any safety precautions?"
            ])
        
        elif intent == QueryIntent.TROUBLESHOOT:
            suggestions.extend([
                "What are common symptoms?",
                "How to diagnose the problem?",
                "What replacement parts are needed?"
            ])
        
        return suggestions[:5]