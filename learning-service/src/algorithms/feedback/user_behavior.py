"""
User interaction analysis for continuous learning and system improvement.
Tracks user behavior patterns to optimize the electrical system generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json

logger = logging.getLogger(__name__)

class InteractionType(Enum):
    COMPONENT_SELECTION = "component_selection"
    COMPONENT_PLACEMENT = "component_placement"
    WIRE_ROUTING = "wire_routing"
    LAYOUT_APPROVAL = "layout_approval"
    LAYOUT_REJECTION = "layout_rejection"
    MANUAL_ADJUSTMENT = "manual_adjustment"
    ZOOM_PAN = "zoom_pan"
    VIEW_CHANGE = "view_change"
    PROPERTY_EDIT = "property_edit"
    EXPORT_DIAGRAM = "export_diagram"
    SAVE_PROJECT = "save_project"
    UNDO_REDO = "undo_redo"

class FeedbackType(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    SUGGESTION = "suggestion"

@dataclass
class UserInteraction:
    """Individual user interaction event."""
    session_id: str
    user_id: str
    timestamp: datetime
    interaction_type: InteractionType
    target_object_id: Optional[str]
    target_object_type: Optional[str]  # "component", "wire", "layout"
    before_state: Dict[str, Any]
    after_state: Dict[str, Any]
    interaction_data: Dict[str, Any]
    duration_ms: int
    success: bool

@dataclass
class UserFeedback:
    """User feedback on generated layouts."""
    session_id: str
    user_id: str
    timestamp: datetime
    layout_id: str
    feedback_type: FeedbackType
    rating: int  # 1-5 scale
    specific_issues: List[str]
    suggestions: List[str]
    preferred_alternatives: List[str]
    context: Dict[str, Any]

@dataclass
class UserProfile:
    """User behavioral profile."""
    user_id: str
    experience_level: str  # "novice", "intermediate", "expert"
    interaction_patterns: Dict[str, float]
    preferences: Dict[str, Any]
    frequent_components: List[str]
    typical_layout_complexity: float
    avg_session_duration: float
    avg_interactions_per_session: float
    error_patterns: List[str]
    last_updated: datetime

@dataclass
class SessionAnalysis:
    """Analysis of a user session."""
    session_id: str
    user_id: str
    start_time: datetime
    end_time: datetime
    total_interactions: int
    interaction_breakdown: Dict[InteractionType, int]
    task_completion_rate: float
    efficiency_score: float
    error_rate: float
    satisfaction_indicators: List[str]
    learning_opportunities: List[str]

class UserBehaviorAnalyzer:
    """Analyzes user behavior patterns for system optimization."""
    
    def __init__(self):
        self.interactions_buffer: List[UserInteraction] = []
        self.feedback_buffer: List[UserFeedback] = []
        self.user_profiles: Dict[str, UserProfile] = {}
        self.session_cache: Dict[str, SessionAnalysis] = {}
        
        # Analysis parameters
        self.interaction_window_hours = 24
        self.min_session_duration_minutes = 5
        self.expertise_threshold_interactions = 100
        
        # Clustering models for pattern detection
        self.interaction_clusterer = None
        self.preference_clusterer = None
        
    def record_interaction(self, interaction: UserInteraction):
        """Record a user interaction event."""
        self.interactions_buffer.append(interaction)
        
        # Update user profile incrementally
        self._update_user_profile_incremental(interaction)
        
        # Trigger analysis if buffer is full
        if len(self.interactions_buffer) > 1000:
            self._process_interaction_buffer()
    
    def record_feedback(self, feedback: UserFeedback):
        """Record user feedback."""
        self.feedback_buffer.append(feedback)
        
        # Update user profile with feedback
        self._incorporate_feedback_into_profile(feedback)
        
        # Process high-priority feedback immediately
        if feedback.feedback_type == FeedbackType.NEGATIVE:
            self._analyze_negative_feedback(feedback)
    
    def analyze_user_session(self, session_id: str) -> SessionAnalysis:
        """Analyze a complete user session."""
        # Get all interactions for this session
        session_interactions = [
            interaction for interaction in self.interactions_buffer
            if interaction.session_id == session_id
        ]
        
        if not session_interactions:
            raise ValueError(f"No interactions found for session {session_id}")
        
        # Sort by timestamp
        session_interactions.sort(key=lambda x: x.timestamp)
        
        # Calculate session metrics
        start_time = session_interactions[0].timestamp
        end_time = session_interactions[-1].timestamp
        duration = end_time - start_time
        
        # Interaction breakdown
        interaction_breakdown = {}
        for interaction_type in InteractionType:
            count = sum(1 for i in session_interactions if i.interaction_type == interaction_type)
            interaction_breakdown[interaction_type] = count
        
        # Calculate efficiency and task completion
        efficiency_score = self._calculate_session_efficiency(session_interactions)
        completion_rate = self._calculate_task_completion_rate(session_interactions)
        error_rate = self._calculate_error_rate(session_interactions)
        
        # Identify satisfaction indicators
        satisfaction_indicators = self._identify_satisfaction_indicators(session_interactions)
        
        # Identify learning opportunities
        learning_opportunities = self._identify_learning_opportunities(session_interactions)
        
        analysis = SessionAnalysis(
            session_id=session_id,
            user_id=session_interactions[0].user_id,
            start_time=start_time,
            end_time=end_time,
            total_interactions=len(session_interactions),
            interaction_breakdown=interaction_breakdown,
            task_completion_rate=completion_rate,
            efficiency_score=efficiency_score,
            error_rate=error_rate,
            satisfaction_indicators=satisfaction_indicators,
            learning_opportunities=learning_opportunities
        )
        
        self.session_cache[session_id] = analysis
        return analysis
    
    def identify_user_patterns(self, user_id: str, 
                             time_window_hours: int = 168) -> Dict[str, Any]:
        """Identify behavioral patterns for a specific user."""
        # Get recent interactions
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_interactions = [
            i for i in self.interactions_buffer
            if i.user_id == user_id and i.timestamp > cutoff_time
        ]
        
        if not recent_interactions:
            return {"error": "No recent interactions found"}
        
        patterns = {}
        
        # 1. Temporal patterns
        patterns['temporal'] = self._analyze_temporal_patterns(recent_interactions)
        
        # 2. Interaction sequence patterns
        patterns['sequences'] = self._analyze_interaction_sequences(recent_interactions)
        
        # 3. Error patterns
        patterns['errors'] = self._analyze_error_patterns(recent_interactions)
        
        # 4. Efficiency patterns
        patterns['efficiency'] = self._analyze_efficiency_patterns(recent_interactions)
        
        # 5. Preference patterns
        patterns['preferences'] = self._analyze_preference_patterns(recent_interactions)
        
        return patterns
    
    def generate_personalized_recommendations(self, user_id: str) -> Dict[str, List[str]]:
        """Generate personalized recommendations for user experience improvement."""
        profile = self.user_profiles.get(user_id)
        if not profile:
            return {"error": "User profile not found"}
        
        recommendations = {
            "ui_improvements": [],
            "workflow_optimizations": [],
            "learning_resources": [],
            "feature_suggestions": []
        }
        
        # Analyze user patterns to generate recommendations
        patterns = self.identify_user_patterns(user_id)
        
        # UI Improvements
        if profile.experience_level == "novice":
            recommendations["ui_improvements"].extend([
                "Enable guided tutorials for complex tasks",
                "Show tooltips for advanced features",
                "Highlight recommended actions"
            ])
        
        # Check for frequent error patterns
        error_patterns = patterns.get('errors', {})
        if error_patterns.get('high_undo_rate', False):
            recommendations["workflow_optimizations"].append(
                "Implement smart undo with preview"
            )
        
        if error_patterns.get('frequent_misplacements', False):
            recommendations["ui_improvements"].append(
                "Add snap-to-grid and alignment guides"
            )
        
        # Efficiency improvements
        efficiency_patterns = patterns.get('efficiency', {})
        if efficiency_patterns.get('slow_component_selection', False):
            recommendations["feature_suggestions"].append(
                "Quick component search and favorites"
            )
        
        # Learning recommendations
        if profile.experience_level in ["novice", "intermediate"]:
            common_errors = profile.error_patterns
            if "wire_routing_violations" in common_errors:
                recommendations["learning_resources"].append(
                    "Wire routing best practices tutorial"
                )
            if "clearance_violations" in common_errors:
                recommendations["learning_resources"].append(
                    "Electrical clearance requirements guide"
                )
        
        return recommendations
    
    def detect_usability_issues(self, min_occurrence_rate: float = 0.1) -> List[Dict[str, Any]]:
        """Detect system-wide usability issues from user behavior."""
        issues = []
        
        # Analyze all recent interactions
        recent_interactions = [
            i for i in self.interactions_buffer
            if i.timestamp > datetime.now() - timedelta(hours=self.interaction_window_hours)
        ]
        
        if not recent_interactions:
            return issues
        
        total_interactions = len(recent_interactions)
        
        # 1. High error rates on specific features
        error_by_type = {}
        for interaction in recent_interactions:
            if not interaction.success:
                error_type = interaction.interaction_type
                error_by_type[error_type] = error_by_type.get(error_type, 0) + 1
        
        for error_type, count in error_by_type.items():
            error_rate = count / total_interactions
            if error_rate > min_occurrence_rate:
                issues.append({
                    "type": "high_error_rate",
                    "feature": error_type.value,
                    "error_rate": error_rate,
                    "severity": "high" if error_rate > 0.3 else "medium",
                    "recommendation": f"Investigate {error_type.value} usability"
                })
        
        # 2. Frequent undo operations
        undo_count = sum(1 for i in recent_interactions 
                        if i.interaction_type == InteractionType.UNDO_REDO)
        undo_rate = undo_count / total_interactions
        
        if undo_rate > 0.15:  # More than 15% undo operations
            issues.append({
                "type": "high_undo_rate",
                "rate": undo_rate,
                "severity": "medium",
                "recommendation": "Improve action prediction and confirmation dialogs"
            })
        
        # 3. Long task completion times
        session_durations = []
        sessions = {}
        for interaction in recent_interactions:
            session_id = interaction.session_id
            if session_id not in sessions:
                sessions[session_id] = {"start": interaction.timestamp, "end": interaction.timestamp}
            else:
                sessions[session_id]["end"] = max(sessions[session_id]["end"], interaction.timestamp)
        
        for session_data in sessions.values():
            duration = (session_data["end"] - session_data["start"]).total_seconds() / 60
            if duration > self.min_session_duration_minutes:
                session_durations.append(duration)
        
        if session_durations:
            avg_duration = np.mean(session_durations)
            if avg_duration > 30:  # More than 30 minutes average
                issues.append({
                    "type": "long_task_completion",
                    "avg_duration_minutes": avg_duration,
                    "severity": "medium",
                    "recommendation": "Streamline workflows and add progress indicators"
                })
        
        # 4. Feature abandonment
        feature_usage = {}
        for interaction in recent_interactions:
            feature = interaction.interaction_type
            feature_usage[feature] = feature_usage.get(feature, 0) + 1
        
        total_features = len(InteractionType)
        unused_features = total_features - len(feature_usage)
        
        if unused_features > total_features * 0.3:
            issues.append({
                "type": "feature_abandonment",
                "unused_feature_count": unused_features,
                "severity": "low",
                "recommendation": "Improve feature discoverability and onboarding"
            })
        
        return issues
    
    def _update_user_profile_incremental(self, interaction: UserInteraction):
        """Update user profile incrementally with new interaction."""
        user_id = interaction.user_id
        
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                experience_level="novice",
                interaction_patterns={},
                preferences={},
                frequent_components=[],
                typical_layout_complexity=1.0,
                avg_session_duration=0.0,
                avg_interactions_per_session=0.0,
                error_patterns=[],
                last_updated=datetime.now()
            )
        
        profile = self.user_profiles[user_id]
        
        # Update interaction patterns
        interaction_type = interaction.interaction_type.value
        if interaction_type not in profile.interaction_patterns:
            profile.interaction_patterns[interaction_type] = 0
        profile.interaction_patterns[interaction_type] += 1
        
        # Update experience level based on total interactions
        total_interactions = sum(profile.interaction_patterns.values())
        if total_interactions > self.expertise_threshold_interactions:
            if profile.experience_level == "novice":
                profile.experience_level = "intermediate"
        if total_interactions > self.expertise_threshold_interactions * 3:
            if profile.experience_level == "intermediate":
                profile.experience_level = "expert"
        
        # Track component usage
        if (interaction.target_object_type == "component" and 
            interaction.target_object_id):
            if interaction.target_object_id not in profile.frequent_components:
                profile.frequent_components.append(interaction.target_object_id)
        
        # Track error patterns
        if not interaction.success:
            error_pattern = f"{interaction_type}_error"
            if error_pattern not in profile.error_patterns:
                profile.error_patterns.append(error_pattern)
        
        profile.last_updated = datetime.now()
    
    def _incorporate_feedback_into_profile(self, feedback: UserFeedback):
        """Incorporate user feedback into user profile."""
        user_id = feedback.user_id
        if user_id not in self.user_profiles:
            return
        
        profile = self.user_profiles[user_id]
        
        # Update preferences based on feedback
        if feedback.feedback_type == FeedbackType.POSITIVE:
            # Extract positive patterns from layout
            for issue in feedback.specific_issues:
                if issue.startswith("good_"):
                    preference_key = issue.replace("good_", "prefer_")
                    profile.preferences[preference_key] = True
        
        elif feedback.feedback_type == FeedbackType.NEGATIVE:
            # Extract negative patterns
            for issue in feedback.specific_issues:
                preference_key = f"avoid_{issue}"
                profile.preferences[preference_key] = True
    
    def _analyze_negative_feedback(self, feedback: UserFeedback):
        """Analyze negative feedback for immediate system improvements."""
        logger.warning(f"Negative feedback received: {feedback.specific_issues}")
        
        # Categorize issues
        layout_issues = []
        ui_issues = []
        performance_issues = []
        
        for issue in feedback.specific_issues:
            if any(keyword in issue.lower() for keyword in 
                   ["placement", "routing", "clearance", "layout"]):
                layout_issues.append(issue)
            elif any(keyword in issue.lower() for keyword in 
                    ["slow", "lag", "freeze", "crash"]):
                performance_issues.append(issue)
            else:
                ui_issues.append(issue)
        
        # Log categorized issues for system improvement
        if layout_issues:
            logger.info(f"Layout algorithm issues: {layout_issues}")
        if ui_issues:
            logger.info(f"UI/UX issues: {ui_issues}")
        if performance_issues:
            logger.info(f"Performance issues: {performance_issues}")
    
    def _calculate_session_efficiency(self, interactions: List[UserInteraction]) -> float:
        """Calculate efficiency score for a session."""
        if not interactions:
            return 0.0
        
        # Factors that contribute to efficiency
        total_interactions = len(interactions)
        successful_interactions = sum(1 for i in interactions if i.success)
        undo_count = sum(1 for i in interactions 
                        if i.interaction_type == InteractionType.UNDO_REDO)
        
        # Calculate base efficiency
        success_rate = successful_interactions / total_interactions
        undo_penalty = min(undo_count / total_interactions, 0.5)
        
        efficiency = success_rate - undo_penalty
        return max(0.0, min(1.0, efficiency))
    
    def _calculate_task_completion_rate(self, interactions: List[UserInteraction]) -> float:
        """Calculate task completion rate for a session."""
        # Count goal-oriented interactions
        completion_indicators = [
            InteractionType.LAYOUT_APPROVAL,
            InteractionType.EXPORT_DIAGRAM,
            InteractionType.SAVE_PROJECT
        ]
        
        completion_count = sum(1 for i in interactions 
                             if i.interaction_type in completion_indicators)
        
        # Estimate based on session length and completion indicators
        if completion_count > 0:
            return 1.0
        elif len(interactions) > 10:  # Substantial work but no completion
            return 0.5
        else:
            return 0.0
    
    def _calculate_error_rate(self, interactions: List[UserInteraction]) -> float:
        """Calculate error rate for a session."""
        if not interactions:
            return 0.0
        
        error_count = sum(1 for i in interactions if not i.success)
        return error_count / len(interactions)
    
    def _identify_satisfaction_indicators(self, interactions: List[UserInteraction]) -> List[str]:
        """Identify indicators of user satisfaction."""
        indicators = []
        
        # Positive indicators
        approval_count = sum(1 for i in interactions 
                           if i.interaction_type == InteractionType.LAYOUT_APPROVAL)
        if approval_count > 0:
            indicators.append("layout_approved")
        
        save_count = sum(1 for i in interactions 
                        if i.interaction_type == InteractionType.SAVE_PROJECT)
        if save_count > 0:
            indicators.append("project_saved")
        
        # Check for smooth workflow (low undo rate)
        undo_count = sum(1 for i in interactions 
                        if i.interaction_type == InteractionType.UNDO_REDO)
        undo_rate = undo_count / len(interactions) if interactions else 0
        
        if undo_rate < 0.1:
            indicators.append("smooth_workflow")
        
        return indicators
    
    def _identify_learning_opportunities(self, interactions: List[UserInteraction]) -> List[str]:
        """Identify opportunities for user learning and improvement."""
        opportunities = []
        
        # High error rate on specific actions
        error_by_type = {}
        for interaction in interactions:
            if not interaction.success:
                error_type = interaction.interaction_type
                error_by_type[error_type] = error_by_type.get(error_type, 0) + 1
        
        for error_type, count in error_by_type.items():
            if count > 2:  # Multiple errors of same type
                opportunities.append(f"improve_{error_type.value}_skills")
        
        # Frequent manual adjustments might indicate need for better understanding
        manual_adjustments = sum(1 for i in interactions 
                               if i.interaction_type == InteractionType.MANUAL_ADJUSTMENT)
        if manual_adjustments > len(interactions) * 0.3:
            opportunities.append("learn_automated_optimization")
        
        return opportunities
    
    def _analyze_temporal_patterns(self, interactions: List[UserInteraction]) -> Dict[str, Any]:
        """Analyze temporal patterns in user interactions."""
        if not interactions:
            return {}
        
        # Group by hour of day
        hourly_activity = {}
        for interaction in interactions:
            hour = interaction.timestamp.hour
            hourly_activity[hour] = hourly_activity.get(hour, 0) + 1
        
        # Find peak hours
        peak_hour = max(hourly_activity.items(), key=lambda x: x[1])[0] if hourly_activity else 0
        
        # Group by day of week
        daily_activity = {}
        for interaction in interactions:
            day = interaction.timestamp.weekday()
            daily_activity[day] = daily_activity.get(day, 0) + 1
        
        return {
            "peak_hour": peak_hour,
            "hourly_distribution": hourly_activity,
            "daily_distribution": daily_activity,
            "most_active_day": max(daily_activity.items(), key=lambda x: x[1])[0] if daily_activity else 0
        }
    
    def _analyze_interaction_sequences(self, interactions: List[UserInteraction]) -> Dict[str, Any]:
        """Analyze common sequences of user interactions."""
        if len(interactions) < 2:
            return {}
        
        # Build sequence patterns
        sequences = []
        for i in range(len(interactions) - 1):
            sequence = (interactions[i].interaction_type, interactions[i+1].interaction_type)
            sequences.append(sequence)
        
        # Count sequence frequencies
        sequence_counts = {}
        for sequence in sequences:
            sequence_counts[sequence] = sequence_counts.get(sequence, 0) + 1
        
        # Find most common sequences
        most_common = sorted(sequence_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "common_sequences": most_common,
            "total_sequences": len(sequences),
            "unique_sequences": len(sequence_counts)
        }
    
    def _analyze_error_patterns(self, interactions: List[UserInteraction]) -> Dict[str, Any]:
        """Analyze error patterns in user interactions."""
        errors = [i for i in interactions if not i.success]
        
        if not errors:
            return {"error_rate": 0.0}
        
        error_types = {}
        for error in errors:
            error_type = error.interaction_type
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Calculate error clustering (consecutive errors)
        consecutive_errors = 0
        max_consecutive = 0
        current_consecutive = 0
        
        for interaction in interactions:
            if not interaction.success:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                if current_consecutive > 1:
                    consecutive_errors += 1
                current_consecutive = 0
        
        return {
            "error_rate": len(errors) / len(interactions),
            "error_types": error_types,
            "consecutive_error_episodes": consecutive_errors,
            "max_consecutive_errors": max_consecutive,
            "high_undo_rate": sum(1 for i in interactions 
                                if i.interaction_type == InteractionType.UNDO_REDO) > len(interactions) * 0.15
        }
    
    def _analyze_efficiency_patterns(self, interactions: List[UserInteraction]) -> Dict[str, Any]:
        """Analyze efficiency patterns in user behavior."""
        if not interactions:
            return {}
        
        # Calculate interaction durations
        durations = [i.duration_ms for i in interactions if i.duration_ms > 0]
        
        # Analyze task-specific efficiency
        task_efficiency = {}
        for interaction_type in InteractionType:
            type_interactions = [i for i in interactions if i.interaction_type == interaction_type]
            if type_interactions:
                avg_duration = np.mean([i.duration_ms for i in type_interactions])
                success_rate = sum(1 for i in type_interactions if i.success) / len(type_interactions)
                
                # Efficiency score combines speed and success
                efficiency = success_rate / (1 + avg_duration / 1000)  # Normalize by seconds
                task_efficiency[interaction_type.value] = efficiency
        
        return {
            "avg_interaction_duration_ms": np.mean(durations) if durations else 0,
            "task_efficiency_scores": task_efficiency,
            "slow_component_selection": task_efficiency.get("component_selection", 1.0) < 0.1,
            "slow_wire_routing": task_efficiency.get("wire_routing", 1.0) < 0.1
        }
    
    def _analyze_preference_patterns(self, interactions: List[UserInteraction]) -> Dict[str, Any]:
        """Analyze user preference patterns."""
        preferences = {}
        
        # Component preferences
        component_interactions = [i for i in interactions 
                                if i.target_object_type == "component"]
        component_usage = {}
        for interaction in component_interactions:
            comp_id = interaction.target_object_id
            if comp_id:
                component_usage[comp_id] = component_usage.get(comp_id, 0) + 1
        
        preferences["favorite_components"] = sorted(
            component_usage.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # View preferences
        view_changes = [i for i in interactions 
                       if i.interaction_type == InteractionType.VIEW_CHANGE]
        if view_changes:
            preferences["frequent_view_changes"] = len(view_changes) > len(interactions) * 0.1
        
        # Manual vs automated preferences
        manual_adjustments = sum(1 for i in interactions 
                               if i.interaction_type == InteractionType.MANUAL_ADJUSTMENT)
        total_placements = sum(1 for i in interactions 
                             if i.interaction_type in [InteractionType.COMPONENT_PLACEMENT, 
                                                     InteractionType.WIRE_ROUTING])
        
        if total_placements > 0:
            manual_ratio = manual_adjustments / total_placements
            preferences["prefers_manual_control"] = manual_ratio > 0.3
        
        return preferences
    
    def _process_interaction_buffer(self):
        """Process accumulated interactions for batch analysis."""
        logger.info(f"Processing {len(self.interactions_buffer)} interactions")
        
        # Perform batch clustering and pattern analysis
        self._update_interaction_clusters()
        
        # Clear processed interactions (keep recent ones)
        cutoff_time = datetime.now() - timedelta(hours=self.interaction_window_hours)
        self.interactions_buffer = [
            i for i in self.interactions_buffer if i.timestamp > cutoff_time
        ]
    
    def _update_interaction_clusters(self):
        """Update interaction pattern clusters."""
        if len(self.interactions_buffer) < 10:
            return
        
        # Prepare feature matrix for clustering
        features = []
        for interaction in self.interactions_buffer:
            feature_vector = [
                interaction.interaction_type.value.__hash__() % 1000,
                interaction.duration_ms,
                1 if interaction.success else 0,
                interaction.timestamp.hour,
                interaction.timestamp.weekday()
            ]
            features.append(feature_vector)
        
        features = np.array(features)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform clustering
        self.interaction_clusterer = DBSCAN(eps=0.5, min_samples=5)
        clusters = self.interaction_clusterer.fit_predict(features_scaled)
        
        # Analyze clusters for insights
        unique_clusters = set(clusters)
        logger.info(f"Identified {len(unique_clusters)} interaction patterns")