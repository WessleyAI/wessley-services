"""
Text chunking and preprocessing utilities for embedding generation.
"""
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import hashlib

from ..core.schemas import TextSpan, Component, Net


class ChunkType(Enum):
    """Types of text chunks."""
    TEXT_SPAN = "text_span"
    COMPONENT_DESCRIPTION = "component_description"
    NET_DESCRIPTION = "net_description"
    TECHNICAL_SUMMARY = "technical_summary"
    PAGE_SUMMARY = "page_summary"


@dataclass
class ProcessedChunk:
    """Processed text chunk ready for embedding."""
    id: str
    text: str
    chunk_type: ChunkType
    project_id: str
    page: int
    metadata: Dict[str, Any]
    word_count: int
    char_count: int
    technical_terms: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'text': self.text,
            'chunk_type': self.chunk_type.value,
            'project_id': self.project_id,
            'page': self.page,
            'metadata': self.metadata,
            'word_count': self.word_count,
            'char_count': self.char_count,
            'technical_terms': self.technical_terms
        }


class TextChunker:
    """
    Advanced text chunking and preprocessing for electrical schematic documents.
    """
    
    def __init__(
        self,
        max_chunk_size: int = 512,
        overlap_size: int = 50,
        min_chunk_size: int = 50
    ):
        """
        Initialize text chunker.
        
        Args:
            max_chunk_size: Maximum characters per chunk
            overlap_size: Character overlap between chunks
            min_chunk_size: Minimum characters per chunk
        """
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
        self.logger = logging.getLogger(__name__)
        
        # Technical terms dictionary for electrical schematics
        self.technical_terms = {
            'components': {
                'resistor', 'capacitor', 'inductor', 'diode', 'transistor', 'ic', 'integrated circuit',
                'mosfet', 'bjt', 'opamp', 'operational amplifier', 'relay', 'fuse', 'connector',
                'switch', 'led', 'zener', 'photodiode', 'thermistor', 'varistor', 'crystal',
                'oscillator', 'transformer', 'choke', 'ferrite bead'
            },
            'electrical': {
                'voltage', 'current', 'resistance', 'capacitance', 'inductance', 'impedance',
                'frequency', 'power', 'ground', 'vcc', 'vdd', 'vss', 'gnd', 'supply', 'signal',
                'digital', 'analog', 'pwm', 'spi', 'i2c', 'uart', 'clock', 'reset', 'enable',
                'drain', 'source', 'gate', 'base', 'collector', 'emitter', 'anode', 'cathode'
            },
            'units': {
                'ohm', 'kohm', 'mohm', 'farad', 'microfarad', 'nanofarad', 'picofarad',
                'henry', 'millihenry', 'microhenry', 'volt', 'millivolt', 'amp', 'milliamp',
                'microamp', 'watt', 'milliwatt', 'hertz', 'kilohertz', 'megahertz', 'gigahertz'
            },
            'values': {
                'k', 'kohm', 'm', 'mohm', 'pf', 'nf', 'uf', 'mh', 'uh', 'nh', 'ma', 'ua',
                'mv', 'v', 'hz', 'khz', 'mhz', 'ghz', 'db', 'dbm'
            }
        }
        
        # Flatten technical terms for easy lookup
        self.all_technical_terms = set()
        for category in self.technical_terms.values():
            self.all_technical_terms.update(category)
    
    def chunk_text_spans(
        self,
        project_id: str,
        text_spans: List[TextSpan]
    ) -> List[ProcessedChunk]:
        """
        Create chunks from OCR text spans with intelligent grouping.
        
        Args:
            project_id: Project identifier
            text_spans: List of text spans from OCR
            
        Returns:
            List of processed chunks
        """
        chunks = []
        
        # Group text spans by page
        pages = {}
        for span in text_spans:
            if span.page not in pages:
                pages[span.page] = []
            pages[span.page].append(span)
        
        # Process each page
        for page_num, page_spans in pages.items():
            # Sort spans by position (top to bottom, left to right)
            sorted_spans = sorted(page_spans, key=lambda s: (s.bbox[1], s.bbox[0]) if s.bbox else (0, 0))
            
            # Group nearby spans together
            grouped_spans = self._group_nearby_spans(sorted_spans)
            
            # Create chunks from groups
            for group_idx, span_group in enumerate(grouped_spans):
                combined_text = self._combine_span_texts(span_group)
                
                if len(combined_text) < self.min_chunk_size:
                    continue
                
                # Split into smaller chunks if needed
                text_chunks = self._split_text_intelligently(combined_text)
                
                for chunk_idx, chunk_text in enumerate(text_chunks):
                    chunk_id = self._generate_chunk_id(project_id, 'text', page_num, group_idx, chunk_idx)
                    
                    chunk = ProcessedChunk(
                        id=chunk_id,
                        text=chunk_text,
                        chunk_type=ChunkType.TEXT_SPAN,
                        project_id=project_id,
                        page=page_num,
                        metadata={
                            'span_count': len(span_group),
                            'avg_confidence': sum(s.confidence for s in span_group) / len(span_group),
                            'bbox': self._calculate_group_bbox(span_group),
                            'engines': list(set(s.engine.value if hasattr(s.engine, 'value') else str(s.engine) for s in span_group))
                        },
                        word_count=len(chunk_text.split()),
                        char_count=len(chunk_text),
                        technical_terms=self._extract_technical_terms(chunk_text)
                    )
                    chunks.append(chunk)
        
        return chunks
    
    def chunk_component_descriptions(
        self,
        project_id: str,
        components: List[Component]
    ) -> List[ProcessedChunk]:
        """
        Create searchable chunks from component descriptions.
        
        Args:
            project_id: Project identifier
            components: List of detected components
            
        Returns:
            List of processed chunks
        """
        chunks = []
        
        for comp_idx, component in enumerate(components):
            # Generate rich component description
            description = self._generate_component_description(component)
            
            chunk_id = self._generate_chunk_id(project_id, 'component', component.page, comp_idx)
            
            chunk = ProcessedChunk(
                id=chunk_id,
                text=description,
                chunk_type=ChunkType.COMPONENT_DESCRIPTION,
                project_id=project_id,
                page=component.page,
                metadata={
                    'component_id': component.id,
                    'component_type': component.type.value if hasattr(component.type, 'value') else str(component.type),
                    'component_value': component.value,
                    'bbox': component.bbox,
                    'confidence': component.confidence,
                    'pin_count': len(component.pins) if component.pins else 0,
                    'has_value': bool(component.value),
                    'provenance': component.provenance
                },
                word_count=len(description.split()),
                char_count=len(description),
                technical_terms=self._extract_technical_terms(description)
            )
            chunks.append(chunk)
        
        return chunks
    
    def chunk_net_descriptions(
        self,
        project_id: str,
        nets: List[Net]
    ) -> List[ProcessedChunk]:
        """
        Create searchable chunks from net descriptions.
        
        Args:
            project_id: Project identifier
            nets: List of electrical nets
            
        Returns:
            List of processed chunks
        """
        chunks = []
        
        for net_idx, net in enumerate(nets):
            # Generate rich net description
            description = self._generate_net_description(net)
            
            chunk_id = self._generate_chunk_id(project_id, 'net', net.page_spans[0] if net.page_spans else 1, net_idx)
            
            chunk = ProcessedChunk(
                id=chunk_id,
                text=description,
                chunk_type=ChunkType.NET_DESCRIPTION,
                project_id=project_id,
                page=net.page_spans[0] if net.page_spans else 1,
                metadata={
                    'net_name': net.name,
                    'connection_count': len(net.connections),
                    'page_spans': net.page_spans,
                    'confidence': net.confidence,
                    'component_ids': [conn.component_id for conn in net.connections],
                    'pin_connections': [(conn.component_id, conn.pin) for conn in net.connections],
                    'is_power_net': self._is_power_net(net.name),
                    'is_signal_net': self._is_signal_net(net.name)
                },
                word_count=len(description.split()),
                char_count=len(description),
                technical_terms=self._extract_technical_terms(description)
            )
            chunks.append(chunk)
        
        return chunks
    
    def create_page_summaries(
        self,
        project_id: str,
        page_chunks: Dict[int, List[ProcessedChunk]]
    ) -> List[ProcessedChunk]:
        """
        Create high-level page summaries from existing chunks.
        
        Args:
            project_id: Project identifier
            page_chunks: Chunks grouped by page number
            
        Returns:
            List of page summary chunks
        """
        summaries = []
        
        for page_num, chunks in page_chunks.items():
            # Aggregate information from all chunks on the page
            component_types = set()
            net_names = set()
            technical_terms = set()
            total_components = 0
            total_nets = 0
            
            for chunk in chunks:
                if chunk.chunk_type == ChunkType.COMPONENT_DESCRIPTION:
                    component_types.add(chunk.metadata.get('component_type', 'unknown'))
                    total_components += 1
                elif chunk.chunk_type == ChunkType.NET_DESCRIPTION:
                    net_names.add(chunk.metadata.get('net_name', 'unknown'))
                    total_nets += 1
                
                technical_terms.update(chunk.technical_terms)
            
            # Generate summary text
            summary_parts = [
                f"Page {page_num} schematic analysis summary.",
                f"Contains {total_components} components and {total_nets} electrical nets."
            ]
            
            if component_types:
                summary_parts.append(f"Component types: {', '.join(sorted(component_types))}.")
            
            if net_names:
                main_nets = [name for name in net_names if any(term in name.lower() for term in ['vcc', 'gnd', 'power', 'supply'])]
                if main_nets:
                    summary_parts.append(f"Power nets: {', '.join(main_nets)}.")
            
            if technical_terms:
                key_terms = sorted(list(technical_terms))[:10]  # Top 10 terms
                summary_parts.append(f"Key technical terms: {', '.join(key_terms)}.")
            
            summary_text = " ".join(summary_parts)
            
            chunk_id = self._generate_chunk_id(project_id, 'page_summary', page_num, 0)
            
            summary_chunk = ProcessedChunk(
                id=chunk_id,
                text=summary_text,
                chunk_type=ChunkType.PAGE_SUMMARY,
                project_id=project_id,
                page=page_num,
                metadata={
                    'component_count': total_components,
                    'net_count': total_nets,
                    'component_types': list(component_types),
                    'net_names': list(net_names),
                    'chunk_count': len(chunks)
                },
                word_count=len(summary_text.split()),
                char_count=len(summary_text),
                technical_terms=list(technical_terms)[:20]  # Top 20 terms
            )
            summaries.append(summary_chunk)
        
        return summaries
    
    def _group_nearby_spans(self, spans: List[TextSpan], proximity_threshold: float = 30.0) -> List[List[TextSpan]]:
        """Group text spans that are spatially close."""
        if not spans:
            return []
        
        groups = []
        current_group = [spans[0]]
        
        for i in range(1, len(spans)):
            current_span = spans[i]
            prev_span = spans[i-1]
            
            # Calculate distance between spans
            if current_span.bbox and prev_span.bbox:
                # Distance from bottom of previous to top of current
                vertical_distance = current_span.bbox[1] - prev_span.bbox[3]
                
                # Horizontal overlap check
                horizontal_overlap = not (
                    current_span.bbox[2] < prev_span.bbox[0] or 
                    current_span.bbox[0] > prev_span.bbox[2]
                )
                
                if vertical_distance < proximity_threshold and (horizontal_overlap or vertical_distance < 10):
                    current_group.append(current_span)
                else:
                    groups.append(current_group)
                    current_group = [current_span]
            else:
                current_group.append(current_span)
        
        groups.append(current_group)
        return groups
    
    def _combine_span_texts(self, spans: List[TextSpan]) -> str:
        """Combine text from multiple spans intelligently."""
        if not spans:
            return ""
        
        # Sort spans by reading order (top to bottom, left to right)
        sorted_spans = sorted(spans, key=lambda s: (s.bbox[1], s.bbox[0]) if s.bbox else (0, 0))
        
        combined_parts = []
        for span in sorted_spans:
            text = span.text.strip()
            if text:
                combined_parts.append(text)
        
        # Join with appropriate separators
        return " ".join(combined_parts)
    
    def _split_text_intelligently(self, text: str) -> List[str]:
        """Split text into chunks at natural boundaries."""
        if len(text) <= self.max_chunk_size:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by sentences first
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding this sentence would exceed max size, start new chunk
            if len(current_chunk) + len(sentence) + 1 > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Sentence is too long, split by words
                    word_chunks = self._split_by_words(sentence)
                    chunks.extend(word_chunks)
            else:
                current_chunk += (" " + sentence if current_chunk else sentence)
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_by_words(self, text: str) -> List[str]:
        """Split text by words when sentences are too long."""
        words = text.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            if len(current_chunk) + len(word) + 1 > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = word
                else:
                    # Single word is too long, just add it
                    chunks.append(word)
            else:
                current_chunk += (" " + word if current_chunk else word)
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _generate_component_description(self, component: Component) -> str:
        """Generate rich description for a component."""
        parts = []
        
        # Basic identification
        parts.append(f"Electronic component {component.id}")
        
        # Type and value
        comp_type = component.type.value if hasattr(component.type, 'value') else str(component.type)
        parts.append(f"is a {comp_type}")
        
        if component.value:
            parts.append(f"with value {component.value}")
        
        # Pin information
        if component.pins:
            pin_names = [pin.name for pin in component.pins if pin.name]
            if pin_names:
                parts.append(f"having pins {', '.join(pin_names)}")
            else:
                parts.append(f"with {len(component.pins)} pins")
        
        # Location
        parts.append(f"located on page {component.page}")
        
        # Confidence
        confidence_desc = "high" if component.confidence > 0.8 else "medium" if component.confidence > 0.5 else "low"
        parts.append(f"detected with {confidence_desc} confidence ({component.confidence:.2f})")
        
        # Physical description based on type
        if comp_type in ['resistor', 'capacitor', 'inductor']:
            parts.append(f"This {comp_type} is a passive electronic component")
            if comp_type == 'resistor':
                parts.append("that resists electrical current flow")
            elif comp_type == 'capacitor':
                parts.append("that stores electrical energy in an electric field")
            elif comp_type == 'inductor':
                parts.append("that stores energy in a magnetic field")
        
        return ". ".join(parts) + "."
    
    def _generate_net_description(self, net: Net) -> str:
        """Generate rich description for an electrical net."""
        parts = []
        
        # Basic identification
        parts.append(f"Electrical net named '{net.name}'")
        
        # Connection information
        if net.connections:
            parts.append(f"connects {len(net.connections)} component pins")
            
            # List some connections
            if len(net.connections) <= 3:
                conn_desc = []
                for conn in net.connections:
                    conn_desc.append(f"{conn.component_id} pin {conn.pin}")
                parts.append(f"specifically {', '.join(conn_desc)}")
            else:
                parts.append(f"including {net.connections[0].component_id} pin {net.connections[0].pin} and others")
        
        # Page information
        if net.page_spans:
            if len(net.page_spans) == 1:
                parts.append(f"appears on page {net.page_spans[0]}")
            else:
                parts.append(f"spans across pages {', '.join(map(str, net.page_spans))}")
        
        # Net type classification
        net_name_lower = net.name.lower()
        if any(term in net_name_lower for term in ['vcc', 'vdd', 'power', '+', 'supply']):
            parts.append("This is a power supply net providing electrical power")
        elif any(term in net_name_lower for term in ['gnd', 'ground', 'vss', '-']):
            parts.append("This is a ground net providing electrical reference")
        elif any(term in net_name_lower for term in ['clk', 'clock']):
            parts.append("This is a clock signal net for timing")
        elif any(term in net_name_lower for term in ['data', 'signal']):
            parts.append("This is a signal net for data transmission")
        else:
            parts.append("This net carries electrical signals between components")
        
        # Confidence
        confidence_desc = "high" if net.confidence > 0.8 else "medium" if net.confidence > 0.5 else "low"
        parts.append(f"detected with {confidence_desc} confidence ({net.confidence:.2f})")
        
        return ". ".join(parts) + "."
    
    def _calculate_group_bbox(self, spans: List[TextSpan]) -> List[float]:
        """Calculate bounding box that encompasses all spans in group."""
        if not spans or not any(s.bbox for s in spans):
            return [0, 0, 0, 0]
        
        valid_bboxes = [s.bbox for s in spans if s.bbox]
        if not valid_bboxes:
            return [0, 0, 0, 0]
        
        min_x = min(bbox[0] for bbox in valid_bboxes)
        min_y = min(bbox[1] for bbox in valid_bboxes)
        max_x = max(bbox[2] for bbox in valid_bboxes)
        max_y = max(bbox[3] for bbox in valid_bboxes)
        
        return [min_x, min_y, max_x, max_y]
    
    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms from text."""
        words = re.findall(r'\b\w+\b', text.lower())
        technical_terms = []
        
        for word in words:
            if word in self.all_technical_terms:
                technical_terms.append(word)
            # Check for component values (e.g., "10k", "100nF")
            elif re.match(r'^\d+[a-z]*$', word):
                technical_terms.append(word)
        
        return list(set(technical_terms))  # Remove duplicates
    
    def _is_power_net(self, net_name: str) -> bool:
        """Check if net is a power net."""
        power_indicators = ['vcc', 'vdd', 'power', '+', 'supply', 'pwr', 'bat', 'vin', 'vout']
        return any(indicator in net_name.lower() for indicator in power_indicators)
    
    def _is_signal_net(self, net_name: str) -> bool:
        """Check if net is a signal net."""
        signal_indicators = ['clk', 'clock', 'data', 'signal', 'sda', 'scl', 'rx', 'tx', 'cs', 'en']
        return any(indicator in net_name.lower() for indicator in signal_indicators)
    
    def _generate_chunk_id(self, project_id: str, chunk_type: str, page: int, *indices) -> str:
        """Generate unique chunk ID."""
        id_parts = [project_id, chunk_type, str(page)] + [str(i) for i in indices]
        id_string = "_".join(id_parts)
        
        # Create hash for uniqueness while keeping readable prefix
        hash_suffix = hashlib.md5(id_string.encode()).hexdigest()[:8]
        return f"{chunk_type}_{page}_{hash_suffix}"


# Convenience function
def create_text_chunker(**kwargs) -> TextChunker:
    """Create text chunker with default settings."""
    return TextChunker(**kwargs)