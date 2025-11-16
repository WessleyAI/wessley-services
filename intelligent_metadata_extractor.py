#!/usr/bin/env python3
"""
Intelligent Metadata Extractor - LLM-Driven Adaptive System

This system:
1. Reads pages from any technical manual
2. Uses LLM to understand what type of content it contains
3. Extracts structured knowledge dynamically
4. Routes to appropriate storage (Neo4j Metadata, Knowledge Graph, Qdrant Vectors)

Works for ANY manual - no hardcoded patterns!

Usage:
    python3 intelligent_metadata_extractor.py \\
        --pdf manual.pdf \\
        --pages 1-15 \\
        --model ollama \\
        --store hybrid
"""

import os
import json
import logging
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from ollama import chat as ollama_chat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# JSON Repair Utilities
# ============================================================================

def repair_json(json_str: str) -> str:
    """
    Attempt to repair common JSON formatting issues from LLM output

    - Fix missing closing braces/brackets
    - Remove trailing commas
    - Fix invalid escape sequences
    - Handle unicode issues
    - Fix nested quotes in strings
    """
    # Remove invalid unicode escape sequences (like \u211 or \u21)
    json_str = re.sub(r'\\u[0-9a-fA-F]{0,3}(?![0-9a-fA-F])', '', json_str)

    # Fix nested JSON in strings like "symbol": "{"a"}"  → "symbol": "{a}"
    json_str = re.sub(r'"\{\\?"([^"]*?)\\?"\}"', r'"{\\1}"', json_str)

    # Count braces and brackets
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    open_brackets = json_str.count('[')
    close_brackets = json_str.count(']')

    # Add missing closing characters at the end
    if open_braces > close_braces:
        json_str += '}' * (open_braces - close_braces)
    if open_brackets > close_brackets:
        json_str += ']' * (open_brackets - close_brackets)

    # Remove trailing commas before closing braces/brackets
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

    return json_str


# ============================================================================
# LLM-Based Content Analyzer
# ============================================================================

class FailureLogger:
    """Logs parsing failures with LLM self-reflection and improvement suggestions"""

    def __init__(self, output_dir: str = "extraction_failures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.failures = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.failure_file = self.output_dir / f"failures_{timestamp}.json"
        self.summary_file = self.output_dir / f"failure_summary_{timestamp}.md"
        logger.info(f"📋 Failure logger initialized: {self.failure_file}")

    def log_failure(self, failure_data: Dict[str, Any]):
        """Add a failure to the log"""
        self.failures.append(failure_data)

        # Append to JSON file immediately
        with open(self.failure_file, 'w') as f:
            json.dump(self.failures, f, indent=2)

    def generate_summary(self):
        """Generate markdown summary of failures and improvements"""
        if not self.failures:
            return

        summary = f"""# Extraction Failure Analysis Report
Generated: {datetime.now().isoformat()}

## Summary
- Total Failures: {len(self.failures)}
- Pages Affected: {', '.join(str(f['page']) for f in self.failures)}

## Failure Details

"""
        for fail in self.failures:
            summary += f"""### Page {fail['page']} - {fail['stage']}

**Error:** {fail['error']}

**What the page contains:**
{fail.get('apparent_data', 'N/A')}

**What we tried to do:**
{fail.get('attempted_task', 'N/A')}

**LLM Suggestions:**
{fail.get('improvement_suggestions', 'N/A')}

**Raw Response (first 500 chars):**
```
{fail.get('raw_response', 'N/A')[:500]}
```

---

"""

        with open(self.summary_file, 'w') as f:
            f.write(summary)

        logger.info(f"📊 Failure summary generated: {self.summary_file}")


class ContentAnalyzer:
    """Uses LLM to understand and extract knowledge from document pages"""

    def __init__(self, model: str = "llama3.1", failure_logger: Optional[FailureLogger] = None):
        self.model = model
        self.failure_logger = failure_logger or FailureLogger()
        self.accumulated_context = {
            "symbol_maps": [],      # Symbol definitions from diagram legends
            "abbreviations": [],    # Abbreviation definitions
            "wire_colors": [],      # Wire color codes
            "component_codes": []   # Component abbreviations/codes
        }
        logger.info(f"🤖 Content Analyzer initialized with model: {model}")

    def add_to_context(self, extracted_data: Dict[str, Any], page_type: str):
        """
        Add extracted lookup tables to accumulated context for future pages
        """
        added_count = 0

        if 'entries' in extracted_data:
            for entry in extracted_data['entries']:
                # Classify the type of entry
                if 'color' in str(entry).lower() or entry.get('type') == 'wire_color':
                    self.accumulated_context['wire_colors'].append(entry)
                    added_count += 1
                elif entry.get('type') == 'component' or 'component' in str(entry).lower():
                    self.accumulated_context['component_codes'].append(entry)
                    added_count += 1
                elif 'code' in entry or 'abbreviation' in entry or entry.get('type') == 'abbreviation':
                    self.accumulated_context['abbreviations'].append(entry)
                    added_count += 1

        if 'symbols' in extracted_data:
            self.accumulated_context['symbol_maps'].extend(extracted_data['symbols'])
            added_count += len(extracted_data['symbols'])

        if added_count > 0:
            total_context = sum(len(v) for v in self.accumulated_context.values())
            logger.info(f"   ✓ Added {added_count} items to context (total: {total_context})")

    def get_context_summary(self) -> str:
        """
        Generate a summary of accumulated context for inclusion in prompts
        """
        if not any(self.accumulated_context.values()):
            return ""

        summary = "\n<document_context>\nPreviously extracted reference data:\n\n"

        if self.accumulated_context['symbol_maps']:
            summary += "Symbol Definitions:\n"
            for sym in self.accumulated_context['symbol_maps'][:20]:  # Limit to 20
                summary += f"  - {sym.get('symbol', '?')}: {sym.get('meaning', '?')}\n"

        if self.accumulated_context['abbreviations']:
            summary += "\nAbbreviations:\n"
            for abbr in self.accumulated_context['abbreviations'][:20]:
                summary += f"  - {abbr.get('code', '?')}: {abbr.get('meaning', '?')}\n"

        if self.accumulated_context['wire_colors']:
            summary += "\nWire Colors:\n"
            for color in self.accumulated_context['wire_colors'][:10]:
                summary += f"  - {color.get('code', '?')}: {color.get('meaning', '?')}\n"

        if self.accumulated_context['component_codes']:
            summary += "\nComponent Codes:\n"
            for comp in self.accumulated_context['component_codes'][:20]:
                summary += f"  - {comp.get('code', '?')}: {comp.get('meaning', '?')}\n"

        summary += "</document_context>\n"
        return summary

    def analyze_failure(self, text: str, page_num: int, stage: str, error: str, raw_response: str) -> Dict[str, Any]:
        """
        Ask LLM to reflect on parsing failure and suggest improvements

        Returns self-analysis with:
        - apparent_data: What data is visible on the page
        - attempted_task: What we were trying to extract
        - why_failed: Why the parsing failed
        - improvement_suggestions: How to improve the approach
        """
        reflection_prompt = f"""You are analyzing a FAILED extraction attempt on page {page_num}.

**Stage that failed:** {stage}
**Error message:** {error}

**Original page text:**
<page_text>
{text[:2000]}
</page_text>

**Raw LLM response that failed to parse:**
<raw_response>
{raw_response[:1000]}
</raw_response>

Please analyze this failure and provide:

1. **What data is actually visible on this page?** (describe the content you can see)
2. **What were we trying to extract?** (based on the stage and prompts)
3. **Why did the JSON parsing fail?** (analyze the raw response)
4. **How can we improve the extraction approach?** (specific suggestions)

Respond with JSON:
{{
    "apparent_data": "detailed description of what's on the page",
    "attempted_task": "what we were trying to extract",
    "why_failed": "root cause of the JSON parse failure",
    "improvement_suggestions": [
        "specific suggestion 1",
        "specific suggestion 2"
    ],
    "alternative_approach": "a different way to extract this data"
}}

Respond ONLY with valid JSON."""

        try:
            response = ollama_chat(
                model=self.model,
                messages=[{"role": "user", "content": reflection_prompt}]
            )

            result_text = response['message']['content']

            # Extract JSON
            json_match = result_text.strip()
            if json_match.startswith('```json'):
                json_match = json_match.split('```json')[1].split('```')[0]
            elif json_match.startswith('```'):
                json_match = json_match.split('```')[1].split('```')[0]

            analysis = json.loads(json_match)
            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze failure for page {page_num}: {e}")
            return {
                "apparent_data": "Could not analyze",
                "attempted_task": stage,
                "why_failed": f"Reflection failed: {e}",
                "improvement_suggestions": ["Manual review needed"],
                "alternative_approach": "Human intervention required"
            }

    def analyze_page_type(self, text: str, page_num: int) -> Dict[str, Any]:
        """
        Ask LLM: What type of content is on this page?

        Returns:
            {
                "page_type": "table_of_contents" | "lookup_table" | "instructions" |
                            "diagram" | "specifications" | "unknown",
                "content_category": "metadata" | "knowledge" | "reference" | "procedural",
                "storage_tier": 1 | 2 | 3 | 4,
                "confidence": 0.0-1.0,
                "reasoning": "explanation"
            }
        """
        context_summary = self.get_context_summary()

        prompt = f"""You are analyzing page {page_num} of a technical manual.
{context_summary}
<page_text>
{text[:2000]}
</page_text>

Classify this page and decide how to store its knowledge.

Respond with JSON:
{{
    "page_type": "table_of_contents | lookup_table | instructions | diagram_legend | specifications | circuit_diagram | unknown",
    "content_category": "metadata | knowledge | reference | procedural",
    "storage_tier": 1,
    "storage_reasoning": "explanation of where this should be stored",
    "contains": ["list of what this page contains"],
    "confidence": 0.95
}}

Storage Tiers:
- Tier 1 (Metadata): Lookup tables, codes, abbreviations that validate other data
- Tier 2 (Knowledge Graph): Rules, specifications, relationships between concepts
- Tier 3 (Document Structure): Table of contents, section hierarchy, navigation
- Tier 4 (Semantic Search): Long-form explanations, procedures, how-to content

Examples:
- Wire color codes → Tier 1 (Metadata lookup table)
- "All ground wires must be 6mm²" → Tier 2 (Knowledge rule)
- Table of Contents → Tier 3 (Document structure)
- "How to troubleshoot starter circuit" → Tier 4 (Semantic searchable content)

Respond ONLY with valid JSON, no other text."""

        try:
            response = ollama_chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )

            result_text = response['message']['content']

            # Extract JSON from response
            json_match = result_text.strip()
            if json_match.startswith('```json'):
                json_match = json_match.split('```json')[1].split('```')[0]
            elif json_match.startswith('```'):
                json_match = json_match.split('```')[1].split('```')[0]

            # Repair JSON before parsing
            json_match = repair_json(json_match)

            result = json.loads(json_match)
            logger.debug(f"Page {page_num} classified as: {result['page_type']} (Tier {result['storage_tier']})")
            return result

        except Exception as e:
            logger.error(f"Failed to analyze page {page_num}: {e}")
            logger.warning(f"   🔍 Running failure analysis...")

            # Run self-reflection on failure
            failure_analysis = self.analyze_failure(
                text=text,
                page_num=page_num,
                stage="page_type_classification",
                error=str(e),
                raw_response=result_text if 'result_text' in locals() else "No response captured"
            )

            # Log the failure
            self.failure_logger.log_failure({
                "page": page_num,
                "stage": "page_type_classification",
                "error": str(e),
                "raw_response": result_text if 'result_text' in locals() else "No response",
                **failure_analysis
            })

            logger.info(f"   📋 Failure logged. Suggestions: {failure_analysis.get('improvement_suggestions', [])}")

            return {
                "page_type": "unknown",
                "content_category": "unknown",
                "storage_tier": 4,  # Default to semantic search
                "storage_reasoning": f"Error: {e}",
                "contains": [],
                "confidence": 0.0
            }

    def extract_structured_data(self, text: str, page_type: str, page_num: int) -> Dict[str, Any]:
        """
        Ask LLM: Extract structured data from this page based on its type

        Returns:
            Dynamic structure based on page_type:
            - lookup_table: {"entries": [{"code": "R", "meaning": "Red"}, ...]}
            - table_of_contents: {"sections": [{"title": "...", "page": 16}, ...]}
            - instructions: {"steps": [...], "warnings": [...]}
            - specifications: {"specs": [{"parameter": "voltage", "value": "12V"}, ...]}
        """
        context_summary = self.get_context_summary()

        prompt = f"""You are extracting structured data from page {page_num} of a technical manual.

Page Type: {page_type}
{context_summary}
<page_text>
{text[:3000]}
</page_text>

IMPORTANT: Extract ONLY the format that matches the page type "{page_type}".
DO NOT return multiple formats. Return ONLY ONE structure that matches this page type.
Use the document context above to interpret symbols, abbreviations, and codes in the text.

Respond with JSON in the appropriate format for THIS SPECIFIC page type:

For lookup_table (wire colors, abbreviations, symbols):
{{
    "entries": [
        {{"code": "R", "meaning": "Red", "usage": "Power/Positive"}},
        {{"code": "ALT", "meaning": "Alternator", "type": "component"}}
    ]
}}

For table_of_contents:
{{
    "sections": [
        {{"title": "Starter Circuit", "start_page": 16, "category": "electrical"}},
        {{"title": "Lighting System", "start_page": 46, "category": "lighting"}}
    ]
}}

For instructions:
{{
    "title": "How to Read Circuit Diagrams",
    "steps": ["step 1", "step 2"],
    "rules": ["rule 1", "rule 2"],
    "warnings": ["warning text"]
}}

For specifications:
{{
    "specs": [
        {{"parameter": "wire_gauge", "value": "6mm²", "applies_to": "ground_wires", "requirement": "minimum"}},
        {{"parameter": "voltage", "value": "12V", "context": "battery nominal voltage"}}
    ]
}}

For diagram_legend:
{{
    "symbols": [
        {{"symbol": "circle with X", "meaning": "ground point"}},
        {{"symbol": "zigzag line", "meaning": "resistor"}}
    ],
    "conventions": ["convention descriptions"]
}}

IMPORTANT:
- Extract EVERYTHING you can find
- Be thorough - this is reference data
- Maintain exact codes/values from the text
- Include context where helpful

Respond ONLY with valid JSON, no other text."""

        try:
            response = ollama_chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )

            result_text = response['message']['content']

            # Extract JSON
            json_match = result_text.strip()
            if json_match.startswith('```json'):
                json_match = json_match.split('```json')[1].split('```')[0]
            elif json_match.startswith('```'):
                json_match = json_match.split('```')[1].split('```')[0]

            # Repair JSON before parsing
            json_match = repair_json(json_match)

            extracted_data = json.loads(json_match)

            # Handle both dict and list responses from LLM
            if isinstance(extracted_data, list):
                # LLM returned array directly - wrap it in expected structure
                logger.warning(f"   ⚠️  LLM returned list instead of object, wrapping as 'entries'")
                extracted_data = {"entries": extracted_data}

            # Handle nested structures - extract the correct nested object
            # Sometimes LLM returns {"lookup_table": {"entries": [...]}} instead of {"entries": [...]}
            if isinstance(extracted_data, dict):
                # Check if there's a nested structure matching page_type
                page_type_key = page_type.split(' | ')[0]  # Handle multi-type like "instructions | lookup_table"

                if page_type_key in extracted_data:
                    logger.info(f"   🔧 Extracting nested '{page_type_key}' structure")
                    extracted_data = extracted_data[page_type_key]
                elif 'lookup_table' in extracted_data and 'entries' not in extracted_data:
                    logger.info(f"   🔧 Extracting nested 'lookup_table' structure")
                    extracted_data = extracted_data['lookup_table']
                elif 'table_of_contents' in extracted_data and 'sections' not in extracted_data:
                    logger.info(f"   🔧 Extracting nested 'table_of_contents' structure")
                    extracted_data = extracted_data['table_of_contents']
                elif 'specifications' in extracted_data and 'specs' not in extracted_data:
                    logger.info(f"   🔧 Extracting nested 'specifications' structure")
                    extracted_data = extracted_data['specifications']

            # Count extracted items
            count = 0
            if isinstance(extracted_data, dict):
                if 'entries' in extracted_data:
                    count = len(extracted_data['entries'])
                elif 'sections' in extracted_data:
                    count = len(extracted_data['sections'])
                elif 'specs' in extracted_data:
                    count = len(extracted_data['specs'])
                elif 'symbols' in extracted_data:
                    count = len(extracted_data['symbols'])
                elif 'steps' in extracted_data:
                    count = len(extracted_data.get('steps', []))

            logger.info(f"   Extracted {count} items from page {page_num}")

            # Log the actual extracted data for visibility
            if count > 0:
                logger.info(f"   📦 Structured Data Preview:")
                if 'entries' in extracted_data:
                    for i, entry in enumerate(extracted_data['entries'][:3], 1):
                        logger.info(f"      {i}. {entry}")
                    if len(extracted_data['entries']) > 3:
                        logger.info(f"      ... and {len(extracted_data['entries']) - 3} more")
                elif 'sections' in extracted_data:
                    for i, section in enumerate(extracted_data['sections'][:3], 1):
                        logger.info(f"      {i}. {section}")
                    if len(extracted_data['sections']) > 3:
                        logger.info(f"      ... and {len(extracted_data['sections']) - 3} more")
                elif 'specs' in extracted_data:
                    for i, spec in enumerate(extracted_data['specs'][:3], 1):
                        logger.info(f"      {i}. {spec}")
                    if len(extracted_data['specs']) > 3:
                        logger.info(f"      ... and {len(extracted_data['specs']) - 3} more")

            return extracted_data

        except Exception as e:
            logger.error(f"Failed to extract data from page {page_num}: {e}")
            logger.warning(f"   🔍 Running failure analysis...")

            # Run self-reflection on failure
            failure_analysis = self.analyze_failure(
                text=text,
                page_num=page_num,
                stage=f"structured_data_extraction ({page_type})",
                error=str(e),
                raw_response=result_text if 'result_text' in locals() else "No response captured"
            )

            # Log the failure
            self.failure_logger.log_failure({
                "page": page_num,
                "stage": f"structured_data_extraction ({page_type})",
                "error": str(e),
                "raw_response": result_text if 'result_text' in locals() else "No response",
                **failure_analysis
            })

            logger.info(f"   📋 Failure logged. Suggestions: {failure_analysis.get('improvement_suggestions', [])}")

            return {"error": str(e), "raw_text": text[:500]}

    def extract_semantic_knowledge(self, text: str, page_num: int) -> List[str]:
        """
        Extract text chunks for semantic search (Tier 4)

        Ask LLM to break down text into meaningful, self-contained chunks
        """
        prompt = f"""You are preparing text from page {page_num} for semantic search.

<page_text>
{text[:4000]}
</page_text>

Break this text into self-contained chunks that would be useful for semantic search.
Each chunk should:
- Be 1-3 sentences
- Contain a complete thought
- Be useful for answering questions

Respond with JSON:
{{
    "chunks": [
        "The starter motor receives power from the battery through relay K1.",
        "When ignition is turned to START, relay K1 closes and sends 12V to starter solenoid.",
        "All ground wires in engine bay must be minimum 6mm² gauge."
    ]
}}

Respond ONLY with valid JSON, no other text."""

        try:
            response = ollama_chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )

            result_text = response['message']['content']

            # Extract JSON
            json_match = result_text.strip()
            if json_match.startswith('```json'):
                json_match = json_match.split('```json')[1].split('```')[0]
            elif json_match.startswith('```'):
                json_match = json_match.split('```')[1].split('```')[0]

            # Repair JSON before parsing
            json_match = repair_json(json_match)

            result = json.loads(json_match)

            # Handle both dict and list responses from LLM
            if isinstance(result, dict):
                chunks = result.get('chunks', [])
            elif isinstance(result, list):
                # LLM returned array directly
                chunks = result
            else:
                chunks = []

            logger.info(f"   Created {len(chunks)} semantic chunks from page {page_num}")

            # Log the actual semantic chunks for visibility
            if chunks:
                logger.info(f"   💬 Semantic Chunks Preview:")
                for i, chunk in enumerate(chunks[:3], 1):
                    preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
                    logger.info(f"      {i}. {preview}")
                if len(chunks) > 3:
                    logger.info(f"      ... and {len(chunks) - 3} more chunks")

            return chunks

        except Exception as e:
            logger.error(f"Failed to extract semantic chunks from page {page_num}: {e}")
            # Fallback: simple sentence splitting
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 30]
            return sentences[:10]  # Max 10 chunks as fallback


# ============================================================================
# Knowledge Router - Decides Where to Store Each Piece
# ============================================================================

class KnowledgeRouter:
    """Routes extracted knowledge to appropriate storage tiers"""

    def __init__(self):
        self.storage = {
            "tier_1_metadata": [],      # Lookup tables (Neo4j :Metadata nodes)
            "tier_2_knowledge": [],      # Knowledge graph (Neo4j :Knowledge nodes)
            "tier_3_structure": [],      # Document structure (Neo4j :Section nodes)
            "tier_4_semantic": []        # Semantic chunks (Qdrant vectors)
        }

    def route(self, page_analysis: Dict, structured_data: Dict,
             semantic_chunks: List[str], page_num: int):
        """
        Route extracted data to appropriate storage tier

        Args:
            page_analysis: Result from analyze_page_type()
            structured_data: Result from extract_structured_data()
            semantic_chunks: Result from extract_semantic_knowledge()
            page_num: Source page number
        """
        tier = page_analysis.get('storage_tier', 4)
        page_type = page_analysis.get('page_type', 'unknown')

        # Tier 1: Metadata (lookup tables)
        if tier == 1 or page_type in ['lookup_table']:
            if 'entries' in structured_data:
                for entry in structured_data['entries']:
                    self.storage['tier_1_metadata'].append({
                        **entry,
                        'page': page_num,
                        'page_type': page_type
                    })
                logger.debug(f"   → Tier 1 (Metadata): {len(structured_data['entries'])} entries")

        # Tier 2: Knowledge Graph (rules, specs, relationships)
        if tier == 2 or page_type in ['specifications', 'instructions']:
            # Specs become knowledge nodes
            if 'specs' in structured_data:
                for spec in structured_data['specs']:
                    self.storage['tier_2_knowledge'].append({
                        'type': 'specification',
                        'content': f"{spec.get('parameter', '')} {spec.get('requirement', '')} {spec.get('value', '')}",
                        'details': spec,
                        'page': page_num
                    })
                logger.debug(f"   → Tier 2 (Knowledge): {len(structured_data['specs'])} specs")

            # Rules and warnings
            if 'rules' in structured_data:
                for rule in structured_data['rules']:
                    self.storage['tier_2_knowledge'].append({
                        'type': 'rule',
                        'content': rule,
                        'page': page_num
                    })

            if 'warnings' in structured_data:
                for warning in structured_data['warnings']:
                    self.storage['tier_2_knowledge'].append({
                        'type': 'warning',
                        'content': warning,
                        'page': page_num
                    })

        # Tier 3: Document Structure (TOC, sections)
        if tier == 3 or page_type == 'table_of_contents':
            if 'sections' in structured_data:
                for section in structured_data['sections']:
                    self.storage['tier_3_structure'].append({
                        **section,
                        'source_page': page_num
                    })
                logger.debug(f"   → Tier 3 (Structure): {len(structured_data['sections'])} sections")

        # Tier 4: Semantic Search (always store chunks for searchability)
        if semantic_chunks:
            for i, chunk in enumerate(semantic_chunks):
                self.storage['tier_4_semantic'].append({
                    'text': chunk,
                    'page': page_num,
                    'page_type': page_type,
                    'chunk_index': i
                })
            logger.debug(f"   → Tier 4 (Semantic): {len(semantic_chunks)} chunks")

    def get_storage_summary(self) -> Dict[str, int]:
        """Get count of items in each storage tier"""
        return {
            "tier_1_metadata": len(self.storage['tier_1_metadata']),
            "tier_2_knowledge": len(self.storage['tier_2_knowledge']),
            "tier_3_structure": len(self.storage['tier_3_structure']),
            "tier_4_semantic": len(self.storage['tier_4_semantic'])
        }

    def save_to_json(self, output_dir: str):
        """Save all tiers to JSON files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for tier_name, data in self.storage.items():
            filepath = output_path / f"{tier_name}.json"
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"💾 Saved {filepath} ({len(data)} items)")


# ============================================================================
# Main Extraction Pipeline
# ============================================================================

def extract_intelligent_metadata(pdf_path: str,
                                 start_page: int = 1,
                                 end_page: int = 15,
                                 model: str = "llama3.1",
                                 dpi: int = 300) -> KnowledgeRouter:
    """
    Main intelligent extraction pipeline

    1. Convert PDF pages to images
    2. OCR to extract text
    3. LLM analyzes page type and storage tier
    4. LLM extracts structured data
    5. LLM creates semantic chunks
    6. Router decides where to store everything

    Returns:
        KnowledgeRouter with all extracted data
    """
    logger.info("=" * 70)
    logger.info("🧠 Intelligent Metadata Extraction")
    logger.info("=" * 70)
    logger.info(f"PDF: {pdf_path}")
    logger.info(f"Pages: {start_page}-{end_page}")
    logger.info(f"Model: {model}")
    logger.info("")

    # Initialize components
    analyzer = ContentAnalyzer(model=model)
    router = KnowledgeRouter()

    # Convert PDF to images
    logger.info(f"🖼️  Converting pages to images (DPI={dpi})...")
    images = convert_from_path(
        pdf_path,
        dpi=dpi,
        first_page=start_page,
        last_page=end_page
    )
    logger.info(f"✓ Converted {len(images)} pages\n")

    # Process each page
    for i, image in enumerate(images, start=start_page):
        logger.info(f"📄 Page {i}")
        logger.info("-" * 70)

        # Step 1: OCR
        logger.info("   1️⃣  Running OCR...")
        text = pytesseract.image_to_string(image, lang='eng')
        logger.info(f"   ✓ Extracted {len(text)} characters")

        # Step 2: Analyze page type
        logger.info("   2️⃣  Analyzing page type with LLM...")
        page_analysis = analyzer.analyze_page_type(text, i)
        logger.info(f"   ✓ Type: {page_analysis['page_type']}")
        logger.info(f"   ✓ Tier: {page_analysis['storage_tier']} ({page_analysis['content_category']})")

        # Step 3: Extract structured data
        logger.info("   3️⃣  Extracting structured data...")
        structured_data = analyzer.extract_structured_data(
            text,
            page_analysis['page_type'],
            i
        )

        # Step 4: Extract semantic chunks
        logger.info("   4️⃣  Creating semantic chunks...")
        semantic_chunks = analyzer.extract_semantic_knowledge(text, i)

        # Step 5: Route to storage
        logger.info("   5️⃣  Routing to storage tiers...")
        router.route(page_analysis, structured_data, semantic_chunks, i)

        logger.info("")

    # Summary
    logger.info("=" * 70)
    logger.info("✅ Extraction Complete")
    logger.info("=" * 70)
    summary = router.get_storage_summary()
    for tier, count in summary.items():
        logger.info(f"   {tier}: {count} items")
    logger.info("=" * 70)

    return router


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Intelligent metadata extractor using LLM understanding"
    )
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--start-page", type=int, default=1, help="First page")
    parser.add_argument("--end-page", type=int, default=15, help="Last page")
    parser.add_argument("--model", default="llama3.1", help="LLM model")
    parser.add_argument("--output", default="metadata_intelligent", help="Output directory")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PDF conversion")

    args = parser.parse_args()

    # Run extraction
    router = extract_intelligent_metadata(
        pdf_path=args.pdf,
        start_page=args.start_page,
        end_page=args.end_page,
        model=args.model,
        dpi=args.dpi
    )

    # Save results
    router.save_to_json(args.output)

    logger.info(f"\n✅ Metadata saved to: {args.output}/")
    logger.info("\nFine Count: $0")


if __name__ == "__main__":
    main()
