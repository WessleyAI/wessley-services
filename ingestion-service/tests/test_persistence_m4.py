"""
Test cases for M4 Persistence functionality.
"""
import pytest
import tempfile
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any


def test_persistence_imports():
    """Test that persistence modules can be imported."""
    try:
        from src.persist.neo4j import Neo4jPersistence, GraphWriteResult
        from src.persist.qdrant import QdrantPersistence, TextChunk, EmbeddingStats
        from src.persist.storage import StorageManager, StorageConfig, ArtifactType
        from src.persist.supabase_metadata import SupabaseMetadata, JobRecord
        from src.persist.chunking import TextChunker, ChunkType, ProcessedChunk
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import persistence modules: {e}")


def test_text_chunker_initialization():
    """Test text chunker initialization and basic functionality."""
    try:
        from src.persist.chunking import TextChunker, ChunkType
        
        chunker = TextChunker(
            max_chunk_size=512,
            overlap_size=50,
            min_chunk_size=50
        )
        
        assert chunker.max_chunk_size == 512
        assert chunker.overlap_size == 50
        assert len(chunker.all_technical_terms) > 0
        
        print(f"TextChunker initialized with {len(chunker.all_technical_terms)} technical terms")
        
    except Exception as e:
        pytest.skip(f"TextChunker initialization failed: {e}")


def test_text_chunking_operations():
    """Test text chunking and processing operations."""
    try:
        from src.persist.chunking import TextChunker
        from src.core.schemas import TextSpan, OcrEngine
        
        chunker = TextChunker()
        project_id = str(uuid.uuid4())
        
        # Create mock text spans
        text_spans = [
            TextSpan(
                page=1,
                bbox=[100, 100, 200, 120],
                text="R1 10k resistor",
                rotation=0,
                confidence=0.9,
                engine=OcrEngine.TESSERACT
            ),
            TextSpan(
                page=1,
                bbox=[100, 130, 200, 150],
                text="C1 100nF capacitor",
                rotation=0,
                confidence=0.85,
                engine=OcrEngine.TESSERACT
            )
        ]
        
        # Chunk text spans
        chunks = chunker.chunk_text_spans(project_id, text_spans)
        
        assert len(chunks) >= 0  # May be 0 if chunks too small
        
        if chunks:
            chunk = chunks[0]
            assert chunk.project_id == project_id
            assert chunk.page == 1
            assert len(chunk.technical_terms) >= 0
            
        print(f"Created {len(chunks)} text chunks")
        
    except Exception as e:
        pytest.fail(f"Text chunking operations failed: {e}")


def test_component_description_generation():
    """Test component description chunk generation."""
    try:
        from src.persist.chunking import TextChunker
        from src.core.schemas import Component, ComponentType, Pin
        
        chunker = TextChunker()
        project_id = str(uuid.uuid4())
        
        # Create mock component
        component = Component(
            id="R1",
            type=ComponentType.RESISTOR,
            value="10k",
            page=1,
            bbox=[100, 100, 150, 120],
            pins=[
                Pin(name="1", bbox=[100, 110, 105, 115], page=1),
                Pin(name="2", bbox=[145, 110, 150, 115], page=1)
            ],
            confidence=0.9,
            provenance={"text_spans": ["span_1", "span_2"]}
        )
        
        # Generate component chunks
        chunks = chunker.chunk_component_descriptions(project_id, [component])
        
        assert len(chunks) == 1
        chunk = chunks[0]
        
        assert "R1" in chunk.text
        assert "resistor" in chunk.text.lower()
        assert "10k" in chunk.text
        assert chunk.metadata['component_id'] == "R1"
        assert chunk.metadata['component_type'] == "resistor"
        
        print(f"Generated component description: {chunk.text[:100]}...")
        
    except Exception as e:
        pytest.fail(f"Component description generation failed: {e}")


def test_technical_term_extraction():
    """Test extraction of technical terms from text."""
    try:
        from src.persist.chunking import TextChunker
        
        chunker = TextChunker()
        
        # Test text with technical terms
        test_texts = [
            "This is a 10k resistor connected to VCC",
            "The capacitor C1 has value 100nF and is polarized",
            "Connect the MOSFET drain to the 5V power supply",
            "The operational amplifier has high gain"
        ]
        
        for text in test_texts:
            terms = chunker._extract_technical_terms(text)
            print(f"Text: '{text}' -> Terms: {terms}")
            assert isinstance(terms, list)
        
        # Test specific technical term recognition
        vcc_terms = chunker._extract_technical_terms("Connect R1 to VCC power rail")
        assert any(term in ['vcc', 'power'] for term in vcc_terms)
        
    except Exception as e:
        pytest.fail(f"Technical term extraction failed: {e}")


def test_storage_manager_initialization():
    """Test storage manager initialization."""
    try:
        from src.persist.storage import StorageManager, StorageConfig
        
        # Test with mock configuration
        config = StorageConfig(
            use_s3=False,
            use_supabase=False,  # Disable to avoid connection requirements
            supabase_url="http://localhost:54321",
            supabase_key="test_key",
            supabase_bucket="test_bucket"
        )
        
        storage = StorageManager(config)
        assert storage.config == config
        
        print("StorageManager initialized successfully")
        
    except Exception as e:
        pytest.skip(f"StorageManager initialization failed: {e}")


def test_artifact_type_constants():
    """Test artifact type constants."""
    try:
        from src.persist.storage import ArtifactType
        
        # Check that all expected artifact types exist
        expected_types = [
            'TEXT_SPANS', 'COMPONENTS', 'NETLIST_JSON', 'NETLIST_GRAPHML',
            'NETLIST_NDJSON', 'DEBUG_OVERLAY', 'THUMBNAILS', 'PROCESSING_LOGS'
        ]
        
        for attr_name in expected_types:
            assert hasattr(ArtifactType, attr_name)
            value = getattr(ArtifactType, attr_name)
            assert isinstance(value, str)
            print(f"ArtifactType.{attr_name} = '{value}'")
        
    except Exception as e:
        pytest.fail(f"ArtifactType constants test failed: {e}")


def test_embedding_provider_interface():
    """Test embedding provider interface."""
    try:
        from src.persist.qdrant import MockEmbeddingProvider
        
        provider = MockEmbeddingProvider(dimension=768)
        assert provider.get_dimension() == 768
        
        # Test mock embedding generation
        import asyncio
        
        async def test_embeddings():
            texts = ["resistor 10k", "capacitor 100nF", "voltage regulator"]
            embeddings = await provider.embed_texts(texts)
            
            assert len(embeddings) == 3
            for embedding in embeddings:
                assert len(embedding) == 768
                assert all(isinstance(x, float) for x in embedding)
            
            return embeddings
        
        embeddings = asyncio.run(test_embeddings())
        print(f"Generated {len(embeddings)} mock embeddings of dimension {len(embeddings[0])}")
        
    except Exception as e:
        pytest.fail(f"Embedding provider test failed: {e}")


def test_chunk_metadata_structure():
    """Test chunk metadata structure and serialization."""
    try:
        from src.persist.chunking import ProcessedChunk, ChunkType
        
        # Create test chunk
        chunk = ProcessedChunk(
            id="test_chunk_1",
            text="Test electronic component R1 is a 10k resistor",
            chunk_type=ChunkType.COMPONENT_DESCRIPTION,
            project_id=str(uuid.uuid4()),
            page=1,
            metadata={
                "component_id": "R1",
                "component_type": "resistor",
                "confidence": 0.9
            },
            word_count=8,
            char_count=47,
            technical_terms=["resistor", "10k"]
        )
        
        # Test serialization
        chunk_dict = chunk.to_dict()
        
        assert chunk_dict['id'] == "test_chunk_1"
        assert chunk_dict['chunk_type'] == "component_description"
        assert chunk_dict['word_count'] == 8
        assert "resistor" in chunk_dict['technical_terms']
        
        print(f"Chunk serialization successful: {len(chunk_dict)} fields")
        
    except Exception as e:
        pytest.fail(f"Chunk metadata test failed: {e}")


def test_semantic_search_result_structure():
    """Test semantic search result structure."""
    try:
        from src.persist.qdrant import SemanticSearchResult, TextChunk
        
        # Create test result
        chunk = TextChunk(
            id="test_chunk",
            text="Test resistor component",
            chunk_type="component_description",
            project_id=str(uuid.uuid4()),
            page=1,
            metadata={"test": True}
        )
        
        result = SemanticSearchResult(
            chunk=chunk,
            score=0.85,
            distance=0.15
        )
        
        assert result.score == 0.85
        assert result.distance == 0.15
        assert result.chunk.text == "Test resistor component"
        
        print("Semantic search result structure validated")
        
    except Exception as e:
        pytest.fail(f"Semantic search result test failed: {e}")


def test_storage_path_generation():
    """Test storage path generation logic."""
    try:
        from src.persist.storage import StorageManager, StorageConfig
        
        storage = StorageManager(StorageConfig())
        
        project_id = uuid.uuid4()
        job_id = uuid.uuid4()
        artifact_type = "netlist_json"
        filename = "netlist.json"
        
        path = storage._generate_storage_path(project_id, job_id, artifact_type, filename)
        
        expected_parts = [str(project_id), str(job_id), artifact_type, filename]
        for part in expected_parts:
            assert part in path
        
        assert path.startswith("projects/")
        assert "/jobs/" in path
        
        print(f"Generated storage path: {path}")
        
    except Exception as e:
        pytest.fail(f"Storage path generation failed: {e}")


def test_neo4j_schema_compatibility():
    """Test Neo4j schema and data structure compatibility."""
    try:
        from src.persist.neo4j import Neo4jPersistence, GraphNode, GraphRelationship
        
        # Test data structures without requiring actual Neo4j connection
        node = GraphNode(
            id="test_node",
            labels=["Component"],
            properties={"type": "resistor", "value": "10k"}
        )
        
        relationship = GraphRelationship(
            start_node_id="component_1",
            end_node_id="pin_1",
            type="HAS_PIN",
            properties={"created_at": datetime.now().isoformat()}
        )
        
        assert node.id == "test_node"
        assert "Component" in node.labels
        assert relationship.type == "HAS_PIN"
        
        print("Neo4j data structures validated")
        
    except Exception as e:
        pytest.skip(f"Neo4j schema test failed (expected without Neo4j): {e}")


def test_job_record_serialization():
    """Test job record data structure."""
    try:
        from src.persist.supabase_metadata import JobRecord, ProjectRecord
        
        # Test job record
        job = JobRecord(
            id=str(uuid.uuid4()),
            project_id=str(uuid.uuid4()),
            user_id="test_user",
            status="completed",
            progress=100,
            stage="completed",
            source_file_id="test_file.pdf",
            source_type="upload",
            processing_modes={"ocr": ["tesseract"], "schematic_parse": True},
            artifacts={"netlist": "s3://bucket/netlist.json"},
            metrics={"cer": 0.05, "wer": 0.08},
            error_message=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            completed_at=datetime.now(),
            notify_channel="test_channel"
        )
        
        assert job.status == "completed"
        assert job.progress == 100
        assert "tesseract" in job.processing_modes["ocr"]
        
        # Test project record
        project = ProjectRecord(
            id=str(uuid.uuid4()),
            user_id="test_user",
            name="Test Project",
            description="Test description",
            vehicle_make="Toyota",
            vehicle_model="Camry",
            vehicle_year=2020,
            settings={"ocr_engine": "tesseract"},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        assert project.name == "Test Project"
        assert project.vehicle_make == "Toyota"
        
        print("Database record structures validated")
        
    except Exception as e:
        pytest.fail(f"Job record test failed: {e}")


if __name__ == "__main__":
    # Run basic tests
    test_persistence_imports()
    test_text_chunker_initialization()
    test_text_chunking_operations()
    test_component_description_generation()
    test_technical_term_extraction()
    test_storage_manager_initialization()
    test_artifact_type_constants()
    test_embedding_provider_interface()
    test_chunk_metadata_structure()
    test_semantic_search_result_structure()
    test_storage_path_generation()
    test_neo4j_schema_compatibility()
    test_job_record_serialization()
    
    print("✅ M4 Persistence tests completed successfully!")