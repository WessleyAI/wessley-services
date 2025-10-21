"""Data import/export API endpoints"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from ..services.import_service import ImportService
from ..models.schemas.import_schema import GraphMLImportSchema, JSONImportSchema

router = APIRouter(prefix="/import-export", tags=["import-export"])

@router.post("/graphml")
async def import_graphml(file: UploadFile = File(...), 
                        vehicle_signature: str = None,
                        import_service: ImportService = Depends()):
    if not vehicle_signature:
        raise HTTPException(status_code=400, detail="vehicle_signature required")
    
    content = await file.read()
    from ..models.schemas.import_schema import ImportMetadata
    metadata = ImportMetadata(
        source_file=file.filename,
        format="graphml",
        vehicle_signature=vehicle_signature
    )
    
    schema = GraphMLImportSchema(
        metadata=metadata,
        graphml_content=content.decode('utf-8')
    )
    
    return await import_service.import_graphml(schema)

@router.post("/json")
async def import_json(data: JSONImportSchema, import_service: ImportService = Depends()):
    return await import_service.import_json(data)

@router.post("/csv")
async def import_csv(file: UploadFile = File(...),
                    vehicle_signature: str = None,
                    entity_type: str = None,
                    import_service: ImportService = Depends()):
    if not vehicle_signature or not entity_type:
        raise HTTPException(status_code=400, detail="vehicle_signature and entity_type required")
    
    content = await file.read()
    from ..models.schemas.import_schema import ImportMetadata, CSVImportSchema
    
    metadata = ImportMetadata(
        source_file=file.filename,
        format="csv",
        vehicle_signature=vehicle_signature
    )
    
    schema = CSVImportSchema(
        metadata=metadata,
        csv_content=content.decode('utf-8'),
        entity_type=entity_type,
        column_mapping={"id": "id", "name": "name", "type": "type"}  # Default mapping
    )
    
    return await import_service.import_csv(schema)

@router.get("/export/{vehicle_signature}")
async def export_data(vehicle_signature: str, format: str = "json",
                     import_service: ImportService = Depends()):
    # This would use vehicle_service.export_vehicle_data
    pass