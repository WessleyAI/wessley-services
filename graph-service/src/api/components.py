"""Component management API endpoints"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from ..services.graph_service import GraphService
from ..repositories.component_repository import ComponentRepository
from ..models.schemas.component_schema import ComponentValidationSchema, ComponentQuerySchema

router = APIRouter(prefix="/components", tags=["components"])

@router.get("/{component_id}")
async def get_component(component_id: str, vehicle_signature: str, 
                       component_repo: ComponentRepository = Depends()):
    component = await component_repo.get_by_id(component_id, vehicle_signature)
    if not component:
        raise HTTPException(status_code=404, detail="Component not found")
    return component

@router.post("/")
async def create_component(component: ComponentValidationSchema, 
                          graph_service: GraphService = Depends()):
    return await graph_service.create_component(component)

@router.get("/")
async def query_components(vehicle_signature: str, 
                          component_type: Optional[str] = None,
                          zone: Optional[str] = None,
                          component_repo: ComponentRepository = Depends()):
    query = ComponentQuerySchema(vehicle_signature=vehicle_signature)
    return await component_repo.find_by_criteria(query)

@router.put("/{component_id}")
async def update_component(component_id: str, vehicle_signature: str, 
                          updates: dict, component_repo: ComponentRepository = Depends()):
    updated = await component_repo.update(component_id, vehicle_signature, updates)
    if not updated:
        raise HTTPException(status_code=404, detail="Component not found")
    return updated

@router.delete("/{component_id}")
async def delete_component(component_id: str, vehicle_signature: str,
                          component_repo: ComponentRepository = Depends()):
    deleted = await component_repo.delete(component_id, vehicle_signature)
    if not deleted:
        raise HTTPException(status_code=404, detail="Component not found")
    return {"message": "Component deleted successfully"}