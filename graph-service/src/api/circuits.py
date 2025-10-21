"""Circuit analysis API endpoints"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List
from ..services.analysis_service import AnalysisService
from ..repositories.circuit_repository import CircuitRepository
from ..models.schemas.circuit_schema import CircuitValidationSchema

router = APIRouter(prefix="/circuits", tags=["circuits"])

@router.get("/{circuit_id}")
async def get_circuit(circuit_id: str, vehicle_signature: str,
                     circuit_repo: CircuitRepository = Depends()):
    circuit = await circuit_repo.get_by_id(circuit_id, vehicle_signature)
    if not circuit:
        raise HTTPException(status_code=404, detail="Circuit not found")
    return circuit

@router.get("/{circuit_id}/analysis")
async def analyze_circuit(circuit_id: str, vehicle_signature: str,
                         analysis_service: AnalysisService = Depends()):
    return await analysis_service.analyze_circuit_comprehensive(circuit_id, vehicle_signature)

@router.get("/{circuit_id}/load-analysis")
async def get_circuit_load(circuit_id: str, vehicle_signature: str,
                          circuit_repo: CircuitRepository = Depends()):
    return await circuit_repo.get_circuit_load_analysis(circuit_id, vehicle_signature)

@router.get("/{circuit_id}/topology")
async def get_circuit_topology(circuit_id: str, vehicle_signature: str,
                              circuit_repo: CircuitRepository = Depends()):
    return await circuit_repo.get_circuit_topology(circuit_id, vehicle_signature)

@router.get("/")
async def list_circuits(vehicle_signature: str, circuit_type: str = None,
                       circuit_repo: CircuitRepository = Depends()):
    if circuit_type:
        return await circuit_repo.find_by_type(vehicle_signature, circuit_type)
    else:
        from ..models.schemas.circuit_schema import CircuitQuerySchema
        query = CircuitQuerySchema(vehicle_signature=vehicle_signature)
        return await circuit_repo.find_by_criteria(query)

@router.post("/")
async def create_circuit(circuit: CircuitValidationSchema,
                        circuit_repo: CircuitRepository = Depends()):
    return await circuit_repo.create(circuit)