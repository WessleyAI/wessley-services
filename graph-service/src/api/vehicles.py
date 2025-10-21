"""Vehicle data API endpoints"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List
from ..services.vehicle_service import VehicleService
from ..repositories.analytics_repository import AnalyticsRepository

router = APIRouter(prefix="/vehicles", tags=["vehicles"])

@router.get("/{vehicle_signature}")
async def get_vehicle(vehicle_signature: str, vehicle_service: VehicleService = Depends()):
    vehicle = await vehicle_service.get_vehicle(vehicle_signature)
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return vehicle

@router.get("/{vehicle_signature}/statistics")
async def get_vehicle_stats(vehicle_signature: str, vehicle_service: VehicleService = Depends()):
    return await vehicle_service.get_vehicle_statistics(vehicle_signature)

@router.get("/{vehicle_signature}/zones")
async def get_vehicle_zones(vehicle_signature: str, vehicle_service: VehicleService = Depends()):
    return await vehicle_service.get_vehicle_zones(vehicle_signature)

@router.post("/{vehicle_signature}/zones")
async def create_standard_zones(vehicle_signature: str, vehicle_service: VehicleService = Depends()):
    return await vehicle_service.create_standard_zones(vehicle_signature)

@router.get("/{vehicle_signature}/validation")
async def validate_vehicle_data(vehicle_signature: str, vehicle_service: VehicleService = Depends()):
    return await vehicle_service.validate_vehicle_data(vehicle_signature)

@router.get("/{vehicle_signature}/export")
async def export_vehicle_data(vehicle_signature: str, format: str = "json",
                             vehicle_service: VehicleService = Depends()):
    return await vehicle_service.export_vehicle_data(vehicle_signature, format)

@router.get("/")
async def list_vehicles(make: str = None, year: int = None,
                       vehicle_service: VehicleService = Depends()):
    return await vehicle_service.list_vehicles(make=make, year=year)

@router.post("/")
async def create_vehicle(signature: str, make: str, model: str, year: int,
                        engine: str = None, market: str = None,
                        vehicle_service: VehicleService = Depends()):
    return await vehicle_service.create_vehicle(signature, make, model, year, engine, market)