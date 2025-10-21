"""System analysis API endpoints"""

from fastapi import APIRouter, HTTPException, Depends
from ..services.analysis_service import AnalysisService
from ..repositories.analytics_repository import AnalyticsRepository

router = APIRouter(prefix="/analysis", tags=["analysis"])

@router.get("/{vehicle_signature}/comprehensive")
async def comprehensive_analysis(vehicle_signature: str, 
                               analysis_service: AnalysisService = Depends()):
    return await analysis_service.perform_comprehensive_analysis(vehicle_signature)

@router.get("/{vehicle_signature}/circuit-loads")
async def circuit_load_analysis(vehicle_signature: str,
                               analysis_service: AnalysisService = Depends()):
    return await analysis_service.analyze_circuit_loads(vehicle_signature)

@router.get("/{vehicle_signature}/fault-tolerance")
async def fault_tolerance_analysis(vehicle_signature: str,
                                  analysis_service: AnalysisService = Depends()):
    return await analysis_service.analyze_fault_tolerance(vehicle_signature)

@router.get("/{vehicle_signature}/power-efficiency")
async def power_efficiency_analysis(vehicle_signature: str,
                                   analysis_service: AnalysisService = Depends()):
    return await analysis_service.analyze_power_efficiency(vehicle_signature)

@router.get("/{vehicle_signature}/thermal")
async def thermal_analysis(vehicle_signature: str,
                          analysis_service: AnalysisService = Depends()):
    return await analysis_service.analyze_thermal_characteristics(vehicle_signature)

@router.get("/{vehicle_signature}/overview")
async def system_overview(vehicle_signature: str,
                         analytics_repo: AnalyticsRepository = Depends()):
    return await analytics_repo.get_system_overview(vehicle_signature)

@router.get("/{vehicle_signature}/data-quality")
async def data_quality_report(vehicle_signature: str,
                             analytics_repo: AnalyticsRepository = Depends()):
    return await analytics_repo.get_data_quality_report(vehicle_signature)