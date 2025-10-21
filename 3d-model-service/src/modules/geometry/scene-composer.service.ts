import { Injectable } from '@nestjs/common';
import * as THREE from 'three';
import { ComponentEntity, ElectricalSystemData, ConnectionEntity } from '../../common/entities/component.entity';
import { ComponentFactoryService } from './component-factory.service';
import { WireFactoryService, WireRoute, HarnessData } from './wire-factory.service';
import { MaterialService } from './material.service';

export interface SceneCompositionOptions {
  quality: 'low' | 'medium' | 'high';
  includeLighting: boolean;
  includeEnvironment: boolean;
  optimizeForWeb: boolean;
  generateShadows: boolean;
  backgroundColor?: string;
  ambientLightIntensity?: number;
  directionalLightIntensity?: number;
}

export interface ComposedScene {
  scene: THREE.Scene;
  components: THREE.Object3D[];
  wires: THREE.Object3D[];
  harnesses: THREE.Object3D[];
  metadata: {
    componentCount: number;
    wireCount: number;
    harnessCount: number;
    triangleCount: number;
    boundingBox: THREE.Box3;
  };
}

@Injectable()
export class SceneComposerService {
  constructor(
    private readonly componentFactory: ComponentFactoryService,
    private readonly wireFactory: WireFactoryService,
    private readonly materialService: MaterialService
  ) {}

  async composeScene(
    electricalData: ElectricalSystemData,
    options: SceneCompositionOptions = {
      quality: 'medium',
      includeLighting: true,
      includeEnvironment: false,
      optimizeForWeb: true,
      generateShadows: true,
      ambientLightIntensity: 0.6,
      directionalLightIntensity: 0.8
    }
  ): Promise<ComposedScene> {
    // Create main scene
    const scene = new THREE.Scene();
    scene.name = `ElectricalSystem_${electricalData.id}`;

    // Set background
    if (options.backgroundColor) {
      scene.background = new THREE.Color(options.backgroundColor);
    }

    // Generate components
    console.log(`Generating ${electricalData.components.length} components...`);
    const components = await this.generateComponents(electricalData.components, options);
    
    // Generate wires and harnesses
    console.log(`Generating ${electricalData.connections.length} wire connections...`);
    const { wires, harnesses } = await this.generateWiring(electricalData, options);

    // Create component groups
    const componentGroup = new THREE.Group();
    componentGroup.name = 'components';
    components.forEach(comp => componentGroup.add(comp));
    
    const wireGroup = new THREE.Group();
    wireGroup.name = 'wires';
    wires.forEach(wire => wireGroup.add(wire));
    
    const harnessGroup = new THREE.Group();
    harnessGroup.name = 'harnesses';
    harnesses.forEach(harness => harnessGroup.add(harness));

    // Add groups to scene
    scene.add(componentGroup);
    scene.add(wireGroup);
    scene.add(harnessGroup);

    // Add lighting
    if (options.includeLighting) {
      this.addLighting(scene, options);
    }

    // Add environment (vehicle frame reference)
    if (options.includeEnvironment) {
      this.addEnvironment(scene, electricalData);
    }

    // Calculate metadata
    const metadata = this.calculateSceneMetadata(scene, components, wires, harnesses);

    return {
      scene,
      components,
      wires,
      harnesses,
      metadata
    };
  }

  private async generateComponents(
    components: ComponentEntity[],
    options: SceneCompositionOptions
  ): Promise<THREE.Object3D[]> {
    const componentMeshes: THREE.Object3D[] = [];

    const componentOptions = {
      quality: options.quality,
      generateHitZone: !options.optimizeForWeb, // Skip hit zones for web optimization
      includeShadows: options.generateShadows
    };

    // Group components by type for organized generation
    const componentsByType = this.groupComponentsByType(components);

    for (const [type, typeComponents] of componentsByType.entries()) {
      console.log(`Generating ${typeComponents.length} ${type} components...`);
      
      const typeMeshes = await this.componentFactory.generateBatch(typeComponents, componentOptions);
      componentMeshes.push(...typeMeshes);
    }

    return componentMeshes;
  }

  private async generateWiring(
    electricalData: ElectricalSystemData,
    options: SceneCompositionOptions
  ): Promise<{ wires: THREE.Object3D[], harnesses: THREE.Object3D[] }> {
    const wires: THREE.Object3D[] = [];
    const harnesses: THREE.Object3D[] = [];

    // Create position lookup for components
    const componentPositions = new Map<string, any>();
    electricalData.components.forEach(comp => {
      componentPositions.set(comp.id, comp.position);
    });

    const wireOptions = {
      quality: options.quality,
      includeShadows: options.generateShadows,
      routingAlgorithm: 'optimized' as const
    };

    // Generate direct wire connections
    for (const connection of electricalData.connections) {
      const startPos = componentPositions.get(connection.fromComponentId);
      const endPos = componentPositions.get(connection.toComponentId);

      if (startPos && endPos) {
        try {
          const wire = await this.wireFactory.generateDirectConnection(
            startPos,
            endPos,
            connection,
            wireOptions
          );
          wires.push(wire);
        } catch (error) {
          console.warn(`Failed to generate wire for connection ${connection.id}:`, error.message);
        }
      }
    }

    // Generate wire harnesses (bundled wires)
    const harnessRoutes = this.extractHarnessRoutes(electricalData);
    const harnessObjects = await this.wireFactory.generateHarnessBatch(harnessRoutes, wireOptions);
    harnesses.push(...harnessObjects);

    return { wires, harnesses };
  }

  private addLighting(scene: THREE.Scene, options: SceneCompositionOptions) {
    // Ambient light for overall illumination
    const ambientLight = new THREE.AmbientLight(
      0xffffff,
      options.ambientLightIntensity || 0.6
    );
    ambientLight.name = 'AmbientLight';
    scene.add(ambientLight);

    // Directional light for shadows and depth
    const directionalLight = new THREE.DirectionalLight(
      0xffffff,
      options.directionalLightIntensity || 0.8
    );
    directionalLight.position.set(5, 5, 5);
    directionalLight.name = 'DirectionalLight';
    
    // Configure shadow mapping
    if (options.generateShadows) {
      directionalLight.castShadow = true;
      directionalLight.shadow.mapSize.width = 2048;
      directionalLight.shadow.mapSize.height = 2048;
      directionalLight.shadow.camera.near = 0.5;
      directionalLight.shadow.camera.far = 50;
    }
    
    scene.add(directionalLight);

    // Additional fill light
    const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
    fillLight.position.set(-5, 2, -5);
    fillLight.name = 'FillLight';
    scene.add(fillLight);
  }

  private addEnvironment(scene: THREE.Scene, electricalData: ElectricalSystemData) {
    // Add vehicle frame reference (basic bounding box)
    const vehicleGroup = new THREE.Group();
    vehicleGroup.name = 'vehicle_frame';

    // Default vehicle dimensions (can be customized per vehicle)
    const vehicleDimensions = {
      length: 4.2,
      width: 1.7,
      height: 1.8
    };

    // Create wireframe vehicle outline
    const vehicleGeometry = new THREE.BoxGeometry(
      vehicleDimensions.length,
      vehicleDimensions.width,
      vehicleDimensions.height
    );
    const vehicleEdges = new THREE.EdgesGeometry(vehicleGeometry);
    const vehicleWireframe = new THREE.LineSegments(
      vehicleEdges,
      new THREE.LineBasicMaterial({ color: 0x444444, opacity: 0.3, transparent: true })
    );
    vehicleWireframe.position.set(vehicleDimensions.length / 2, 0, vehicleDimensions.height / 2);
    vehicleWireframe.name = 'vehicle_outline';

    vehicleGroup.add(vehicleWireframe);

    // Add coordinate system reference
    const axesHelper = new THREE.AxesHelper(0.5);
    axesHelper.name = 'coordinate_axes';
    vehicleGroup.add(axesHelper);

    scene.add(vehicleGroup);
  }

  private groupComponentsByType(components: ComponentEntity[]): Map<string, ComponentEntity[]> {
    const groups = new Map<string, ComponentEntity[]>();

    components.forEach(component => {
      const type = component.type;
      if (!groups.has(type)) {
        groups.set(type, []);
      }
      groups.get(type)!.push(component);
    });

    return groups;
  }

  private extractHarnessRoutes(electricalData: ElectricalSystemData): HarnessData[] {
    // Extract harness routing data (this would come from spatial analysis)
    // For now, create basic harness groupings by zone
    const harnesses: HarnessData[] = [];
    
    // Group connections by zone/area for harness bundling
    const zoneConnections = new Map<string, ConnectionEntity[]>();
    
    electricalData.connections.forEach(connection => {
      // Determine harness zone based on component locations
      const fromComponent = electricalData.components.find(c => c.id === connection.fromComponentId);
      const toComponent = electricalData.components.find(c => c.id === connection.toComponentId);
      
      if (fromComponent && toComponent) {
        const zone = this.determineHarnessZone(fromComponent, toComponent);
        if (!zoneConnections.has(zone)) {
          zoneConnections.set(zone, []);
        }
        zoneConnections.get(zone)!.push(connection);
      }
    });

    // Create harness data for each zone
    zoneConnections.forEach((connections, zone) => {
      if (connections.length > 1) { // Only create harness for multiple wires
        harnesses.push({
          id: `harness_${zone}`,
          path: this.calculateHarnessPath(connections, electricalData.components),
          thickness: Math.min(0.02 + connections.length * 0.003, 0.05), // Scale with wire count
          type: zone
        });
      }
    });

    return harnesses;
  }

  private determineHarnessZone(fromComponent: ComponentEntity, toComponent: ComponentEntity): string {
    // Simple zone determination based on component zones
    const fromZone = fromComponent.zone || 'unknown';
    const toZone = toComponent.zone || 'unknown';
    
    if (fromZone === toZone) {
      return fromZone;
    }
    
    // Create inter-zone harness name
    return `${fromZone}_to_${toZone}`;
  }

  private calculateHarnessPath(connections: ConnectionEntity[], components: ComponentEntity[]): any[] {
    // Simplified harness path calculation
    // In a real implementation, this would use sophisticated routing algorithms
    
    if (connections.length === 0) return [];
    
    // Get all component positions involved in these connections
    const componentPositions = new Map<string, any>();
    components.forEach(comp => {
      componentPositions.set(comp.id, comp.position);
    });
    
    // For now, create a simple path between connection endpoints
    const firstConnection = connections[0];
    const startPos = componentPositions.get(firstConnection.fromComponentId);
    const endPos = componentPositions.get(firstConnection.toComponentId);
    
    if (startPos && endPos) {
      return [startPos, endPos];
    }
    
    return [];
  }

  private calculateSceneMetadata(
    scene: THREE.Scene,
    components: THREE.Object3D[],
    wires: THREE.Object3D[],
    harnesses: THREE.Object3D[]
  ) {
    let triangleCount = 0;
    const boundingBox = new THREE.Box3();

    // Calculate triangle count and bounding box
    scene.traverse((object) => {
      if (object instanceof THREE.Mesh && object.geometry) {
        const geometry = object.geometry;
        if (geometry.index) {
          triangleCount += geometry.index.count / 3;
        } else if (geometry.attributes.position) {
          triangleCount += geometry.attributes.position.count / 3;
        }
        
        // Update bounding box
        object.geometry.computeBoundingBox();
        if (object.geometry.boundingBox) {
          boundingBox.union(object.geometry.boundingBox);
        }
      }
    });

    return {
      componentCount: components.length,
      wireCount: wires.length,
      harnessCount: harnesses.length,
      triangleCount: Math.floor(triangleCount),
      boundingBox
    };
  }
}