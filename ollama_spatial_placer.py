#!/usr/bin/env python3
"""
Ollama-based spatial placement for automotive electrical components
Version 2: With contextual map and scale awareness
Uses local LLM (Llama 3.1 8B) to suggest 3D coordinates based on:
- Schematic position
- Peripheral component context
- Component sizes and scale
- Collision avoidance
Cost: $0 (runs locally)
"""

import os
import json
import math
import re
from neo4j import GraphDatabase
import ollama
from datetime import datetime

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

# ============================================================================
# COMPONENT SIZE DATABASE (typical automotive components in mm)
# ============================================================================
COMPONENT_SIZES = {
    "battery": 240,
    "alternator": 180,
    "starter": 200,
    "relay": 30,
    "fuse": 20,
    "fuse_box": 200,
    "relay_box": 150,
    "ecu": 180,
    "sensor": 40,
    "ground": 10,
    "connector": 40,
    "light": 150,
    "switch": 25,
    "default": 50
}

# Minimum clearance between components (mm)
MIN_CLEARANCE = 50

# Major landmarks (always included in context)
MAJOR_LANDMARKS = ["engine", "battery", "fuse_box", "relay_box", "alternator", "starter"]

print("🤖 Ollama Spatial Placement Engine v2")
print("=" * 70)
print(f"LLM: Llama 3.1 8B (local)")
print(f"Cost: $0.00")
print(f"Features: Contextual map, scale awareness, collision avoidance")
print()

# Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ============================================================================
# COMPONENT QUERIES
# ============================================================================

def get_components_without_spatial(session):
    """Get all components that don't have spatial placement yet"""
    result = session.run("""
        MATCH (c:Component)
        WHERE c.spatial_x IS NULL
        RETURN c.id as id,
               c.type as type,
               c.name as name,
               c.page as page,
               c.bbox_x1 as x1,
               c.bbox_y1 as y1,
               c.bbox_x2 as x2,
               c.bbox_y2 as y2,
               c.description as description
        ORDER BY c.page, c.bbox_y1
    """)
    return [dict(record) for record in result]

def get_already_placed_components(session):
    """Get all components that already have spatial coordinates"""
    result = session.run("""
        MATCH (c:Component)
        WHERE c.spatial_x IS NOT NULL
        RETURN c.id as id,
               c.type as type,
               c.name as name,
               c.spatial_x as x,
               c.spatial_y as y,
               c.spatial_z as z,
               c.spatial_zone as zone,
               c.estimated_size as size
    """)
    return [dict(record) for record in result]

def analyze_schematic_position(component):
    """Analyze position in schematic for context"""
    # Assume standard A4 page size at 300 DPI: ~2480 × 3508 pixels
    # Use actual page dimensions if available
    page_width = 1200  # Approximate from component data
    page_height = 1600

    x_center = (component['x1'] + component['x2']) / 2
    y_center = (component['y1'] + component['y2']) / 2

    # Normalize to 0-1
    norm_x = x_center / page_width
    norm_y = y_center / page_height

    # Determine quadrant
    h_pos = "left" if norm_x < 0.5 else "right"
    v_pos = "top" if norm_y < 0.5 else "bottom"
    quadrant = f"{v_pos}-{h_pos}"

    return {
        "quadrant": quadrant,
        "horizontal": h_pos,
        "vertical": v_pos,
        "normalized_x": round(norm_x, 3),
        "normalized_y": round(norm_y, 3)
    }

def infer_zone_from_position(position, comp_type):
    """Infer likely vehicle zone based on schematic position and component type"""
    zones = []

    # Engine bay components
    if comp_type in ['battery', 'starter', 'alternator', 'relay', 'fuse']:
        if position['horizontal'] == 'left':
            zones.append("Engine Bay - Driver Side")
        else:
            zones.append("Engine Bay - Passenger Side")

        if position['vertical'] == 'top':
            zones.append("Upper Engine Compartment")
        else:
            zones.append("Lower Engine Compartment")

    # Ground points
    if comp_type == 'ground':
        zones.append("Chassis Ground Point")

    return zones

# ============================================================================
# SIZE EXTRACTION
# ============================================================================

def extract_component_size(component):
    """
    Extract component size from OCR/description text or use type-based defaults
    Returns size in mm
    """
    # Try to extract from description/name
    text = " ".join([
        component.get('description', ''),
        component.get('name', '')
    ]).lower()

    # Look for size patterns: "200mm", "50 mm", etc.
    mm_pattern = r'(\d+)\s*mm'
    matches = re.findall(mm_pattern, text)
    if matches:
        # Use largest dimension found
        return max(int(m) for m in matches)

    # Fallback to type-based default
    comp_type = (component.get('type', '') or '').lower()
    for key in COMPONENT_SIZES:
        if key in comp_type:
            return COMPONENT_SIZES[key]

    return COMPONENT_SIZES['default']

# ============================================================================
# PERIPHERAL COMPONENT FINDER
# ============================================================================

def calculate_3d_distance(p1, p2):
    """Calculate Euclidean distance between two 3D points"""
    return math.sqrt(
        (p1['x'] - p2['x'])**2 +
        (p1['y'] - p2['y'])**2 +
        (p1['z'] - p2['z'])**2
    )

def estimate_initial_position(component, position):
    """
    Rough initial position estimate from schematic quadrant
    LLM will refine this based on peripheral context
    """
    h_pos = position['horizontal']
    v_pos = position['vertical']
    comp_type = (component.get('type', '') or '').lower()

    # Base estimate from quadrant
    x = -400 if h_pos == 'left' else 400
    y = 600 if v_pos == 'top' else 0
    z = 300 if v_pos == 'top' else 0

    # Type-specific adjustments
    if 'battery' in comp_type:
        x, y, z = 350, 400, 100
    elif 'starter' in comp_type:
        x, y, z = -300, 0, -100
    elif 'alternator' in comp_type:
        x, y, z = 200, 200, 100
    elif any(t in comp_type for t in ['relay', 'fuse']):
        x, y, z = -350, 300, 400

    return {"x": x, "y": y, "z": z}

def get_peripheral_components(estimate_pos, all_placed, max_distance=800):
    """
    Get nearby components within max_distance + always include landmarks
    Returns sorted by distance (closest first)
    """
    peripherals = []
    landmarks = []

    for placed in all_placed:
        distance = calculate_3d_distance(estimate_pos, placed)
        placed_with_distance = {**placed, "distance": round(distance, 1)}

        # Check if it's a major landmark
        comp_type = (placed.get('type', '') or '').lower()
        is_landmark = any(lm in comp_type or lm in placed.get('name', '').lower()
                         for lm in MAJOR_LANDMARKS)

        if is_landmark:
            landmarks.append(placed_with_distance)
        elif distance <= max_distance:
            peripherals.append(placed_with_distance)

    # Combine: landmarks first, then closest peripherals
    all_peripherals = landmarks + peripherals
    all_peripherals.sort(key=lambda x: x['distance'])

    # Return top 8 (avoid overwhelming LLM)
    return all_peripherals[:8]

# ============================================================================
# HYBRID SEMI-STRUCTURED PROMPT BUILDER
# ============================================================================

def build_contextual_map_prompt(component, position, estimate_pos, peripherals, component_size):
    """
    Build hybrid semi-structured prompt with:
    - Markdown tables for spatial data
    - Clear sections with headers
    - Scale awareness
    - Collision avoidance instructions
    """

    zones = infer_zone_from_position(position, component.get('type', ''))

    # Build peripheral map table
    peripheral_table = ""
    if peripherals:
        peripheral_table = """
[PERIPHERAL_MAP]
Components already placed near your estimated position:

Component Name       | Type      | Position (x,y,z)        | Size  | Distance
--------------------|-----------|-------------------------|-------|----------"""
        for p in peripherals:
            name = (p.get('name', 'Unknown') or 'Unknown')[:18].ljust(18)
            ptype = (p.get('type', 'N/A') or 'N/A')[:9].ljust(9)
            pos = f"({p['x']:4.0f},{p['y']:4.0f},{p['z']:4.0f})"
            size = f"{p.get('size', 50):3.0f}mm"
            dist = f"{p['distance']:4.0f}mm"
            peripheral_table += f"\n{name} | {ptype} | {pos} | {size} | {dist}"

        peripheral_table += "\n"
    else:
        peripheral_table = "\n[PERIPHERAL_MAP]\nNo components placed yet - you are the first!\n"

    prompt = f"""[TASK] Place automotive electrical component in 3D vehicle model

[COMPONENT_TO_PLACE]
ID: {component['id']}
Type: {component.get('type', 'unknown')}
Name: {component.get('name', 'Unknown')}
Size: ~{component_size}mm (estimated)
Description: {component.get('description', 'N/A')}

[SCHEMATIC_CONTEXT]
Schematic quadrant: {position['quadrant']}
Position on page: {position['normalized_x']:.0%} from left, {position['normalized_y']:.0%} from top
Likely zones: {', '.join(zones) if zones else 'Engine bay (inferred)'}

[INITIAL_ESTIMATE]
Based on schematic analysis: ({estimate_pos['x']}, {estimate_pos['y']}, {estimate_pos['z']}) mm
⚠️  This is just a rough guess - YOU MUST ADJUST based on peripheral components!
{peripheral_table}
[CRITICAL_RULES]
1. SCALE AWARENESS: Components have physical size - they cannot overlap
2. MINIMUM CLEARANCE: Keep at least {MIN_CLEARANCE}mm from ALL nearby components
3. SIZE CONSIDERATION: This component needs ~{component_size}mm + {MIN_CLEARANCE}mm = {component_size + MIN_CLEARANCE}mm total space
4. COLLISION CHECK: Review peripheral map - ensure no overlap with existing components
5. REALISTIC COORDINATES: Engine bay bounds are X∈[-700,700], Y∈[-300,600], Z∈[-400,800]

[COORDINATE_SYSTEM]
Origin (0,0,0): Center of engine block (~600mm reference size)
X-axis: Negative = driver side (-), Positive = passenger side (+)
Y-axis: Negative = rear/firewall (-), Positive = front/radiator (+)
Z-axis: Negative = below engine (-), Positive = above engine (+)

[SCALE_REFERENCE]
Engine block: ~600mm (large landmark)
Battery (typical): ~240mm at (350, 400, 100)
Fuse box (typical): ~200mm at (-350, 300, 400)
Your component: ~{component_size}mm

[INSTRUCTIONS]
Step 1: Review peripheral map above - identify occupied space
Step 2: Find open area with {component_size + MIN_CLEARANCE}mm clearance
Step 3: Choose realistic zone for {component.get('type', 'this component type')}
Step 4: Verify no overlap with existing components
Step 5: Return JSON response

[OUTPUT_FORMAT]
Respond with VALID JSON ONLY:
{{
  "x": <integer in mm>,
  "y": <integer in mm>,
  "z": <integer in mm>,
  "confidence": <float 0.0-1.0>,
  "zone": "<descriptive zone name>",
  "reasoning": "<explain placement and mention clearances>",
  "nearest_component": "<name of closest component>",
  "clearance": <distance in mm to nearest component>
}}"""

    return prompt

# ============================================================================
# VALIDATION AND OVERLAP DETECTION
# ============================================================================

def check_overlap(new_component, all_placed):
    """
    Check if new component overlaps with any existing components
    Returns (is_valid, nearest_conflict, min_distance)
    """
    new_pos = {"x": new_component['x'], "y": new_component['y'], "z": new_component['z']}
    new_size = new_component.get('estimated_size', 50)

    min_distance = float('inf')
    nearest_conflict = None

    for placed in all_placed:
        placed_pos = {"x": placed['x'], "y": placed['y'], "z": placed['z']}
        distance = calculate_3d_distance(new_pos, placed_pos)

        if distance < min_distance:
            min_distance = distance
            nearest_conflict = placed

        # Check if too close (overlap or insufficient clearance)
        placed_size = placed.get('size', 50)
        required_clearance = (new_size / 2) + (placed_size / 2) + MIN_CLEARANCE

        if distance < required_clearance:
            return False, placed, distance  # Overlap detected!

    return True, nearest_conflict, min_distance

def validate_bounds(component):
    """Check if component is within engine bay bounds"""
    x, y, z = component['x'], component['y'], component['z']

    # Engine bay bounds
    if not (-700 <= x <= 700):
        return False, f"X={x} out of bounds [-700, 700]"
    if not (-300 <= y <= 600):
        return False, f"Y={y} out of bounds [-300, 600]"
    if not (-400 <= z <= 800):
        return False, f"Z={z} out of bounds [-400, 800]"

    return True, "OK"

# ============================================================================
# LLM CALL WITH CONTEXTUAL MAP
# ============================================================================

def get_spatial_suggestion_with_context(component, position, all_placed, session):
    """
    Call Ollama with full contextual map:
    1. Extract component size
    2. Estimate initial position
    3. Find peripheral components
    4. Build contextual prompt
    5. Get LLM suggestion
    6. Validate result
    """

    # 1. Extract component size
    component_size = extract_component_size(component)

    # 2. Estimate initial position from schematic
    estimate_pos = estimate_initial_position(component, position)

    # 3. Find peripheral components (nearby + landmarks)
    peripherals = get_peripheral_components(estimate_pos, all_placed, max_distance=800)

    # 4. Build contextual prompt
    prompt = build_contextual_map_prompt(component, position, estimate_pos, peripherals, component_size)

    # 5. Call Ollama
    try:
        response = ollama.generate(
            model="llama3.1:8b",
            prompt=prompt,
            format="json",
            options={
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": 500  # Allow longer response for detailed reasoning
            }
        )

        # Parse JSON response
        suggestion = json.loads(response['response'])

        # Validate required fields
        required = ['x', 'y', 'z', 'confidence', 'zone', 'reasoning']
        if not all(k in suggestion for k in required):
            print(f"  ⚠️  Incomplete response from LLM")
            return None

        # Add component size to result
        suggestion['estimated_size'] = component_size

        return suggestion

    except json.JSONDecodeError as e:
        print(f"  ❌ JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"  ❌ Ollama error: {e}")
        return None

def update_component_spatial(session, component_id, spatial_data):
    """Update component with spatial placement data including contextual info"""
    session.run("""
        MATCH (c:Component {id: $id})
        SET c.spatial_x = $x,
            c.spatial_y = $y,
            c.spatial_z = $z,
            c.spatial_confidence = $confidence,
            c.spatial_zone = $zone,
            c.spatial_reasoning = $reasoning,
            c.estimated_size = $size,
            c.nearest_component = $nearest,
            c.clearance = $clearance,
            c.spatial_method = 'llm_ollama_v2_contextual',
            c.spatial_generated_at = datetime()
    """,
    id=component_id,
    x=spatial_data['x'],
    y=spatial_data['y'],
    z=spatial_data['z'],
    confidence=spatial_data['confidence'],
    zone=spatial_data['zone'],
    reasoning=spatial_data['reasoning'],
    size=spatial_data.get('estimated_size', 50),
    nearest=spatial_data.get('nearest_component', 'none'),
    clearance=spatial_data.get('clearance', 0)
    )

# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================

with driver.session() as session:
    components = get_components_without_spatial(session)

    if not components:
        print("✅ All components already have spatial placement!")
        print()
    else:
        total = len(components)
        print(f"📊 Found {total} components without spatial placement")
        print()

        success_count = 0
        failed_count = 0

        for i, comp in enumerate(components, 1):
            print(f"[{i}/{total}] Processing {comp['id']} ({comp.get('type', 'unknown')})...")

            # Get already-placed components for contextual map
            placed_components = get_already_placed_components(session)
            print(f"  📍 Context: {len(placed_components)} components already placed")

            # Analyze schematic position
            position = analyze_schematic_position(comp)
            print(f"  📄 Schematic: {position['quadrant']} quadrant")

            # Get spatial suggestion with contextual map
            spatial = get_spatial_suggestion_with_context(comp, position, placed_components, session)

            if spatial:
                print(f"  🎯 Suggested: ({spatial['x']}, {spatial['y']}, {spatial['z']}) mm")
                print(f"  📏 Size: ~{spatial.get('estimated_size', 50)} mm")

                # Validate bounds
                bounds_ok, bounds_msg = validate_bounds(spatial)
                if not bounds_ok:
                    print(f"  ⚠️  Bounds check failed: {bounds_msg}")
                    failed_count += 1
                    continue

                # Check for overlaps
                if placed_components:
                    valid, conflict, min_dist = check_overlap(spatial, placed_components)
                    if not valid:
                        print(f"  ⚠️  Overlap detected with {conflict.get('name', 'component')} (distance: {min_dist:.0f}mm)")
                        failed_count += 1
                        continue
                    print(f"  ✓ Clearance OK: {min_dist:.0f}mm from nearest component")

                print(f"  🏷️  Zone: {spatial['zone']}")
                print(f"  💯 Confidence: {spatial['confidence']:.0%}")
                if spatial.get('nearest_component') and spatial.get('nearest_component') != 'none':
                    print(f"  🔗 Nearest: {spatial['nearest_component']}")
                print(f"  💭 Reasoning: {spatial['reasoning'][:100]}...")

                # Update Neo4j
                update_component_spatial(session, comp['id'], spatial)
                success_count += 1
                print(f"  ✅ Updated in Neo4j")
            else:
                failed_count += 1
                print(f"  ❌ Failed to generate spatial placement")

            print()

        print("=" * 70)
        print("✅ Spatial Placement Complete!")
        print()
        print(f"📊 Results:")
        print(f"  • Total components: {total}")
        print(f"  • Successfully placed: {success_count}")
        print(f"  • Failed: {failed_count}")
        print(f"  • Success rate: {success_count/total*100:.1f}%")
        print()
        print(f"💰 Cost: $0.00 (local Ollama)")
        print()
        print("🚀 Next steps:")
        print("  • Validate placements for overlaps")
        print("  • Generate 3D model using spatial coordinates")
        print("  • Query: MATCH (c:Component) WHERE c.spatial_x IS NOT NULL RETURN c")

driver.close()

print()
print("💰 Fine Count: $0")
