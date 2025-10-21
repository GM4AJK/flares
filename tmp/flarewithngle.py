"""
Satellite Flare Prediction Program
Predicts when a satellite might produce a flare similar to Iridium flares
based on specular reflection from solar panels at a configurable angle from perpendicular to the Sun.

FIXED VERSION: Searches all possible rotation directions around the sun direction.
"""

from skyfield.api import load, EarthSatellite, Topos
from skyfield import almanac
import numpy as np
from datetime import datetime, timedelta

def rotate_vector_around_axis(vector, axis, angle_deg):
    """
    Rotate a vector around an arbitrary axis by a given angle using Rodrigues' rotation formula.
    
    Parameters:
    - vector: The vector to rotate (3D numpy array)
    - axis: The axis of rotation (3D numpy array, will be normalized)
    - angle_deg: Rotation angle in degrees
    
    Returns:
    - Rotated vector (3D numpy array)
    """
    angle_rad = np.radians(angle_deg)
    axis = axis / np.linalg.norm(axis)  # Normalize axis
    
    # Rodrigues' rotation formula:
    # v_rot = v*cos(θ) + (k×v)*sin(θ) + k*(k·v)*(1-cos(θ))
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    rotated = (vector * cos_angle +
               np.cross(axis, vector) * sin_angle +
               axis * np.dot(axis, vector) * (1 - cos_angle))
    
    return rotated

def calculate_specular_angle_optimized(satellite_pos, sun_pos, observer_pos, panel_offset_angle=0.0):
    """
    Calculate the minimum specular angle by searching all possible rotation directions.
    
    When panel_offset_angle > 0, the panel can be rotated around the sun direction
    in any direction. This function finds the rotation that produces the best
    (smallest) specular angle.
    
    Parameters:
    - satellite_pos: Position vector of satellite relative to Earth center
    - sun_pos: Position vector of Sun relative to Earth center
    - observer_pos: Position vector of observer relative to Earth center
    - panel_offset_angle: Angle in degrees to offset the panel from perpendicular to Sun (default 0.0)
    
    Returns:
    - best_angle: Minimum specular angle in degrees (0 = perfect reflection toward observer)
    - best_panel_normal: The actual panel normal vector that produces the best reflection
    - rotation_axis: The axis around which the rotation was performed
    """
    # Vector from Sun to satellite (incident direction)
    sun_to_sat = satellite_pos - sun_pos
    sun_to_sat_unit = sun_to_sat / np.linalg.norm(sun_to_sat)
    
    # Base normal vector of solar panel (perpendicular to Sun direction)
    # Points toward the Sun
    base_panel_normal = -sun_to_sat_unit
    
    # Vector from satellite to observer
    sat_to_obs = observer_pos - satellite_pos
    sat_to_obs_unit = sat_to_obs / np.linalg.norm(sat_to_obs)
    
    # If no offset, use perpendicular to sun
    if abs(panel_offset_angle) < 0.001:
        reflection_vector = sun_to_sat_unit - 2 * np.dot(sun_to_sat_unit, base_panel_normal) * base_panel_normal
        cos_angle = np.dot(reflection_vector, sat_to_obs_unit)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        return angle, base_panel_normal, None
    
    # Search through all possible rotation directions around the sun direction
    # The rotation axis must be perpendicular to the sun direction
    best_angle = 180.0
    best_panel_normal = base_panel_normal
    best_rotation_axis = None
    
    # Create two orthogonal vectors perpendicular to sun direction
    if abs(sun_to_sat_unit[2]) < 0.9:
        perp1 = np.cross(sun_to_sat_unit, np.array([0, 0, 1]))
    else:
        perp1 = np.cross(sun_to_sat_unit, np.array([1, 0, 0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    
    perp2 = np.cross(sun_to_sat_unit, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)
    
    # Search through rotation directions (azimuthal angle around sun direction)
    num_samples = 72  # Check every 5 degrees
    for i in range(num_samples):
        azimuth = 2 * np.pi * i / num_samples
        
        # Rotation axis is a linear combination of the two perpendicular vectors
        rotation_axis = np.cos(azimuth) * perp1 + np.sin(azimuth) * perp2
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        
        # Rotate the base panel normal
        panel_normal = rotate_vector_around_axis(base_panel_normal, rotation_axis, panel_offset_angle)
        panel_normal = panel_normal / np.linalg.norm(panel_normal)
        
        # Calculate specular reflection
        reflection_vector = sun_to_sat_unit - 2 * np.dot(sun_to_sat_unit, panel_normal) * panel_normal
        cos_angle = np.dot(reflection_vector, sat_to_obs_unit)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        # Keep track of best result
        if angle < best_angle:
            best_angle = angle
            best_panel_normal = panel_normal
            best_rotation_axis = rotation_axis
    
    return best_angle, best_panel_normal, best_rotation_axis

def calculate_flare_magnitude(angle, distance_km, base_magnitude=-8.0):
    """
    Estimate the visual magnitude of the flare based on specular angle and distance.
    
    Parameters:
    - angle: Specular reflection angle in degrees
    - distance_km: Distance from observer to satellite in km
    - base_magnitude: Peak magnitude at perfect reflection and reference distance
    
    Returns:
    - Estimated visual magnitude (lower = brighter)
    """
    # Flare intensity drops off rapidly with angle
    # Using a narrow Gaussian-like falloff (Iridium flares were very directional)
    angle_factor = np.exp(-(angle**2) / (2 * 0.5**2))  # sigma = 0.5 degrees
    
    # Distance factor (inverse square law)
    reference_distance = 800.0  # km
    distance_factor = (reference_distance / distance_km) ** 2
    
    # Convert to magnitude (logarithmic scale)
    if angle_factor * distance_factor > 0:
        magnitude = base_magnitude - 2.5 * np.log10(angle_factor * distance_factor)
    else:
        magnitude = 99.0  # Not visible
    
    return magnitude

def test_specific_time(satellite, observer, test_time, panel_offset_angle, ts, eph):
    """
    Test a specific time to verify flare detection.
    
    Parameters:
    - satellite: Skyfield EarthSatellite object
    - observer: Skyfield Topos observer location
    - test_time: Skyfield Time object for the specific test time
    - panel_offset_angle: Panel offset angle in degrees
    - ts: Timescale object
    - eph: Ephemeris object
    
    Returns:
    - Dictionary with test results
    """
    sun = eph['sun']
    earth = eph['earth']
    
    print("\n" + "=" * 85)
    print(f"TEST CASE: {test_time.utc_iso()}")
    print("=" * 85)
    
    # Create observer geocentric position
    observer_geocentric = earth + observer
    
    # Get satellite position from observer (topocentric)
    difference = (satellite - observer).at(test_time)
    alt, az, distance = difference.altaz()
    
    print(f"Satellite Position:")
    print(f"  Altitude: {alt.degrees:.2f}°")
    print(f"  Azimuth: {az.degrees:.2f}°")
    print(f"  Distance: {distance.km:.2f} km")
    
    # Get sun position
    sun_topocentric = (sun - observer_geocentric).at(test_time)
    sun_alt = sun_topocentric.altaz()[0].degrees
    print(f"\nSun Position:")
    print(f"  Altitude: {sun_alt:.2f}°")
    
    # Get geocentric positions for specular angle calculation
    sat_geocentric = earth + satellite
    sat_pos = sat_geocentric.at(test_time).position.km
    sun_pos = sun.at(test_time).position.km
    obs_pos = observer_geocentric.at(test_time).position.km
    
    # Calculate sun-satellite-observer angle
    sat_to_sun = sun_pos - sat_pos
    sat_to_obs = obs_pos - sat_pos
    sat_to_sun_unit = sat_to_sun / np.linalg.norm(sat_to_sun)
    sat_to_obs_unit = sat_to_obs / np.linalg.norm(sat_to_obs)
    sun_obs_angle = np.degrees(np.arccos(np.clip(np.dot(sat_to_sun_unit, sat_to_obs_unit), -1.0, 1.0)))
    
    print(f"\nGeometry:")
    print(f"  Sun-Satellite-Observer Angle: {sun_obs_angle:.2f}°")
    print(f"  Expected optimal panel offset: {sun_obs_angle/2:.2f}°")
    print(f"  Configured panel offset: {panel_offset_angle}°")
    
    # Calculate minimum specular angle across all rotation directions
    spec_angle, panel_normal, rotation_axis = calculate_specular_angle_optimized(
        sat_pos, sun_pos, obs_pos, panel_offset_angle
    )
    
    # Calculate angle between panel normal and sun direction (for verification)
    sun_to_sat = sat_pos - sun_pos
    sun_to_sat_unit = sun_to_sat / np.linalg.norm(sun_to_sat)
    angle_to_sun = np.degrees(np.arccos(np.clip(-np.dot(panel_normal, sun_to_sat_unit), -1.0, 1.0)))
    
    # Calculate magnitude
    magnitude = calculate_flare_magnitude(spec_angle, distance.km)
    
    print(f"\nFlare Results:")
    print(f"  Specular Angle: {spec_angle:.6f}°")
    print(f"  Panel Angle to Sun: {angle_to_sun:.2f}°")
    print(f"  Visual Magnitude: {magnitude:.2f}")
    
    # Determine if flare would be detected
    angle_threshold = 2.0
    if spec_angle <= angle_threshold:
        print(f"\n  ✓ FLARE DETECTED (specular angle {spec_angle:.3f}° ≤ {angle_threshold}°)")
        test_passed = True
    else:
        print(f"\n  ✗ NO FLARE (specular angle {spec_angle:.3f}° > {angle_threshold}°)")
        test_passed = False
    
    print("=" * 85)
    
    return {
        'test_passed': test_passed,
        'specular_angle': spec_angle,
        'magnitude': magnitude,
        'altitude': alt.degrees,
        'sun_obs_angle': sun_obs_angle,
        'angle_to_sun': angle_to_sun
    }

def predict_satellite_flares(satellite, observer, start_time, end_time, 
                            panel_offset_angle=0.0,
                            time_step_seconds=10, angle_threshold=2.0, 
                            min_elevation=10.0):
    """
    Predict satellite flare events over a time window.
    
    Parameters:
    - satellite: Skyfield EarthSatellite object
    - observer: Skyfield Topos observer location
    - start_time: Skyfield Time object for start
    - end_time: Skyfield Time object for end
    - panel_offset_angle: Angle in degrees to offset panel from perpendicular to Sun (default 0.0)
    - time_step_seconds: Time resolution for scanning
    - angle_threshold: Maximum specular angle (degrees) to consider a flare
    - min_elevation: Minimum elevation above horizon (degrees)
    
    Returns:
    - List of flare events with timing and brightness information
    """
    ts = load.timescale()
    eph = load('de421.bsp')
    sun = eph['sun']
    earth = eph['earth']
    
    # Generate time array
    total_seconds = (end_time.utc_datetime() - start_time.utc_datetime()).total_seconds()
    num_steps = int(total_seconds / time_step_seconds)
    times = [start_time.utc_datetime() + timedelta(seconds=i*time_step_seconds) 
             for i in range(num_steps + 1)]
    time_objects = [ts.utc(t.year, t.month, t.day, t.hour, t.minute, t.second) 
                    for t in times]
    
    flare_events = []
    
    print(f"\nScanning for flares from {start_time.utc_iso()} to {end_time.utc_iso()}")
    print(f"Time step: {time_step_seconds} seconds")
    print(f"Panel offset angle: {panel_offset_angle}° from perpendicular to Sun")
    print(f"  (Searching all rotation directions around sun axis)")
    print(f"Angle threshold: {angle_threshold}°")
    print(f"Minimum elevation: {min_elevation}°")
    print("-" * 85)
    
    for t in time_objects:
        # Create observer geocentric position
        observer_geocentric = earth + observer
        
        # Get satellite position from observer (topocentric)
        difference = (satellite - observer).at(t)
        alt, az, distance = difference.altaz()
        
        # Check if satellite is above horizon
        if alt.degrees < min_elevation:
            continue
        
        # Get geocentric positions for specular angle calculation
        sat_geocentric = earth + satellite
        sat_pos = sat_geocentric.at(t).position.km
        sun_pos = sun.at(t).position.km
        obs_pos = observer_geocentric.at(t).position.km
        
        # Calculate minimum specular angle across all rotation directions
        spec_angle, panel_normal, rotation_axis = calculate_specular_angle_optimized(
            sat_pos, sun_pos, obs_pos, panel_offset_angle
        )
        
        # Check if this could be a flare
        if spec_angle <= angle_threshold:
            magnitude = calculate_flare_magnitude(spec_angle, distance.km)
            
            # Check if Sun is below horizon (satellite visible in darkness/twilight)
            sun_topocentric = (sun - observer_geocentric).at(t)
            sun_alt = sun_topocentric.altaz()[0].degrees
            sun_condition = "Day" if sun_alt > -6 else ("Twilight" if sun_alt > -18 else "Night")
            
            # Calculate angle between panel normal and sun direction (for verification)
            sun_to_sat = sat_pos - sun_pos
            sun_to_sat_unit = sun_to_sat / np.linalg.norm(sun_to_sat)
            angle_to_sun = np.degrees(np.arccos(np.clip(-np.dot(panel_normal, sun_to_sat_unit), -1.0, 1.0)))
            
            flare_events.append({
                'time': t,
                'utc_datetime': t.utc_datetime(),
                'specular_angle': spec_angle,
                'magnitude': magnitude,
                'altitude': alt.degrees,
                'azimuth': az.degrees,
                'distance_km': distance.km,
                'sun_altitude': sun_alt,
                'visibility': sun_condition,
                'panel_normal': panel_normal,
                'rotation_axis': rotation_axis,
                'angle_to_sun': angle_to_sun
            })
    
    return flare_events

def print_flare_results(flare_events):
    """Print flare prediction results in a formatted table."""
    if not flare_events:
        print("\nNo flare events predicted in the given time window.")
        print("Try increasing the angle_threshold or expanding the time window.")
        return
    
    print(f"\n{'Time (UTC)':<20} {'Alt':<6} {'Az':<7} {'Spec':<7} {'→Sun':<6} {'Mag':<6} {'Dist':<8} {'Vis':<10}")
    print(f"{'':<20} {'(°)':<6} {'(°)':<7} {'(°)':<7} {'(°)':<6} {'':<6} {'(km)':<8} {'':<10}")
    print("-" * 85)
    
    for event in flare_events:
        time_str = event['utc_datetime'].strftime('%Y-%m-%d %H:%M:%S')
        print(f"{time_str:<20} "
              f"{event['altitude']:>5.1f}° "
              f"{event['azimuth']:>6.1f}° "
              f"{event['specular_angle']:>6.2f}° "
              f"{event['angle_to_sun']:>5.1f}° "
              f"{event['magnitude']:>5.1f} "
              f"{event['distance_km']:>7.1f} "
              f"{event['visibility']:<10}")
    
    # Find peak flare
    brightest = min(flare_events, key=lambda x: x['magnitude'])
    print("-" * 85)
    print(f"Peak flare: {brightest['utc_datetime'].strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Magnitude: {brightest['magnitude']:.1f}")
    print(f"  Altitude: {brightest['altitude']:.1f}°, Azimuth: {brightest['azimuth']:.1f}°")
    print(f"  Specular angle: {brightest['specular_angle']:.3f}°")
    print(f"  Panel angle to sun: {brightest['angle_to_sun']:.2f}°")

def main():
    """Main function to run the satellite flare prediction."""
    # Load timescale
    ts = load.timescale()
    eph = load('de421.bsp')
    
    # Satellite TLE (STARLINK-32653)
    line1 = '1 62199U 24229H   25291.43054285  .00031631  00000-0  11531-2 0  9993'
    line2 = '2 62199  43.0013 347.2445 0000606 261.3848  98.6935 15.27606777 48889'
    satellite = EarthSatellite(line1, line2, 'STARLINK-32653', ts)
    
    # Observer location (San Francisco Bay Area coordinates)
    observer = Topos(latitude_degrees=37.654444, longitude_degrees=-122.473943)
    
    # Panel offset angle (degrees from perpendicular to Sun)
    # 27.97° is the angle required for perfect reflection based on geometry
    panel_offset_angle = 27.97
    
    print("=" * 85)
    print("SATELLITE FLARE PREDICTION (FIXED VERSION)")
    print("=" * 85)
    print(f"Satellite: {satellite.name}")
    print(f"Observer: {observer.latitude.degrees:.6f}°N, {abs(observer.longitude.degrees):.6f}°W")
    print(f"Panel offset from Sun-perpendicular: {panel_offset_angle}°")
    
    # TEST CASE: Verify specific time that should produce a flare
    test_time = ts.utc(2025, 10, 19, 13, 31, 18)
    test_result = test_specific_time(satellite, observer, test_time, panel_offset_angle, ts, eph)
    
    # Time window for general scanning
    start_time = ts.utc(2025, 10, 19, 13, 30)
    end_time = ts.utc(2025, 10, 19, 13, 34)
    
    # Predict flares
    flare_events = predict_satellite_flares(
        satellite=satellite,
        observer=observer,
        start_time=start_time,
        end_time=end_time,
        panel_offset_angle=panel_offset_angle,
        time_step_seconds=1,  # 1-second resolution for accuracy
        angle_threshold=2.0,   # Within 2 degrees of perfect reflection
        min_elevation=10.0     # At least 10° above horizon
    )
    
    # Display results
    print_flare_results(flare_events)
    print("=" * 85)
    
    # Summary
    print("\nTEST SUMMARY:")
    print("-" * 85)
    if test_result['test_passed']:
        print("✓ Test case PASSED: Flare detected at expected time")
    else:
        print("✗ Test case FAILED: Flare not detected at expected time")
    
    if flare_events:
        # Check if test time is in the results
        test_time_found = any(
            abs((event['utc_datetime'] - test_time.utc_datetime()).total_seconds()) < 1
            for event in flare_events
        )
        if test_time_found:
            print("✓ Test time appears in scan results")
        else:
            print("⚠ Test time not found in scan results (may need finer time resolution)")
    
    print(f"\nTotal flares detected: {len(flare_events)}")
    print("=" * 85)

if __name__ == "__main__":
    main()