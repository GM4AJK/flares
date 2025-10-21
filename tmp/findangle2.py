"""
Satellite Mirror Flare Orientation Optimizer

Given a satellite, observer, and time, finds the optimal orientation of a
mirror surface to produce the maximum flare (brightest specular reflection).

This treats the reflecting surface as a simple mirror, not constrained by
solar panel requirements.

Author: A-j-K
Date: 2025-10-21
"""

from skyfield.api import load, EarthSatellite, Topos
import numpy as np
import matplotlib.pyplot as plt

def calculate_perfect_reflection_normal(satellite_pos, sun_pos, observer_pos):
    """
    Calculate the exact mirror normal required for perfect specular reflection.
    
    For perfect reflection, the mirror normal must bisect the angle between the
    incident ray (from sun) and the reflected ray (to observer).
    
    This is pure geometry - the law of reflection states:
    angle of incidence = angle of reflection, both measured from the normal.
    
    Parameters:
    - satellite_pos: Position vector of satellite relative to Earth center (km)
    - sun_pos: Position vector of Sun relative to Earth center (km)
    - observer_pos: Position vector of observer relative to Earth center (km)
    
    Returns:
    - mirror_normal: Normal vector for perfect reflection
    - sun_incident_angle: Angle between sun direction and mirror normal (degrees)
    - specular_angle: Angle between reflected ray and observer direction (degrees)
    - angle_to_sun: Angle between mirror normal and sun direction (degrees)
    - half_angle: Half of the sun-satellite-observer angle (degrees)
    """
    # Vector from Sun to satellite (incident direction)
    sun_to_sat = satellite_pos - sun_pos
    sun_to_sat_unit = sun_to_sat / np.linalg.norm(sun_to_sat)
    
    # Vector from satellite to observer (desired reflection direction)
    sat_to_obs = observer_pos - satellite_pos
    sat_to_obs_unit = sat_to_obs / np.linalg.norm(sat_to_obs)
    
    # Vector from satellite to sun (direction toward sun)
    sat_to_sun = sun_pos - satellite_pos
    sat_to_sun_unit = sat_to_sun / np.linalg.norm(sat_to_sun)
    
    # Calculate the angle between sun and observer (as seen from satellite)
    sun_obs_angle = np.degrees(np.arccos(np.clip(np.dot(sat_to_sun_unit, sat_to_obs_unit), -1.0, 1.0)))
    half_angle = sun_obs_angle / 2.0
    
    # For perfect specular reflection, the mirror normal bisects the angle
    # between the incident ray and the reflected ray.
    # Normal = normalize(incident_unit + reflected_unit)
    mirror_normal = sun_to_sat_unit + sat_to_obs_unit
    mirror_normal = mirror_normal / np.linalg.norm(mirror_normal)
    
    # Calculate sun incident angle (angle between incoming sun ray and mirror normal)
    # This should equal the reflection angle for perfect reflection
    sun_incident_angle = np.degrees(np.arccos(np.clip(-np.dot(sun_to_sat_unit, mirror_normal), -1.0, 1.0)))
    
    # Calculate angle between mirror normal and direction TO sun
    angle_to_sun = np.degrees(np.arccos(np.clip(np.dot(mirror_normal, sat_to_sun_unit), -1.0, 1.0)))
    
    # Verify this creates perfect reflection using law of reflection
    reflection_vector = sun_to_sat_unit - 2 * np.dot(sun_to_sat_unit, mirror_normal) * mirror_normal
    cos_angle = np.dot(reflection_vector, sat_to_obs_unit)
    specular_angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    
    # Calculate the reflection angle (should equal incident angle)
    reflection_angle = np.degrees(np.arccos(np.clip(-np.dot(reflection_vector, mirror_normal), -1.0, 1.0)))
    
    return {
        'mirror_normal': mirror_normal,
        'sun_incident_angle': sun_incident_angle,
        'reflection_angle': reflection_angle,
        'specular_angle': specular_angle,
        'angle_to_sun': angle_to_sun,
        'half_angle': half_angle,
        'sun_obs_angle': sun_obs_angle
    }

def convert_normal_to_local_angles(mirror_normal, satellite_pos):
    """
    Convert a mirror normal vector to azimuth/elevation in satellite's local frame.
    
    Parameters:
    - mirror_normal: Normal vector in geocentric coordinates
    - satellite_pos: Satellite position vector in geocentric coordinates (km)
    
    Returns:
    - azimuth: Angle in degrees (0°=East, 90°=North, 180°=West, 270°=South)
    - elevation: Angle in degrees (-90°=Nadir, 0°=Tangent to Earth, +90°=Zenith)
    """
    # Local coordinate system at satellite position
    # Z-axis: radial direction pointing away from Earth center (zenith)
    zenith = satellite_pos / np.linalg.norm(satellite_pos)
    
    # X-axis: pointing eastward
    north_pole = np.array([0, 0, 1])
    east = np.cross(north_pole, zenith)
    
    if np.linalg.norm(east) < 0.001:
        # Near poles - use arbitrary perpendicular direction
        east = np.array([1, 0, 0])
        east = east - np.dot(east, zenith) * zenith
    
    east = east / np.linalg.norm(east)
    
    # Y-axis: northward (completes right-handed system)
    north = np.cross(zenith, east)
    north = north / np.linalg.norm(north)
    
    # Transform mirror normal to local coordinates
    local_x = np.dot(mirror_normal, east)
    local_y = np.dot(mirror_normal, north)
    local_z = np.dot(mirror_normal, zenith)
    
    # Calculate elevation (angle from horizontal plane)
    elevation = np.degrees(np.arcsin(np.clip(local_z, -1.0, 1.0)))
    
    # Calculate azimuth (angle in horizontal plane)
    azimuth = np.degrees(np.arctan2(local_y, local_x))
    if azimuth < 0:
        azimuth += 360
    
    return azimuth, elevation

def calculate_specular_angle_with_rotation(satellite_pos, sun_pos, observer_pos, 
                                           azimuth_deg, elevation_deg):
    """
    Calculate the specular reflection angle for a given mirror orientation.
    
    The mirror orientation is defined by two angles:
    - azimuth: rotation around the satellite's zenith-nadir axis
    - elevation: tilt angle from horizontal plane
    
    Parameters:
    - satellite_pos: Position vector of satellite relative to Earth center (km)
    - sun_pos: Position vector of Sun relative to Earth center (km)
    - observer_pos: Position vector of observer relative to Earth center (km)
    - azimuth_deg: Mirror azimuth angle in degrees (0-360)
    - elevation_deg: Mirror elevation angle in degrees (-90 to 90)
    
    Returns:
    - specular_angle: Angle between reflected ray and observer direction (degrees)
    - mirror_normal: Normal vector of the reflecting surface
    - sun_incident_angle: Angle between sun ray and mirror normal (degrees)
    - angle_to_sun: Angle between mirror normal and direction to sun (degrees)
    """
    # Convert angles to radians
    azimuth = np.radians(azimuth_deg)
    elevation = np.radians(elevation_deg)
    
    # Local coordinate system at satellite position
    zenith = satellite_pos / np.linalg.norm(satellite_pos)
    
    north_pole = np.array([0, 0, 1])
    east = np.cross(north_pole, zenith)
    
    if np.linalg.norm(east) < 0.001:
        east = np.array([1, 0, 0])
        east = east - np.dot(east, zenith) * zenith
    
    east = east / np.linalg.norm(east)
    north = np.cross(zenith, east)
    north = north / np.linalg.norm(north)
    
    # Calculate mirror normal in local coordinates
    local_normal = np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation)
    ])
    
    # Transform to global geocentric coordinates
    rotation_matrix = np.column_stack([east, north, zenith])
    mirror_normal = rotation_matrix @ local_normal
    mirror_normal = mirror_normal / np.linalg.norm(mirror_normal)
    
    # Vector from Sun to satellite (incident direction)
    sun_to_sat = satellite_pos - sun_pos
    sun_to_sat_unit = sun_to_sat / np.linalg.norm(sun_to_sat)
    
    # Vector from satellite to observer
    sat_to_obs = observer_pos - satellite_pos
    sat_to_obs_unit = sat_to_obs / np.linalg.norm(sat_to_obs)
    
    # Vector from satellite to sun
    sat_to_sun = sun_pos - satellite_pos
    sat_to_sun_unit = sat_to_sun / np.linalg.norm(sat_to_sun)
    
    # Calculate reflected ray using law of reflection
    dot_product = np.dot(sun_to_sat_unit, mirror_normal)
    reflection_vector = sun_to_sat_unit - 2 * dot_product * mirror_normal
    
    # Angle between reflection and observer direction
    cos_angle = np.dot(reflection_vector, sat_to_obs_unit)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    specular_angle = np.degrees(np.arccos(cos_angle))
    
    # Calculate sun incident angle
    sun_incident_angle = np.degrees(np.arccos(np.clip(-dot_product, -1.0, 1.0)))
    
    # Calculate angle to sun (for orientation reference)
    angle_to_sun = np.degrees(np.arccos(np.clip(np.dot(mirror_normal, sat_to_sun_unit), -1.0, 1.0)))
    
    return specular_angle, mirror_normal, sun_incident_angle, angle_to_sun

def calculate_flare_brightness(specular_angle, distance_km, reflectivity=1.0):
    """
    Calculate relative brightness of the mirror flare.
    
    For a perfect mirror, brightness depends only on:
    - How close to perfect specular reflection (very narrow peak)
    - Distance to observer (inverse square law)
    - Mirror reflectivity (assumed 1.0 for perfect mirror)
    
    Parameters:
    - specular_angle: Deviation from perfect reflection (degrees)
    - distance_km: Distance from observer to satellite (km)
    - reflectivity: Mirror reflectivity (0.0 to 1.0, default 1.0 for perfect mirror)
    
    Returns:
    - brightness: Relative brightness value (higher = brighter)
    - magnitude: Visual magnitude (lower = brighter)
    """
    # Specular reflection falls off very rapidly with angle
    # Mirrors produce extremely narrow reflection peaks (much narrower than diffuse surfaces)
    sigma = 0.5  # degrees - very narrow for mirror-like reflection
    angle_factor = np.exp(-(specular_angle**2) / (2 * sigma**2))
    
    # Distance factor (inverse square law)
    reference_distance = 800.0  # km (typical LEO satellite distance)
    distance_factor = (reference_distance / distance_km) ** 2
    
    # Total brightness
    brightness = angle_factor * distance_factor * reflectivity
    
    # Convert to magnitude (logarithmic scale)
    if brightness > 0:
        base_magnitude = -8.0  # Peak magnitude for perfect mirror at reference distance
        magnitude = base_magnitude - 2.5 * np.log10(brightness)
    else:
        magnitude = 99.0  # Not visible
    
    return brightness, magnitude

def optimize_mirror_orientation(satellite_pos, sun_pos, observer_pos, distance_km):
    """
    Find the optimal mirror orientation for maximum flare brightness.
    
    Calculates the theoretically perfect orientation for specular reflection.
    
    Parameters:
    - satellite_pos: Satellite position vector (km, geocentric)
    - sun_pos: Sun position vector (km, geocentric)
    - observer_pos: Observer position vector (km, geocentric)
    - distance_km: Distance from observer to satellite (km)
    
    Returns:
    - Dictionary containing optimal orientation and flare properties
    """
    print("  Calculating perfect mirror reflection geometry...")
    
    # Calculate the exact mirror normal for perfect reflection
    reflection_data = calculate_perfect_reflection_normal(
        satellite_pos, sun_pos, observer_pos
    )
    
    mirror_normal = reflection_data['mirror_normal']
    sun_inc_angle = reflection_data['sun_incident_angle']
    spec_angle = reflection_data['specular_angle']
    angle_to_sun = reflection_data['angle_to_sun']
    half_angle = reflection_data['half_angle']
    
    print(f"    Perfect reflection normal calculated")
    print(f"    Specular angle: {spec_angle:.8f}° (should be ~0 for perfect reflection)")
    print(f"    Sun incident angle: {sun_inc_angle:.2f}°")
    print(f"    Reflection angle: {reflection_data['reflection_angle']:.2f}° (should equal incident)")
    print(f"    Angle to sun (normal vs sun direction): {angle_to_sun:.2f}°")
    print(f"    Sun-Satellite-Observer half-angle: {half_angle:.2f}°")
    
    # Convert to local azimuth/elevation angles
    azimuth, elevation = convert_normal_to_local_angles(mirror_normal, satellite_pos)
    
    print(f"    Optimal azimuth: {azimuth:.2f}°")
    print(f"    Optimal elevation: {elevation:.2f}°")
    
    # Calculate brightness
    brightness, magnitude = calculate_flare_brightness(spec_angle, distance_km)
    
    print(f"    Predicted magnitude: {magnitude:.2f}")
    print(f"    Relative brightness: {brightness:.6e}")
    
    # Determine mirror orientation relative to key directions
    zenith = satellite_pos / np.linalg.norm(satellite_pos)
    angle_to_zenith = np.degrees(np.arccos(np.clip(np.dot(mirror_normal, zenith), -1.0, 1.0)))
    
    nadir = -zenith
    angle_to_nadir = np.degrees(np.arccos(np.clip(np.dot(mirror_normal, nadir), -1.0, 1.0)))
    
    return {
        'azimuth': azimuth,
        'elevation': elevation,
        'brightness': brightness,
        'magnitude': magnitude,
        'specular_angle': spec_angle,
        'sun_incident_angle': sun_inc_angle,
        'reflection_angle': reflection_data['reflection_angle'],
        'angle_to_sun': angle_to_sun,
        'angle_to_zenith': angle_to_zenith,
        'angle_to_nadir': angle_to_nadir,
        'half_angle': half_angle,
        'sun_obs_angle': reflection_data['sun_obs_angle'],
        'mirror_normal': mirror_normal,
        'optimization_failed': False
    }

def scan_orientation_space(satellite_pos, sun_pos, observer_pos, distance_km,
                           azimuth_range=(0, 360), elevation_range=(-90, 90),
                           azimuth_steps=72, elevation_steps=36):
    """
    Scan through all possible mirror orientations to create visualization data.
    
    Parameters:
    - satellite_pos, sun_pos, observer_pos: Position vectors (km, geocentric)
    - distance_km: Distance from observer to satellite (km)
    - azimuth_range: Range of azimuth angles to scan (degrees)
    - elevation_range: Range of elevation angles to scan (degrees)
    - azimuth_steps: Number of azimuth samples
    - elevation_steps: Number of elevation samples
    
    Returns:
    - azimuths: Array of azimuth values
    - elevations: Array of elevation values
    - magnitudes: 2D array of magnitude values
    - specular_angles: 2D array of specular angles
    - angles_to_sun: 2D array of angles to sun
    """
    azimuths = np.linspace(azimuth_range[0], azimuth_range[1], azimuth_steps)
    elevations = np.linspace(elevation_range[0], elevation_range[1], elevation_steps)
    
    magnitudes = np.zeros((len(elevations), len(azimuths)))
    specular_angles = np.zeros((len(elevations), len(azimuths)))
    angles_to_sun = np.zeros((len(elevations), len(azimuths)))
    
    for i, elev in enumerate(elevations):
        for j, azim in enumerate(azimuths):
            try:
                spec_angle, _, sun_inc_angle, angle_to_sun = calculate_specular_angle_with_rotation(
                    satellite_pos, sun_pos, observer_pos, azim, elev
                )
                _, magnitude = calculate_flare_brightness(spec_angle, distance_km)
                
                magnitudes[i, j] = magnitude
                specular_angles[i, j] = spec_angle
                angles_to_sun[i, j] = angle_to_sun
            except Exception:
                magnitudes[i, j] = 99.0
                specular_angles[i, j] = 180.0
                angles_to_sun[i, j] = 90.0
    
    return azimuths, elevations, magnitudes, specular_angles, angles_to_sun

def plot_orientation_heatmap(azimuths, elevations, magnitudes, angles_to_sun, optimal_result):
    """
    Create visualization showing flare brightness for all mirror orientations.
    
    Parameters:
    - azimuths: Array of azimuth values
    - elevations: Array of elevation values
    - magnitudes: 2D array of magnitude values
    - angles_to_sun: 2D array of angles to sun
    - optimal_result: Dictionary with optimal orientation data
    
    Returns:
    - fig: Matplotlib figure object
    """
    # Clip magnitudes for better visualization
    magnitudes_clipped = np.clip(magnitudes, -10, 10)
    
    fig = plt.figure(figsize=(20, 6))
    
    # Left plot: Magnitude heatmap
    ax1 = plt.subplot(131)
    im1 = ax1.contourf(azimuths, elevations, magnitudes_clipped, levels=50, cmap='RdYlGn_r')
    
    if not optimal_result.get('optimization_failed', False):
        ax1.plot(optimal_result['azimuth'], optimal_result['elevation'], 
                 'r*', markersize=25, markeredgecolor='white', markeredgewidth=2,
                 label=f"Perfect Reflection\nAz={optimal_result['azimuth']:.1f}°, El={optimal_result['elevation']:.1f}°\nMag={optimal_result['magnitude']:.1f}")
    
    ax1.set_xlabel('Azimuth (degrees)\n0°=East, 90°=North, 180°=West, 270°=South', fontsize=11)
    ax1.set_ylabel('Elevation (degrees)\n-90°=Nadir, 0°=Horizon, +90°=Zenith', fontsize=11)
    ax1.set_title('Mirror Flare Magnitude vs Orientation\n(Lower = Brighter)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, 360)
    ax1.set_ylim(-90, 90)
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Visual Magnitude', fontsize=10)
    
    # Middle plot: Angle to Sun heatmap
    ax2 = plt.subplot(132)
    im2 = ax2.contourf(azimuths, elevations, angles_to_sun, levels=50, cmap='twilight')
    
    if not optimal_result.get('optimization_failed', False):
        ax2.plot(optimal_result['azimuth'], optimal_result['elevation'], 
                 'r*', markersize=25, markeredgecolor='white', markeredgewidth=2,
                 label=f"Angle to Sun: {optimal_result['angle_to_sun']:.1f}°")
    
    ax2.set_xlabel('Azimuth (degrees)\n0°=East, 90°=North, 180°=West, 270°=South', fontsize=11)
    ax2.set_ylabel('Elevation (degrees)\n-90°=Nadir, 0°=Horizon, +90°=Zenith', fontsize=11)
    ax2.set_title('Mirror Normal Angle to Sun Direction\n(0°=Toward Sun, 90°=Perpendicular, 180°=Away)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, 360)
    ax2.set_ylim(-90, 90)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Angle to Sun (degrees)', fontsize=10)
    
    # Right plot: Polar view (upper hemisphere only)
    ax3 = plt.subplot(133, projection='polar')
    
    # Filter for positive elevations only
    pos_elev_mask = elevations >= 0
    elevations_pos = elevations[pos_elev_mask]
    magnitudes_pos = magnitudes_clipped[pos_elev_mask, :]
    
    theta = np.radians(azimuths)
    r = elevations_pos
    THETA, R = np.meshgrid(theta, r)
    
    im3 = ax3.contourf(THETA, R, magnitudes_pos, levels=50, cmap='RdYlGn_r')
    
    if not optimal_result.get('optimization_failed', False) and optimal_result['elevation'] >= 0:
        ax3.plot(np.radians(optimal_result['azimuth']), optimal_result['elevation'],
                 'r*', markersize=25, markeredgecolor='white', markeredgewidth=2)
    
    ax3.set_theta_zero_location('N')
    ax3.set_theta_direction(-1)
    ax3.set_ylim(0, 90)
    ax3.set_title('Polar View (Upper Hemisphere)\n(Radial = Elevation)', fontsize=13, fontweight='bold')
    cbar3 = plt.colorbar(im3, ax=ax3, pad=0.1)
    cbar3.set_label('Magnitude', fontsize=10)
    
    plt.tight_layout()
    return fig

def print_geometry_info(satellite_pos, sun_pos, observer_pos):
    """Print information about the geometric configuration."""
    # Calculate key vectors
    sat_to_sun = sun_pos - satellite_pos
    sat_to_obs = observer_pos - satellite_pos
    
    sat_to_sun_unit = sat_to_sun / np.linalg.norm(sat_to_sun)
    sat_to_obs_unit = sat_to_obs / np.linalg.norm(sat_to_obs)
    
    # Angle between sun and observer as seen from satellite
    cos_angle = np.dot(sat_to_sun_unit, sat_to_obs_unit)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    
    print(f"Geometric Configuration:")
    print(f"  Sun-Satellite-Observer Angle: {angle:.2f}°")
    print(f"  Sun Distance from Satellite: {np.linalg.norm(sat_to_sun):,.0f} km")
    print(f"  Observer Distance from Satellite: {np.linalg.norm(sat_to_obs):.2f} km")
    print(f"\n  Perfect Mirror Reflection: ALWAYS POSSIBLE")
    print(f"  The mirror normal bisects the sun-satellite-observer angle.")
    print(f"  Required mirror tilt from sun direction: {angle/2:.2f}°")
    print()

def main():
    """Main function to optimize mirror orientation for maximum flare."""
    # Load timescale and ephemeris
    ts = load.timescale()
    
    print("Loading ephemeris data...")
    eph = load('de421.bsp')
    sun = eph['sun']
    earth = eph['earth']
    
    # Satellite TLE (STARLINK-32653)
    line1 = '1 62199U 24229H   25291.43054285  .00031631  00000-0  11531-2 0  9993'
    line2 = '2 62199  43.0013 347.2445 0000606 261.3848  98.6935 15.27606777 48889'
    satellite = EarthSatellite(line1, line2, 'STARLINK-32653', ts)
    
    # Observer location (San Francisco Bay Area)
    observer = Topos(latitude_degrees=37.654444, longitude_degrees=-122.473943)
    
    # Specific time for flare analysis: 2025-10-19T13:31:18Z
    time = ts.utc(2025, 10, 19, 13, 31, 18)
    
    print()
    print("=" * 80)
    print("SATELLITE MIRROR FLARE ORIENTATION OPTIMIZER")
    print("=" * 80)
    print(f"Satellite: {satellite.name}")
    print(f"Observer Location: {observer.latitude.degrees:.6f}°N, {abs(observer.longitude.degrees):.6f}°W")
    print(f"Analysis Time: {time.utc_iso()}")
    print("=" * 80)
    print()
    
    # Get geocentric positions
    observer_geocentric = earth + observer
    sat_geocentric = earth + satellite
    
    satellite_pos = sat_geocentric.at(time).position.km
    sun_pos = sun.at(time).position.km
    observer_pos = observer_geocentric.at(time).position.km
    
    # Get satellite visibility from observer (topocentric)
    difference = (satellite - observer).at(time)
    alt, az, distance = difference.altaz()
    
    print(f"Satellite Visibility:")
    print(f"  Altitude: {alt.degrees:.2f}°")
    print(f"  Azimuth: {az.degrees:.2f}°")
    print(f"  Distance: {distance.km:.2f} km")
    
    if alt.degrees < 0:
        print(f"  WARNING: Satellite is below horizon!")
    elif alt.degrees < 10:
        print(f"  Note: Satellite is low on horizon (atmospheric effects may be significant)")
    
    print()
    
    # Get sun position from observer
    sun_topocentric = (sun - observer_geocentric).at(time)
    sun_alt, sun_az, sun_dist = sun_topocentric.altaz()
    
    sun_condition = "above horizon (daylight)" if sun_alt.degrees > 0 else \
                   "civil twilight" if sun_alt.degrees > -6 else \
                   "nautical twilight" if sun_alt.degrees > -12 else \
                   "astronomical twilight" if sun_alt.degrees > -18 else \
                   "night (astronomical darkness)"
    
    print(f"Sun Position:")
    print(f"  Altitude: {sun_alt.degrees:.2f}°")
    print(f"  Azimuth: {sun_az.degrees:.2f}°")
    print(f"  Condition: {sun_condition}")
    
    if sun_alt.degrees > -6:
        print(f"  Note: Bright sky conditions - flare may be difficult to observe")
    
    print()
    
    # Print geometric configuration
    print_geometry_info(satellite_pos, sun_pos, observer_pos)
    
    # Optimize mirror orientation
    print("Calculating Optimal Mirror Orientation...")
    print("-" * 80)
    
    optimal = optimize_mirror_orientation(
        satellite_pos, sun_pos, observer_pos, distance.km
    )
    
    print()
    print("=" * 80)
    print("OPTIMAL MIRROR ORIENTATION FOR PERFECT SPECULAR REFLECTION")
    print("=" * 80)
    print()
    print(f"Mirror Orientation (Satellite Local Frame):")
    print(f"  Azimuth: {optimal['azimuth']:.2f}°")
    print(f"    (0°=East, 90°=North, 180°=West, 270°=South)")
    print(f"  Elevation: {optimal['elevation']:.2f}°")
    print(f"    (-90°=Nadir/downward, 0°=Horizontal, +90°=Zenith/upward)")
    print()
    print(f"Mirror Orientation Relative to Key Directions:")
    print(f"  Angle to Sun: {optimal['angle_to_sun']:.2f}°")
    print(f"    (Angle between mirror normal and direction toward sun)")
    print(f"  Angle to Zenith: {optimal['angle_to_zenith']:.2f}°")
    print(f"    (Angle between mirror normal and upward direction)")
    print(f"  Angle to Nadir: {optimal['angle_to_nadir']:.2f}°")
    print(f"    (Angle between mirror normal and downward direction)")
    print()
    print(f"Reflection Geometry (Law of Reflection):")
    print(f"  Sun-Satellite-Observer Angle: {optimal['sun_obs_angle']:.2f}°")
    print(f"  Half-Angle (mirror tilt): {optimal['half_angle']:.2f}°")
    print(f"  Sun Incident Angle: {optimal['sun_incident_angle']:.2f}°")
    print(f"  Reflection Angle: {optimal['reflection_angle']:.2f}°")
    print(f"  Verification: Incident = Reflection? {abs(optimal['sun_incident_angle'] - optimal['reflection_angle']) < 0.01}")
    print(f"  Specular Angle: {optimal['specular_angle']:.8f}°")
    print(f"    (Deviation from perfect reflection - should be ~0°)")
    print()
    print(f"Predicted Mirror Flare:")
    print(f"  Visual Magnitude: {optimal['magnitude']:.2f}")
    print(f"    (Lower = brighter; Sun=-26.7, Venus=-4.6, Iridium flare=-8)")
    print(f"  Relative Brightness: {optimal['brightness']:.6e}")
    print()
    print(f"Mirror Normal Vector (Geocentric Cartesian, unit vector):")
    print(f"  X: {optimal['mirror_normal'][0]:>9.6f}")
    print(f"  Y: {optimal['mirror_normal'][1]:>9.6f}")
    print(f"  Z: {optimal['mirror_normal'][2]:>9.6f}")
    print(f"  Magnitude: {np.linalg.norm(optimal['mirror_normal']):.6f}")
    print()
    
    # Interpretation
    print("Physical Interpretation:")
    if optimal['angle_to_sun'] < 45:
        print(f"  The mirror faces predominantly TOWARD the sun ({optimal['angle_to_sun']:.1f}° off)")
    elif optimal['angle_to_sun'] < 90:
        print(f"  The mirror faces somewhat toward the sun ({optimal['angle_to_sun']:.1f}° off)")
    elif optimal['angle_to_sun'] < 135:
        print(f"  The mirror faces somewhat away from the sun ({optimal['angle_to_sun']:.1f}° from sun)")
    else:
        print(f"  The mirror faces predominantly AWAY from the sun ({optimal['angle_to_sun']:.1f}° from sun)")
    
    if optimal['elevation'] > 45:
        print(f"  The mirror tilts upward (elevation {optimal['elevation']:.1f}°)")
    elif optimal['elevation'] > -45:
        print(f"  The mirror is nearly horizontal (elevation {optimal['elevation']:.1f}°)")
    else:
        print(f"  The mirror tilts downward (elevation {optimal['elevation']:.1f}°)")
    
    print()
    print("=" * 80)
    print()
    
    # Scan orientation space for visualization
    print("Generating Orientation Heatmaps...")
    print("  (This may take a minute...)")
    
    azimuths, elevations, magnitudes, spec_angles, angles_to_sun = scan_orientation_space(
        satellite_pos, sun_pos, observer_pos, distance.km,
        azimuth_steps=180,  # High resolution for smooth plots
        elevation_steps=90   # Cover full range from -90 to +90
    )
    
    print("  Creating visualization...")
    
    # Create and save heatmap
    fig = plot_orientation_heatmap(azimuths, elevations, magnitudes, angles_to_sun, optimal)
    plt.savefig('mirror_flare_orientation_heatmap.png', dpi=150, bbox_inches='tight')
    print()
    print("Heatmap saved as 'mirror_flare_orientation_heatmap.png'")
    print()
    print("=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print("\nSummary:")
    print(f"For a mirror at the satellite to reflect sunlight to the observer,")
    print(f"it must be tilted at {optimal['half_angle']:.2f}° from the sun direction,")
    print(f"bisecting the {optimal['sun_obs_angle']:.2f}° angle between sun and observer.")
    print()
    
    # Display plot
    plt.show()

if __name__ == "__main__":
    main()