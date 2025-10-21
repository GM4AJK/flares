"""
Satellite Flare Orientation Optimizer
Given a satellite, observer, and time, finds the optimal orientation of the
reflecting surface to produce the maximum flare (brightest reflection).

This version removes the constraint that panels must face the sun,
allowing any orientation for maximum specular reflection.

Author: A-j-K
Date: 2025-10-21
"""

from skyfield.api import load, EarthSatellite, Topos
import numpy as np
import matplotlib.pyplot as plt

def calculate_perfect_reflection_normal(satellite_pos, sun_pos, observer_pos):
    """
    Calculate the exact panel normal required for perfect specular reflection.
    
    For perfect reflection, the normal must bisect the angle between the
    incident ray (from sun) and the reflected ray (to observer).
    
    Parameters:
    - satellite_pos: Position vector of satellite relative to Earth center (km)
    - sun_pos: Position vector of Sun relative to Earth center (km)
    - observer_pos: Position vector of observer relative to Earth center (km)
    
    Returns:
    - panel_normal: Normal vector for perfect reflection
    - sun_incident_angle: Angle between sun direction and panel normal (degrees)
    - specular_angle: Angle between reflected ray and observer direction (degrees)
    - angle_to_sun: Angle between panel normal and sun direction (degrees)
    """
    # Vector from Sun to satellite (incident direction)
    sun_to_sat = satellite_pos - sun_pos
    sun_to_sat_unit = sun_to_sat / np.linalg.norm(sun_to_sat)
    
    # Vector from satellite to observer (reflection direction)
    sat_to_obs = observer_pos - satellite_pos
    sat_to_obs_unit = sat_to_obs / np.linalg.norm(sat_to_obs)
    
    # Vector from satellite to sun (direction toward sun)
    sat_to_sun = sun_pos - satellite_pos
    sat_to_sun_unit = sat_to_sun / np.linalg.norm(sat_to_sun)
    
    # For perfect specular reflection, the normal bisects the angle
    # between the incident and reflected rays
    # Normal = normalize(incident_unit + reflected_unit)
    panel_normal = sun_to_sat_unit + sat_to_obs_unit
    panel_normal = panel_normal / np.linalg.norm(panel_normal)
    
    # Calculate sun incident angle (angle between incoming sun ray and panel normal)
    sun_incident_angle = np.degrees(np.arccos(np.clip(-np.dot(sun_to_sat_unit, panel_normal), -1.0, 1.0)))
    
    # Calculate angle between panel normal and direction TO sun
    angle_to_sun = np.degrees(np.arccos(np.clip(np.dot(panel_normal, sat_to_sun_unit), -1.0, 1.0)))
    
    # Verify this creates perfect reflection
    reflection_vector = sun_to_sat_unit - 2 * np.dot(sun_to_sat_unit, panel_normal) * panel_normal
    cos_angle = np.dot(reflection_vector, sat_to_obs_unit)
    specular_angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    
    return panel_normal, sun_incident_angle, specular_angle, angle_to_sun

def convert_normal_to_local_angles(panel_normal, satellite_pos):
    """
    Convert a panel normal vector to azimuth/elevation in satellite's local frame.
    
    Parameters:
    - panel_normal: Normal vector in geocentric coordinates
    - satellite_pos: Satellite position vector in geocentric coordinates (km)
    
    Returns:
    - azimuth: Angle in degrees (0°=East, 90°=North, 180°=West, 270°=South)
    - elevation: Angle in degrees (-90°=Nadir, 0°=Tangent to Earth, +90°=Zenith)
    """
    # Local coordinate system at satellite position
    # Z-axis: radial direction (nadir, pointing toward Earth center)
    nadir = -satellite_pos / np.linalg.norm(satellite_pos)
    
    # X-axis: pointing eastward
    north_pole = np.array([0, 0, 1])
    east = np.cross(north_pole, nadir)
    
    if np.linalg.norm(east) < 0.001:
        east = np.array([1, 0, 0])
        east = east - np.dot(east, nadir) * nadir
    
    east = east / np.linalg.norm(east)
    
    # Y-axis: northward
    north = np.cross(nadir, east)
    north = north / np.linalg.norm(north)
    
    # Transform panel normal to local coordinates
    local_x = np.dot(panel_normal, east)
    local_y = np.dot(panel_normal, north)
    local_z = np.dot(panel_normal, -nadir)  # Negative because we want "up" not "down"
    
    # Calculate elevation (angle from tangent plane)
    elevation = np.degrees(np.arcsin(np.clip(local_z, -1.0, 1.0)))
    
    # Calculate azimuth (angle in tangent plane)
    azimuth = np.degrees(np.arctan2(local_y, local_x))
    if azimuth < 0:
        azimuth += 360
    
    return azimuth, elevation

def calculate_specular_angle_with_rotation(satellite_pos, sun_pos, observer_pos, 
                                           azimuth_deg, elevation_deg):
    """
    Calculate the specular reflection angle for a given panel orientation.
    
    The panel orientation is defined by two angles:
    - azimuth: rotation around the satellite's radial (nadir) vector
    - elevation: tilt angle from tangent plane
    
    Parameters:
    - satellite_pos: Position vector of satellite relative to Earth center (km)
    - sun_pos: Position vector of Sun relative to Earth center (km)
    - observer_pos: Position vector of observer relative to Earth center (km)
    - azimuth_deg: Panel azimuth angle in degrees (0-360)
    - elevation_deg: Panel elevation angle in degrees (-90 to 90)
    
    Returns:
    - specular_angle: Angle between reflected ray and observer direction (degrees)
    - panel_normal: Normal vector of the reflecting surface
    - sun_incident_angle: Angle between sun ray and panel normal (degrees)
    - angle_to_sun: Angle between panel normal and direction to sun (degrees)
    """
    # Convert angles to radians
    azimuth = np.radians(azimuth_deg)
    elevation = np.radians(elevation_deg)
    
    # Local coordinate system at satellite position
    nadir = -satellite_pos / np.linalg.norm(satellite_pos)
    
    north_pole = np.array([0, 0, 1])
    east = np.cross(north_pole, nadir)
    
    if np.linalg.norm(east) < 0.001:
        east = np.array([1, 0, 0])
        east = east - np.dot(east, nadir) * nadir
    
    east = east / np.linalg.norm(east)
    north = np.cross(nadir, east)
    north = north / np.linalg.norm(north)
    
    # Calculate panel normal in local coordinates
    local_normal = np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation)
    ])
    
    # Transform to global geocentric coordinates
    rotation_matrix = np.column_stack([east, north, -nadir])
    panel_normal = rotation_matrix @ local_normal
    panel_normal = panel_normal / np.linalg.norm(panel_normal)
    
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
    dot_product = np.dot(sun_to_sat_unit, panel_normal)
    reflection_vector = sun_to_sat_unit - 2 * dot_product * panel_normal
    
    # Angle between reflection and observer direction
    cos_angle = np.dot(reflection_vector, sat_to_obs_unit)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    specular_angle = np.degrees(np.arccos(cos_angle))
    
    # Calculate sun incident angle
    sun_incident_angle = np.degrees(np.arccos(np.clip(-dot_product, -1.0, 1.0)))
    
    # Calculate angle to sun (for solar panel orientation reference)
    angle_to_sun = np.degrees(np.arccos(np.clip(np.dot(panel_normal, sat_to_sun_unit), -1.0, 1.0)))
    
    return specular_angle, panel_normal, sun_incident_angle, angle_to_sun

def calculate_flare_brightness(specular_angle, distance_km):
    """
    Calculate relative brightness of the flare.
    
    Note: This version does NOT include sun illumination constraints.
    We're purely calculating specular reflection geometry.
    
    Parameters:
    - specular_angle: Deviation from perfect reflection (degrees)
    - distance_km: Distance from observer to satellite (km)
    
    Returns:
    - brightness: Relative brightness value (higher = brighter)
    - magnitude: Visual magnitude (lower = brighter)
    """
    # Specular reflection falls off very rapidly with angle
    sigma = 0.5  # degrees - controls how narrow the flare is
    angle_factor = np.exp(-(specular_angle**2) / (2 * sigma**2))
    
    # Distance factor (inverse square law)
    reference_distance = 800.0  # km
    distance_factor = (reference_distance / distance_km) ** 2
    
    # Total brightness (no sun illumination factor)
    brightness = angle_factor * distance_factor
    
    # Convert to magnitude (logarithmic scale)
    if brightness > 0:
        base_magnitude = -8.0  # Peak magnitude for perfect conditions
        magnitude = base_magnitude - 2.5 * np.log10(brightness)
    else:
        magnitude = 99.0  # Not visible
    
    return brightness, magnitude

def optimize_panel_orientation(satellite_pos, sun_pos, observer_pos, distance_km):
    """
    Find the optimal panel orientation for maximum flare brightness.
    
    Calculates the theoretically perfect orientation for specular reflection.
    
    Parameters:
    - satellite_pos: Satellite position vector (km, geocentric)
    - sun_pos: Sun position vector (km, geocentric)
    - observer_pos: Observer position vector (km, geocentric)
    - distance_km: Distance from observer to satellite (km)
    
    Returns:
    - Dictionary containing optimal orientation and flare properties
    """
    print("  Calculating perfect reflection geometry...")
    
    # Calculate the exact panel normal for perfect reflection
    panel_normal, sun_inc_angle, spec_angle, angle_to_sun = calculate_perfect_reflection_normal(
        satellite_pos, sun_pos, observer_pos
    )
    
    print(f"    Perfect reflection normal calculated")
    print(f"    Specular angle: {spec_angle:.6f}° (should be ~0 for perfect reflection)")
    print(f"    Sun incident angle: {sun_inc_angle:.2f}°")
    print(f"    Angle to sun (normal vs sun direction): {angle_to_sun:.2f}°")
    
    # Convert to local azimuth/elevation angles
    azimuth, elevation = convert_normal_to_local_angles(panel_normal, satellite_pos)
    
    print(f"    Optimal azimuth: {azimuth:.2f}°")
    print(f"    Optimal elevation: {elevation:.2f}°")
    
    # Calculate brightness (without sun illumination constraint)
    brightness, magnitude = calculate_flare_brightness(spec_angle, distance_km)
    
    print(f"    Predicted magnitude: {magnitude:.2f}")
    print(f"    Relative brightness: {brightness:.6e}")
    
    # Determine if panel faces toward or away from sun
    if angle_to_sun < 90:
        panel_orientation = "toward sun (front-lit)"
    else:
        panel_orientation = "away from sun (back-lit)"
    
    return {
        'azimuth': azimuth,
        'elevation': elevation,
        'brightness': brightness,
        'magnitude': magnitude,
        'specular_angle': spec_angle,
        'sun_incident_angle': sun_inc_angle,
        'angle_to_sun': angle_to_sun,
        'panel_normal': panel_normal,
        'panel_orientation': panel_orientation,
        'optimization_failed': False
    }

def scan_orientation_space(satellite_pos, sun_pos, observer_pos, distance_km,
                           azimuth_range=(0, 360), elevation_range=(-90, 90),
                           azimuth_steps=72, elevation_steps=36):
    """
    Scan through all possible panel orientations to create visualization data.
    
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
    Create visualization showing flare brightness for all panel orientations.
    
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
    im1 = ax1.contourf(azimuths, elevations, magnitudes_clipped, levels=30, cmap='RdYlGn_r')
    
    if not optimal_result.get('optimization_failed', False):
        ax1.plot(optimal_result['azimuth'], optimal_result['elevation'], 
                 'r*', markersize=25, markeredgecolor='black', markeredgewidth=1.5,
                 label=f"Optimal: Az={optimal_result['azimuth']:.1f}°, El={optimal_result['elevation']:.1f}°\nMag={optimal_result['magnitude']:.1f}")
    
    ax1.set_xlabel('Azimuth (degrees)\n0°=East, 90°=North, 180°=West, 270°=South', fontsize=11)
    ax1.set_ylabel('Elevation (degrees)\n-90°=Nadir, 0°=Tangent, +90°=Zenith', fontsize=11)
    ax1.set_title('Flare Magnitude vs Panel Orientation\n(Lower = Brighter)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, 360)
    ax1.set_ylim(-90, 90)
    cbar1 = plt.colorbar(im1, ax=ax1, label='Magnitude')
    cbar1.set_label('Visual Magnitude', fontsize=10)
    
    # Middle plot: Angle to Sun heatmap
    ax2 = plt.subplot(132)
    im2 = ax2.contourf(azimuths, elevations, angles_to_sun, levels=30, cmap='twilight')
    
    if not optimal_result.get('optimization_failed', False):
        ax2.plot(optimal_result['azimuth'], optimal_result['elevation'], 
                 'r*', markersize=25, markeredgecolor='black', markeredgewidth=1.5,
                 label=f"Angle to Sun: {optimal_result['angle_to_sun']:.1f}°")
    
    ax2.set_xlabel('Azimuth (degrees)\n0°=East, 90°=North, 180°=West, 270°=South', fontsize=11)
    ax2.set_ylabel('Elevation (degrees)\n-90°=Nadir, 0°=Tangent, +90°=Zenith', fontsize=11)
    ax2.set_title('Angle Between Panel Normal and Sun Direction\n(0°=Toward Sun, 180°=Away)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, 360)
    ax2.set_ylim(-90, 90)
    cbar2 = plt.colorbar(im2, ax=ax2, label='Angle to Sun')
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
    
    im3 = ax3.contourf(THETA, R, magnitudes_pos, levels=30, cmap='RdYlGn_r')
    
    if not optimal_result.get('optimization_failed', False) and optimal_result['elevation'] >= 0:
        ax3.plot(np.radians(optimal_result['azimuth']), optimal_result['elevation'],
                 'r*', markersize=25, markeredgecolor='black', markeredgewidth=1.5)
    
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
    print(f"  Perfect Reflection: ALWAYS POSSIBLE (any geometry)")
    print(f"  Note: Panel orientation can produce perfect specular reflection")
    print()

def main():
    """Main function to optimize satellite panel orientation for maximum flare."""
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
    print("SATELLITE FLARE ORIENTATION OPTIMIZER")
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
        print(f"  Note: Sky may be too bright for optimal flare observation")
    
    print()
    
    # Print geometric configuration
    print_geometry_info(satellite_pos, sun_pos, observer_pos)
    
    # Optimize panel orientation
    print("Calculating Optimal Panel Orientation...")
    print("-" * 80)
    
    optimal = optimize_panel_orientation(
        satellite_pos, sun_pos, observer_pos, distance.km
    )
    
    print()
    print("=" * 80)
    print("OPTIMAL ORIENTATION FOR PERFECT SPECULAR REFLECTION")
    print("=" * 80)
    print()
    print(f"Panel Orientation (Satellite Local Frame):")
    print(f"  Azimuth: {optimal['azimuth']:.2f}°")
    print(f"    (0°=East, 90°=North, 180°=West, 270°=South)")
    print(f"  Elevation: {optimal['elevation']:.2f}°")
    print(f"    (-90°=Nadir/toward Earth, 0°=Tangent, +90°=Zenith/away from Earth)")
    print()
    print(f"Panel Orientation Relative to Sun:")
    print(f"  Angle to Sun: {optimal['angle_to_sun']:.2f}°")
    print(f"    (Angle between panel normal and direction toward sun)")
    print(f"    (0° = panel faces directly toward sun)")
    print(f"    (90° = panel edge-on to sun)")
    print(f"    (180° = panel faces directly away from sun)")
    print(f"  Configuration: {optimal['panel_orientation']}")
    print()
    print(f"Reflection Geometry:")
    print(f"  Specular Angle: {optimal['specular_angle']:.6f}°")
    print(f"    (Deviation from perfect reflection - should be ~0°)")
    print(f"  Sun Incident Angle: {optimal['sun_incident_angle']:.2f}°")
    print(f"    (Angle between incoming sun ray and panel normal)")
    print()
    print(f"Predicted Flare Brightness:")
    print(f"  Visual Magnitude: {optimal['magnitude']:.2f}")
    print(f"    (Lower = brighter; Sun=-26.7, Venus=-4.6, naked eye limit~6)")
    print(f"  Relative Brightness: {optimal['brightness']:.6e}")
    print()
    print(f"Panel Normal Vector (Geocentric Cartesian):")
    print(f"  X: {optimal['panel_normal'][0]:>9.6f}")
    print(f"  Y: {optimal['panel_normal'][1]:>9.6f}")
    print(f"  Z: {optimal['panel_normal'][2]:>9.6f}")
    print(f"  Magnitude: {np.linalg.norm(optimal['panel_normal']):.6f}")
    print()
    
    if optimal['angle_to_sun'] > 90:
        print(f"NOTE: The optimal reflection requires the panel to face AWAY from the sun.")
        print(f"      This configuration would not receive direct sunlight.")
        print(f"      For solar panels, this is geometrically possible but not practical.")
    else:
        print(f"NOTE: The optimal reflection has the panel facing TOWARD the sun.")
        print(f"      This configuration receives direct sunlight (suitable for solar panels).")
    
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
    plt.savefig('flare_orientation_heatmap.png', dpi=150, bbox_inches='tight')
    print()
    print("Heatmap saved as 'flare_orientation_heatmap.png'")
    print()
    print("=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    
    # Display plot
    plt.show()

if __name__ == "__main__":
    main()