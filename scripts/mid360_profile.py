#!/usr/bin/env python3
"""The MID-360 as an Isaac Sim RTX lidar profile (OmniLidar USDA), from the Livox sim's pattern.

  mid360_profile.py --csv .../ros2_livox_simulation/scan_mode/mid360.csv --out mid360.usda
                    [--samples 20000] [--max-range 40.0] [--min-range 0.1]

The pattern file (index, azimuth deg, zenith deg) is what the Gazebo plugin casts
(urdf/mid360.xacro): each 10 Hz scan the next FRAME_POINTS consecutive points, every k-th
with k = ceil(FRAME_POINTS / samples). Here each such window is one emitter state, and the
sensor moves to the next state every rendered frame. Isaac 5.0 cycles only the first ~10 of
a profile's states (silently), so isaac_sim.py writes one and sets it to the next window
before each scan, which casts the plugin's rays in its order.
A ray is +x pitched by zenith - 90 deg, then yawed by azimuth (the plugin's Euler), so its
elevation is 90 - zenith. Every ray fires at time 0 of its scan, as the plugin casts the
scan at one instant; livox_sim_bridge.py gives every point the scan's stamp. The attributes
are Isaac Sim 5.0's OmniSensorGenericLidarCoreAPI, whose azimuth turns clockwise.

The pure parts import without Isaac; tests/test_mid360_profile.py covers them.
"""

import argparse
import sys

FRAME_POINTS = 20000      # pattern points in one 100 ms frame (urdf/mid360.xacro)
SCAN_RATE_HZ = 10
PRIM = 'MID360'


def read_pattern(path):
    """[(azimuth_deg, zenith_deg)] from the pattern csv, in file order."""
    pts = []
    with open(path) as f:
        header = f.readline()
        if 'Azimuth' not in header or 'Zenith' not in header:
            raise ValueError('%s: expected a Time/s,Azimuth/deg,Zenith/deg header, got %r' % (path, header.strip()))
        for n, line in enumerate(f, 2):
            cols = line.split(',')
            if len(cols) != 3:
                raise ValueError('%s:%d: expected 3 columns' % (path, n))
            pts.append((float(cols[1]), float(cols[2])))
    if len(pts) < FRAME_POINTS:
        raise ValueError('%s: %d points, less than one %d-point frame' % (path, len(pts), FRAME_POINTS))
    return pts


def stride(samples):
    if not 1 <= samples <= FRAME_POINTS:
        raise ValueError('samples %d: expected 1 to %d' % (samples, FRAME_POINTS))
    return -(-FRAME_POINTS // samples)


def ray_angles(azimuth_deg, zenith_deg):
    """(azimuth, elevation) in degrees, azimuth in [-180, 180)."""
    az = (azimuth_deg + 180.0) % 360.0 - 180.0
    return az, 90.0 - zenith_deg


def emitter_states(pattern, samples=FRAME_POINTS):
    """One [(azimuth, elevation)] per scan of the pattern, as the plugin walks it."""
    k = stride(samples)
    frames = len(pattern) // FRAME_POINTS
    return [[ray_angles(*pattern[f * FRAME_POINTS + j]) for j in range(0, FRAME_POINTS, k)]
            for f in range(frames)]


def _array(decl, values, fmt):
    return '    %s = [%s]' % (decl, ', '.join(fmt % v for v in values))


def render_usda(states, max_range=40.0, min_range=0.1):
    n = len(states[0])
    if any(len(s) != n for s in states):
        raise ValueError('emitter states differ in length')
    if not 0.0 < min_range < max_range:
        raise ValueError('ranges %g-%g m: expected 0 < min < max' % (min_range, max_range))
    schemas = ['OmniSensorGenericLidarCoreAPI'] + ['OmniSensorGenericLidarCoreEmitterStateAPI:s%03d' % i
                                                   for i in range(2, len(states) + 1)]
    out = ['#usda 1.0', '(', '    metersPerUnit = 1', '    upAxis = "Z"', '    defaultPrim = "%s"' % PRIM, ')', '',
           'def OmniLidar "%s" (' % PRIM, '    prepend apiSchemas = [%s]' % ', '.join('"%s"' % a for a in schemas), ')', '{',
           '    string omni:sensor:modelName = "LidarCore"',
           '    string omni:sensor:marketName = "Livox MID-360 (rover_sim)"',
           '    float omni:sensor:tickRate = %d.0' % SCAN_RATE_HZ,
           '    uint omni:sensor:Core:scanRateBaseHz = %d' % SCAN_RATE_HZ,
           '    uint omni:sensor:Core:reportRateBaseHz = %d' % SCAN_RATE_HZ,
           '    token omni:sensor:Core:scanType = "SOLID_STATE"',
           '    token omni:sensor:Core:rayType = "IDEALIZED"',
           '    float omni:sensor:Core:nearRangeM = %r' % min_range,
           '    float omni:sensor:Core:farRangeM = %r' % max_range,
           '    float omni:sensor:Core:rangeResolutionM = 0.01',
           '    float omni:sensor:Core:rangeAccuracyM = 0.0',
           '    float omni:sensor:Core:waveLengthNm = 905.0',
           '    token omni:sensor:Core:intensityProcessing = "NORMALIZATION"',
           '    token omni:sensor:Core:intensityMappingType = "LINEAR"',
           '    uint omni:sensor:Core:maxReturns = 1',
           '    uint omni:sensor:Core:numberOfEmitters = %d' % n,
           '    uint omni:sensor:Core:numberOfChannels = %d' % n,
           '    uint omni:sensor:Core:numLines = 1',
           '    uint[] omni:sensor:Core:numRaysPerLine = [%d]' % n,
           '    uint omni:sensor:Core:rangeCount = 1',
           '    float[] omni:sensor:Core:rangesMinM = [%r]' % min_range,
           '    float[] omni:sensor:Core:rangesMaxM = [%r]' % max_range,
           '    uint omni:sensor:Core:emitterStateCount = %d' % len(states),
           '    uint omni:sensor:Core:stateResolutionStep = 1',
           '    float omni:sensor:Core:azimuthErrorMean = 0.0',
           '    float omni:sensor:Core:azimuthErrorStd = 0.0',
           '    float omni:sensor:Core:elevationErrorMean = 0.0',
           '    float omni:sensor:Core:elevationErrorStd = 0.0']
    channels = list(range(1, n + 1))
    for i, s in enumerate(states, 1):
        key = 'omni:sensor:Core:emitterState:s%03d:' % i
        out += [_array('float[] %sazimuthDeg' % key, [-a for a, _ in s], '%.3f'),  # Isaac's turns clockwise
                _array('float[] %selevationDeg' % key, [e for _, e in s], '%.3f'),
                _array('uint[] %sfireTimeNs' % key, [0] * n, '%d'),
                _array('uint[] %schannelId' % key, channels, '%d'),
                _array('uint[] %srangeId' % key, [0] * n, '%d'),
                _array('uint[] %sbank' % key, [0] * n, '%d')]
    out.append('}')
    return '\n'.join(out) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--samples', type=int, default=FRAME_POINTS)
    ap.add_argument('--max-range', type=float, default=40.0)
    ap.add_argument('--min-range', type=float, default=0.1)
    args = ap.parse_args(argv)
    states = emitter_states(read_pattern(args.csv), args.samples)
    with open(args.out, 'w') as f:
        f.write(render_usda(states, args.max_range, args.min_range))
    print('%s: %d emitter states of %d rays (every %d of %d)'
          % (args.out, len(states), len(states[0]), stride(args.samples), FRAME_POINTS))
    return 0


if __name__ == '__main__':
    sys.exit(main())
