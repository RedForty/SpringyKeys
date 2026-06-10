"""
Runs springy filter functions on selected keyframes in the graph editor.

This tool applies physics-based spring damping and smoothing filters to dense
animation curves. Each filter ("solver") is registered in the SOLVERS table
with its own tuning sliders, preset bank, and a pure ``process`` function, so
new filters can be added without touching the UI or preset plumbing.

The spring/damper maths are ports of Daniel Holden's "Spring-It-On" article:
https://theorangeduck.com/page/spring-roll-call
"""

import math
from functools import partial
from maya import cmds, mel


# Globals ------------------------------------------------------------------- #

UNDO_OPEN = False
KEY_DATA = {}
SELECTION_FINGERPRINT = {}
DELTA_TIME = 1 / 30.0      # Shared simulation timestep (frame time) for solvers
LAST_SOLVER = None         # Last solver applied; re-run when Delta Time changes
TIMELINE = mel.eval('string $tmpString=$gPlayBackSlider') # pylint: disable=E1111
GRAPH_EDITOR = 'graphEditor1GraphEd'

# UI control names, populated by ui().
DT_SLIDER = None                  # The shared Delta Time slider
SLIDER_CONTROLS = {}              # {solver_key: {param_name: slider_control}}
PRESET_BUTTONS = {}               # {solver_key: [button, ...]}

# Undo/redo slider sync. Slider values are pure UI state and are not part of
# Maya's undo queue, so we track them ourselves and mirror them when the user
# undoes/redoes one of our operations (chunks tagged with CHUNK_NAME).
CHUNK_NAME = 'SpringyKeys'
RESTING_STATE = None       # slider values at rest (before the next operation)
SLIDER_UNDO_STACK = []     # resting states, one per applied operation
SLIDER_REDO_STACK = []

PRESET_COUNT = 3


# Selection / keyframe IO --------------------------------------------------- #

def is_equal(lst: list):
    """Check if all items in a list are equal.

    :param list lst: List of values to compare
    :return: True if all items are equal or list is empty, False otherwise
    :rtype: bool
    """
    return not lst or lst.count(lst[0]) == len(lst)


def get_current_selection_fingerprint():
    """Get fingerprint of current Graph Editor keyframe selection.

    :return: Dictionary mapping curve names to (start_time, end_time) tuples.
             Empty dict if no valid selection.
    :rtype: dict
    """
    if not cmds.animCurveEditor(GRAPH_EDITOR, exists=True):
        return {}

    if not cmds.animCurveEditor(GRAPH_EDITOR, q=True, areCurvesSelected=True):
        return {}

    selected_curves = cmds.keyframe(q=True, selected=True, name=True) or []
    if not selected_curves:
        return {}

    fingerprint = {}

    for curve in selected_curves:
        selected_index = cmds.keyframe(curve, q=True, selected=True, indexValue=True)
        if len(selected_index) <= 1:
            continue  # Skip single keyframes

        selected_times = cmds.keyframe(curve, q=True, selected=True)
        start_time = selected_times[0]
        end_time = selected_times[-1]

        fingerprint[curve] = (start_time, end_time)

    return fingerprint


def selection_changed(SELECTION_FINGERPRINT: dict):
    """Check if current selection differs from stored fingerprint.

    :param dict SELECTION_FINGERPRINT: Stored fingerprint to compare against
    :return: True if selection has changed, False if unchanged
    :rtype: bool
    """
    current_fingerprint = get_current_selection_fingerprint()
    return current_fingerprint != SELECTION_FINGERPRINT


def get_selected_keyframe_data():
    """Extract keyframe data from currently selected curves in Graph Editor.

    :return: Dictionary mapping curve names to their keyframe data:
             {'curve_name': {'times': [...], 'values': [...], 'pre_value': float}}
             Returns None if no valid selection or data found
    :rtype: dict or None

    .. note::
        - Filters out single keyframe selections (need at least 2 to interpolate)
        - Skips flat curves (all values equal)
        - Uses continuous time range between first and last selected keys
        - Captures the value one frame before the selection so solvers can
          compute the incoming velocity at the start
    """
    if not cmds.animCurveEditor(GRAPH_EDITOR, exists=True):
        cmds.error("{} not found.".format(GRAPH_EDITOR))
        return  # Cannot find graph editor?

    if not cmds.animCurveEditor(GRAPH_EDITOR, q=True, areCurvesSelected=True):
        cmds.warning("No keys selected to operate on.")
        return

    selected_curves = cmds.keyframe(q=True, selected=True, name=True) or []
    if not selected_curves:
        return None  # How did the last check not catch this?

    key_data = {}

    for curve in selected_curves:

        selected_index = cmds.keyframe(curve, q=True, selected=True, indexValue=True)
        if len(selected_index) == 1:
            continue  # Bounce

        selected_times = cmds.keyframe(curve, q=True, selected=True)

        time_range = []
        time_range.extend([float(x) for x in range(int(selected_times[0])
                          , int(selected_times[-1]) + 1)])

        value_range = []
        for time in time_range:
            value = cmds.keyframe(curve, q=True, time=(time, ), eval=True, valueChange=True)
            value_range.extend(value)

        if is_equal(value_range):
            continue  # Ignore flat curves

        # Get the values at the two frames before the first selected key so
        # solvers can compute the incoming velocity (and, for inertialization,
        # the source position/velocity) at the start of the selection.
        pre_time = time_range[0] - 1
        pre_value = cmds.keyframe(curve, q=True, time=(pre_time,), eval=True, valueChange=True)
        pre_value = pre_value[0] if pre_value else value_range[0]

        pre2_time = time_range[0] - 2
        pre2_value = cmds.keyframe(curve, q=True, time=(pre2_time,), eval=True, valueChange=True)
        pre2_value = pre2_value[0] if pre2_value else pre_value

        key_data[curve] = { "times": time_range
                          , "values": value_range
                          , "pre_value": pre_value
                          , "pre2_value": pre2_value
                          }

    if key_data:
        return key_data

    return None


def apply_values(curve: str, times: list, values: list):
    """Apply new values to keyframes on an animation curve.

    :param str curve: Name of the Maya animation curve
    :param list times: List of time values (frame numbers)
    :param list values: List of corresponding values

    .. note::
        Modifies existing keyframes at the specified times. If no keyframe
        exists at a given time, Maya will ignore it. Works on sparse curves!
    """
    for time, value in zip(times, values):
        cmds.keyframe(curve, e=True, time=(time,), valueChange=value)


# Spring / damper maths ----------------------------------------------------- #

def damping_ratio_to_stiffness(ratio: float, damping: float):
    """Convert damping ratio and damping coefficient to spring stiffness."""
    return (damping / (ratio * 2.0))**2


def damping_ratio_to_damping(ratio: float, stiffness: float):
    """Convert damping ratio and stiffness to damping coefficient."""
    return ratio * 2.0 * (stiffness ** 0.5)


def halflife_to_damping(halflife: float, eps: float=1e-5):
    """Convert halflife duration to damping coefficient.

    Shorter halflife = faster decay = higher damping coefficient.
    """
    return (4.0 * 0.69314718056) / (halflife + eps)


def halflife_to_lag(halflife: float):
    """Convert a halflife to its equivalent lag time (halflife / ln 2)."""
    return halflife / 0.69314718056


def fast_atan(x: float):
    """Fast polynomial approximation of arctangent (radians)."""
    z = abs(x)
    w = 1.0 / z if z > 1.0 else z
    y = (math.pi / 4.0) * w - w * (w - 1) * (0.2447 + 0.0663 * w)
    new_y = math.pi / 2.0 - y if z > 1.0 else y
    return math.copysign(new_y, x)


def fast_negexp(x: float):
    """Fast rational approximation of e^(-x)."""
    return 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)


def lerp(a: float, b: float, t: float):
    """Linear interpolation between two values."""
    return (1.0 - t) * a + t * b


def damper(x: float, g: float, factor: float):
    """Simple linear-interpolation damper toward a goal (frame-rate dependent)."""
    return lerp(x, g, factor)


def damper_exact(x: float, g: float, halflife: float, dt: float, eps: float=1e-5):
    """Exact exponential damper toward a goal (frame-rate independent, no overshoot).

    :param float x: Current value
    :param float g: Goal value
    :param float halflife: Time for the remaining distance to halve
    :param float dt: Time step
    :return: Damped value
    :rtype: float
    """
    return lerp(x, g, 1.0 - fast_negexp((0.69314718056 * dt) / (halflife + eps)))


def spring_damper_exact_ratio(
    x: float,
    v: float,
    x_goal: float,
    v_goal: float,
    damping_ratio: float,
    halflife: float,
    dt: float,
    eps: float=1e-5,
    ):
    """Exact spring-damper parameterized by damping ratio and halflife.

    Handles critically damped (ratio ~= 1), under damped (ratio < 1, oscillates)
    and over damped (ratio > 1) cases.

    :return: (new_position, new_velocity) after time step dt
    :rtype: tuple
    """
    g = x_goal
    q = v_goal
    d = halflife_to_damping(halflife)
    s = damping_ratio_to_stiffness(damping_ratio, d)
    c = g + (d * q) / (s + eps)
    y = d / 2.0

    if abs(s - (d * d) / 4.0) < eps:  # Critically Damped
        j0 = x - c
        j1 = v + j0 * y

        eydt = fast_negexp(y * dt)

        new_x = j0 * eydt + dt * j1 * eydt + c
        new_v = -y * j0 * eydt - y * dt * j1 * eydt + j1 * eydt

    elif s - (d * d) / 4.0 > 0.0:  # Under Damped
        w = math.sqrt(s - (d * d) / 4.0)
        j = math.sqrt((v + y * (x - c))**2 / (w * w + eps) + (x - c)**2)
        p = fast_atan((v + (x - c) * y) / (-(x - c) * w + eps))

        j = j if (x - c) > 0.0 else -j

        eydt = fast_negexp(y * dt)

        new_x = j * eydt * math.cos(w * dt + p) + c
        new_v = -y * j * eydt * math.cos(w * dt + p) - w * j * eydt * math.sin(w * dt + p)

    else:  # Over Damped
        y0 = (d + math.sqrt(d * d - 4 * s)) / 2.0
        y1 = (d - math.sqrt(d * d - 4 * s)) / 2.0
        j1 = (c * y0 - x * y0 - v) / (y1 - y0)
        j0 = x - j1 - c

        ey0dt = fast_negexp(y0 * dt)
        ey1dt = fast_negexp(y1 * dt)

        new_x = j0 * ey0dt + j1 * ey1dt + c
        new_v = -y0 * j0 * ey0dt - y1 * j1 * ey1dt

    return new_x, new_v


def critical_spring_damper_exact(x: float, v: float, x_goal: float,
                                 v_goal: float, halflife: float, dt: float):
    """Exact critically damped spring: fastest approach to goal without overshoot.

    :return: (new_position, new_velocity) after time step dt
    :rtype: tuple
    """
    d = halflife_to_damping(halflife)
    c = x_goal + (d * v_goal) / ((d * d) / 4.0)
    y = d / 2.0
    j0 = x - c
    j1 = v + j0 * y
    eydt = fast_negexp(y * dt)
    new_x = eydt * (j0 + j1 * dt) + c
    new_v = eydt * (v - j1 * y * dt)
    return new_x, new_v


def simple_spring_damper_exact(x: float, v: float, x_goal: float,
                               halflife: float, dt: float):
    """Simple critically damped spring toward a goal (goal velocity assumed 0).

    :return: (new_position, new_velocity) after time step dt
    :rtype: tuple
    """
    y = halflife_to_damping(halflife) / 2.0
    j0 = x - x_goal
    j1 = v + j0 * y
    eydt = fast_negexp(y * dt)
    new_x = eydt * (j0 + j1 * dt) + x_goal
    new_v = eydt * (v - j1 * y * dt)
    return new_x, new_v


def velocity_spring_damper_exact(x: float, v: float, xi: float, x_goal: float,
                                 v_goal: float, halflife: float, dt: float):
    """Velocity spring: drive toward a goal at a target speed ``v_goal``.

    ``xi`` is an intermediate tracker advanced toward ``x_goal`` at speed
    ``v_goal``; the spring chases a lag-compensated projection of it. Faithful
    port of Holden's velocity_spring_damper_exact.

    :return: (new_position, new_velocity, new_xi)
    :rtype: tuple
    """
    x_diff = (1.0 if (x_goal - xi) > 0.0 else -1.0) * v_goal

    t_goal_future = halflife_to_lag(halflife)
    if abs(x_goal - xi) > t_goal_future * v_goal:
        x_goal_future = xi + x_diff * t_goal_future
    else:
        x_goal_future = x_goal

    x, v = simple_spring_damper_exact(x, v, x_goal_future, halflife, dt)

    if abs(x_goal - xi) > dt * v_goal:
        xi = xi + x_diff * dt
    else:
        xi = x_goal

    return x, v, xi


def decay_spring_damper_exact(x: float, v: float, halflife: float, dt: float):
    """Exact decay spring: no goal, decays the value/velocity toward zero.

    Used to continue (extrapolate) motion that eases to a stop.

    :return: (new_position, new_velocity) after time step dt
    :rtype: tuple
    """
    y = halflife_to_damping(halflife) / 2.0
    j1 = v + x * y
    eydt = fast_negexp(y * dt)
    new_x = eydt * (x + j1 * dt)
    new_v = eydt * (v - j1 * y * dt)
    return new_x, new_v


def double_spring_damper_exact(x: float, v: float, xi: float, vi: float,
                               x_goal: float, halflife: float, dt: float):
    """Exact double critically damped spring: extra-smooth with a softer start.

    Chains two critical springs through an intermediate state (xi, vi).

    :return: (new_position, new_velocity, new_xi, new_vi)
    :rtype: tuple
    """
    xi, vi = critical_spring_damper_exact(xi, vi, x_goal, 0.0, 0.5 * halflife, dt)
    x, v = critical_spring_damper_exact(x, v, xi, vi, 0.5 * halflife, dt)
    return x, v, xi, vi


def resonant_spring_damper_exact(x: float, v: float, x_goal: float,
                                 frequency: float, halflife: float,
                                 dt: float, eps: float=1e-5):
    """Under-damped spring with an explicit oscillation frequency (Hz).

    Produces resonant overshoot/wobble around the goal that decays over the
    halflife. Used to layer secondary motion on top of a driving curve.

    :param float frequency: Damped oscillation frequency in Hz
    :param float halflife: Time for the oscillation amplitude to halve
    :return: (new_position, new_velocity) after time step dt
    :rtype: tuple
    """
    c = x_goal
    w = 2.0 * math.pi * frequency            # damped angular frequency
    y = halflife_to_damping(halflife) / 2.0  # decay rate
    j = math.sqrt((v + y * (x - c))**2 / (w * w + eps) + (x - c)**2)
    p = fast_atan((v + (x - c) * y) / (-(x - c) * w + eps))
    j = j if (x - c) > 0.0 else -j
    eydt = fast_negexp(y * dt)
    new_x = j * eydt * math.cos(w * dt + p) + c
    new_v = -y * j * eydt * math.cos(w * dt + p) - w * j * eydt * math.sin(w * dt + p)
    return new_x, new_v


# Solver process functions -------------------------------------------------- #
#
# Each takes (params: dict, data: dict, dt: float) and returns the new value
# list for one curve. ``data`` has 'values', 'times' and 'pre_value'. They are
# pure (no Maya calls) so they can be unit tested outside Maya.

def process_lerp(params: dict, data: dict, dt: float):
    """Simple iterative lerp smoothing (frame-rate dependent)."""
    factor = params['factor']
    x = data['values'][0]
    out = []
    for goal in data['values']:
        x = damper(x, goal, factor)
        out.append(x)
    return out


def process_spring_ratio(params: dict, data: dict, dt: float):
    """Full spring-damper; can overshoot/oscillate depending on damping ratio."""
    damping_ratio = params['damping']
    halflife = params['halflife']
    values = data['values']
    x = values[0]
    v = (x - data['pre_value']) / dt
    out = [x]
    for goal in values[1:]:
        x, v = spring_damper_exact_ratio(x, v, goal, 0.0, damping_ratio, halflife, dt)
        out.append(x)
    return out


def process_damper_exact(params: dict, data: dict, dt: float):
    """Exponential smoothing toward each value; no overshoot."""
    halflife = params['halflife']
    values = data['values']
    x = values[0]
    out = [x]
    for goal in values[1:]:
        x = damper_exact(x, goal, halflife, dt)
        out.append(x)
    return out


def process_critical_spring(params: dict, data: dict, dt: float):
    """Critically damped spring; preserves momentum, no overshoot."""
    halflife = params['halflife']
    values = data['values']
    x = values[0]
    v = (x - data['pre_value']) / dt
    out = [x]
    for goal in values[1:]:
        x, v = critical_spring_damper_exact(x, v, goal, 0.0, halflife, dt)
        out.append(x)
    return out


def process_double_spring(params: dict, data: dict, dt: float):
    """Double critically damped spring; extra-smooth with a softer start."""
    halflife = params['halflife']
    values = data['values']
    x = values[0]
    v = (x - data['pre_value']) / dt
    xi = values[0]
    vi = v
    out = [x]
    for goal in values[1:]:
        x, v, xi, vi = double_spring_damper_exact(x, v, xi, vi, goal, halflife, dt)
        out.append(x)
    return out


def process_velocity_spring(params: dict, data: dict, dt: float):
    """Speed-limited follow: move toward the curve at a capped speed.

    ``speed`` is a fraction (0-1) of the curve's own peak speed, so the control
    is curve-adaptive: 1.0 lets the motion keep up (like a critical spring),
    lower values cap the speed and introduce lag.
    """
    halflife = params['halflife']
    fraction = params['speed']
    values = data['values']
    peak_speed = max((abs(values[i] - values[i - 1]) / dt
                      for i in range(1, len(values))), default=0.0)
    v_goal = fraction * peak_speed
    x = values[0]
    v = (x - data['pre_value']) / dt
    xi = values[0]
    out = [x]
    for goal in values[1:]:
        x, v, xi = velocity_spring_damper_exact(x, v, xi, goal, v_goal, halflife, dt)
        out.append(x)
    return out


def process_extrapolation(params: dict, data: dict, dt: float, eps: float=1e-5):
    """Coast from the entry velocity, easing to a stop; ignores the goal values.

    Replaces the selection with inertial follow-through: it starts at the first
    key with the incoming velocity and decays that velocity to zero over the
    halflife, ignoring the original keyframe values. Good for ballistic tails.
    """
    halflife = params['halflife']
    values = data['values']
    x = values[0]
    v = (x - data['pre_value']) / dt
    y = 0.69314718056 / (halflife + eps)
    out = [x]
    for _ in values[1:]:
        eydt = fast_negexp(y * dt)
        x = x + (v / (y + eps)) * (1.0 - eydt)
        v = v * eydt
        out.append(x)
    return out


def process_inertialize(params: dict, data: dict, dt: float):
    """Blend out a discontinuity at the start of the selection, easing to shape.

    Inertialization removes a position/velocity pop at the selection boundary:
    the output starts matching the preceding (source) motion and decays the
    offset to zero over the halflife, converging onto the original selection.
    Select starting *at* the discontinuity for it to act on it; a smooth start
    yields little change (there is no pop to remove).
    """
    halflife = params['halflife']
    values = data['values']
    pre = data['pre_value']
    pre2 = data.get('pre2_value', pre)

    # Source: the preceding motion extrapolated to the first selected frame.
    src_v = (pre - pre2) / dt
    src_x = pre + src_v * dt
    # Destination: the original selection.
    dst_v = (values[1] - values[0]) / dt

    off_x = src_x - values[0]
    off_v = src_v - dst_v
    out = [values[0] + off_x]
    for value in values[1:]:
        off_x, off_v = decay_spring_damper_exact(off_x, off_v, halflife, dt)
        out.append(value + off_x)
    return out


def process_resonance(params: dict, data: dict, dt: float):
    """Resonant secondary motion: an under-damped oscillator driven by the curve.

    Blends a resonant (wobbling) version of the motion over the original by
    ``amount``. ``frequency`` sets the wobble speed and ``halflife`` how fast it
    settles. At amount 0 the curve is unchanged; at 1 it is fully springy.
    """
    frequency = params['frequency']
    halflife = params['halflife']
    amount = params['amount']
    values = data['values']
    x = values[0]
    v = (x - data['pre_value']) / dt
    out = [x]
    for goal in values[1:]:
        x, v = resonant_spring_damper_exact(x, v, goal, frequency, halflife, dt)
        out.append(lerp(goal, x, amount))
    return out


# Solver registry ----------------------------------------------------------- #
#
# Order here is the order shown in the window. Add a solver by adding an entry;
# the UI, presets and undo syncing all pick it up automatically.

SOLVER_ORDER = [
    'critical',
    'spring',
    'damper_exact',
    'critical_spring',
    'double_spring',
    'velocity_spring',
    'extrapolation',
    'inertialize',
    'resonance',
]

SOLVERS = {
    'critical': {
        'title': 'Critical Damping (Lerp)',
        'params': [
            {'name': 'factor', 'label': 'Damping Factor', 'min': 0.0, 'max': 1.0, 'default': 0.1},
        ],
        'process': process_lerp,
    },
    'spring': {
        'title': 'Spring Damper (Ratio)',
        'params': [
            {'name': 'damping', 'label': 'Damping Ratio', 'min': 0.001, 'max': 1.0, 'default': 0.4},
            {'name': 'halflife', 'label': 'Halflife', 'min': 0.0, 'max': 1.0, 'default': 0.25},
        ],
        'process': process_spring_ratio,
    },
    'damper_exact': {
        'title': 'Exact Damper',
        'params': [
            {'name': 'halflife', 'label': 'Halflife', 'min': 0.0, 'max': 1.0, 'default': 0.15},
        ],
        'process': process_damper_exact,
    },
    'critical_spring': {
        'title': 'Critical Spring',
        'params': [
            {'name': 'halflife', 'label': 'Halflife', 'min': 0.0, 'max': 1.0, 'default': 0.2},
        ],
        'process': process_critical_spring,
    },
    'double_spring': {
        'title': 'Double Spring',
        'params': [
            {'name': 'halflife', 'label': 'Halflife', 'min': 0.0, 'max': 1.0, 'default': 0.2},
        ],
        'process': process_double_spring,
    },
    'velocity_spring': {
        'title': 'Velocity Spring (Speed Limit)',
        'params': [
            {'name': 'halflife', 'label': 'Halflife', 'min': 0.0, 'max': 1.0, 'default': 0.2},
            {'name': 'speed', 'label': 'Speed (x peak)', 'min': 0.0, 'max': 1.0, 'default': 0.5},
        ],
        'process': process_velocity_spring,
    },
    'extrapolation': {
        'title': 'Extrapolation (Follow-through)',
        'params': [
            {'name': 'halflife', 'label': 'Halflife', 'min': 0.001, 'max': 2.0, 'default': 0.4},
        ],
        'process': process_extrapolation,
    },
    'inertialize': {
        'title': 'Inertialization',
        'params': [
            {'name': 'halflife', 'label': 'Halflife', 'min': 0.0, 'max': 1.0, 'default': 0.25},
        ],
        'process': process_inertialize,
    },
    'resonance': {
        'title': 'Resonance (Secondary Motion)',
        'params': [
            {'name': 'frequency', 'label': 'Frequency (Hz)', 'min': 0.1, 'max': 15.0, 'default': 3.0},
            {'name': 'halflife', 'label': 'Halflife', 'min': 0.001, 'max': 2.0, 'default': 0.3},
            {'name': 'amount', 'label': 'Amount', 'min': 0.0, 'max': 1.0, 'default': 1.0},
        ],
        'process': process_resonance,
    },
}


# Apply --------------------------------------------------------------------- #

def get_param_values(key: str):
    """Read a solver's slider values as a {param_name: value} dict."""
    return {p['name']: cmds.floatSliderGrp(SLIDER_CONTROLS[key][p['name']], q=True, value=True)
            for p in SOLVERS[key]['params']}


def apply_solver(key: str, *args):
    """Run a solver over the selection. Slider drag callback for that solver.

    :param str key: Solver key
    :param args: Trailing Maya callback args (unused)
    """
    global LAST_SOLVER
    begin()
    if not KEY_DATA:
        return
    LAST_SOLVER = key

    params = get_param_values(key)
    dt = max(DELTA_TIME, 1e-3)
    process = SOLVERS[key]['process']

    for curve, data in KEY_DATA.items():
        new_values = process(params, data, dt)
        apply_values(curve, data['times'], new_values)


def update_dt(*args):
    """Delta Time slider callback: update the shared timestep and re-apply.

    :param args: args[0] is the new delta time value
    """
    global DELTA_TIME
    begin()
    DELTA_TIME = max(args[0], 1e-3)
    framerate = round(1.0 / DELTA_TIME, 2)
    cmds.floatSliderGrp(DT_SLIDER, e=True, label=f'Delta time ({framerate}fps) ')
    if LAST_SOLVER:
        apply_solver(LAST_SOLVER)


# Undo / redo slider sync --------------------------------------------------- #

def all_slider_controls():
    """All slider control names: the shared Delta Time plus every solver param."""
    controls = [DT_SLIDER] if DT_SLIDER else []
    for key in SOLVER_ORDER:
        controls.extend(SLIDER_CONTROLS.get(key, {}).values())
    return controls


def capture_slider_state():
    """Snapshot the current values of all sliders, keyed by control name."""
    return {control: cmds.floatSliderGrp(control, q=True, value=True)
            for control in all_slider_controls()}


def restore_slider_state(state: dict):
    """Push a captured slider snapshot back onto the sliders.

    Also resyncs the DELTA_TIME global and Delta Time label from the slider.
    """
    global DELTA_TIME
    for control, value in state.items():
        cmds.floatSliderGrp(control, e=True, value=value)

    if DT_SLIDER:
        DELTA_TIME = max(cmds.floatSliderGrp(DT_SLIDER, q=True, value=True), 1e-3)
        framerate = round(1.0 / DELTA_TIME, 2)
        cmds.floatSliderGrp(DT_SLIDER, e=True, label=f'Delta time ({framerate}fps) ')


def on_undo():
    """Mirror an undo of one of our operations back onto the sliders.

    Guarded by CHUNK_NAME so it only reacts when the operation just undone was
    a SpringyKeys apply, leaving unrelated undos alone.
    """
    global RESTING_STATE
    if not SLIDER_UNDO_STACK:
        return
    if cmds.undoInfo(q=True, redoName=True) != CHUNK_NAME:
        return  # The thing just undone was not ours

    state = SLIDER_UNDO_STACK.pop()
    SLIDER_REDO_STACK.append(capture_slider_state())
    restore_slider_state(state)
    RESTING_STATE = state


def on_redo():
    """Mirror a redo of one of our operations back onto the sliders."""
    global RESTING_STATE
    if not SLIDER_REDO_STACK:
        return
    if cmds.undoInfo(q=True, undoName=True) != CHUNK_NAME:
        return  # The thing just redone was not ours

    state = SLIDER_REDO_STACK.pop()
    SLIDER_UNDO_STACK.append(capture_slider_state())
    restore_slider_state(state)
    RESTING_STATE = state


# Session ------------------------------------------------------------------- #

def begin():
    """Initialize a processing session and capture keyframe data.

    .. note::
        - Only captures a fresh snapshot if the selection has changed
        - Opens a named Maya undo chunk only when starting to slide
        - Records the pre-operation slider state for undo syncing
    """
    global KEY_DATA
    global SELECTION_FINGERPRINT
    global UNDO_OPEN

    if selection_changed(SELECTION_FINGERPRINT):
        KEY_DATA = get_selected_keyframe_data()
        SELECTION_FINGERPRINT = get_current_selection_fingerprint()

    if not UNDO_OPEN:
        # Remember the slider state as it was before this operation so an undo
        # can restore it. A fresh operation invalidates the redo history.
        if RESTING_STATE is not None:
            SLIDER_UNDO_STACK.append(RESTING_STATE)
            SLIDER_REDO_STACK.clear()
        cmds.undoInfo(openChunk=True, chunkName=CHUNK_NAME)
        UNDO_OPEN = True


def complete(*args):
    """Slider release callback: close the undo chunk and record the resting state."""
    global UNDO_OPEN
    global RESTING_STATE

    if UNDO_OPEN:
        cmds.undoInfo(closeChunk=True)
        UNDO_OPEN = False
        RESTING_STATE = capture_slider_state()


# Presets ------------------------------------------------------------------- #

def optionvar_name(key: str, index: int):
    """Per-solver, per-slot optionVar name for a preset."""
    return f'springyKeys_{key}_preset_{index}'


def get_solver_param_values(key: str):
    """Read a solver's slider values as an ordered list (storage order)."""
    return [cmds.floatSliderGrp(SLIDER_CONTROLS[key][p['name']], q=True, value=True)
            for p in SOLVERS[key]['params']]


def set_solver_param_values(key: str, values: list):
    """Push an ordered list of values back onto a solver's sliders."""
    for spec, value in zip(SOLVERS[key]['params'], values):
        cmds.floatSliderGrp(SLIDER_CONTROLS[key][spec['name']], e=True, value=value)


def refresh_preset_button(key: str, index: int):
    """Update a preset button's label and tooltip to reflect its saved state."""
    buttons = PRESET_BUTTONS.get(key, [])
    if index >= len(buttons):
        return

    button = buttons[index]
    name = optionvar_name(key, index)

    if cmds.optionVar(exists=name):
        values = [float(x) for x in cmds.optionVar(q=name).split(',')]
        labels = [p['label'] for p in SOLVERS[key]['params']]
        lines = ['{0}: {1:.3f}'.format(label, value)
                 for label, value in zip(labels, values)]
        annotation = 'Preset {0} (saved)\n{1}\nLeft-click: apply   Right-click: edit'.format(
            index + 1, '\n'.join(lines))
        cmds.button(button, e=True, label=f'{index + 1} *', annotation=annotation)
    else:
        cmds.button(
            button, e=True, label=f'{index + 1}',
            annotation=f'Preset {index + 1} (empty)\nRight-click to save current values'
        )


def load_preset(key: str, index: int, *args):
    """Restore a preset and apply its solver to the selection (left-click)."""
    name = optionvar_name(key, index)
    if not cmds.optionVar(exists=name):
        cmds.warning(f'Preset {index + 1} is empty. Right-click to save current values.')
        return

    values = [float(x) for x in cmds.optionVar(q=name).split(',')]
    set_solver_param_values(key, values)
    apply_solver(key)
    complete()


def save_preset(key: str, index: int, *args):
    """Store a solver's current slider values into a preset slot (right-click)."""
    name = optionvar_name(key, index)
    values = get_solver_param_values(key)
    cmds.optionVar(stringValue=(name, ','.join(repr(v) for v in values)))
    refresh_preset_button(key, index)


def clear_preset(key: str, index: int, *args):
    """Remove a preset slot's saved values (right-click menu)."""
    name = optionvar_name(key, index)
    if cmds.optionVar(exists=name):
        cmds.optionVar(remove=name)
    refresh_preset_button(key, index)


def build_preset_row(key: str):
    """Build a horizontal row of preset buttons for a solver."""
    PRESET_BUTTONS[key] = []
    cmds.rowLayout(numberOfColumns=PRESET_COUNT, columnAttach=[
        (i + 1, 'both', 2) for i in range(PRESET_COUNT)])
    for i in range(PRESET_COUNT):
        button = cmds.button( label=f'{i + 1}', width=44, height=24
                            , command=partial(load_preset, key, i) )  # pylint: disable=E1111
        cmds.popupMenu( parent=button, button=3 )
        cmds.menuItem( label='Save current values', command=partial(save_preset, key, i) )
        cmds.menuItem( label='Clear saved values', command=partial(clear_preset, key, i) )
        PRESET_BUTTONS[key].append(button)
    cmds.setParent('..')


# UI ------------------------------------------------------------------------ #

def build_solver_frame(key: str):
    """Build one collapsible frame: a solver's sliders plus its preset row."""
    spec = SOLVERS[key]
    SLIDER_CONTROLS[key] = {}

    cmds.frameLayout( label=spec['title'], collapsable=True, collapse=False
                    , marginWidth=4, marginHeight=4 )
    cmds.rowLayout( numberOfColumns=2, adjustableColumn=1
                  , columnAttach=[(1, 'both', 0), (2, 'both', 6)] )

    cmds.columnLayout( adjustableColumn=True )
    for p in spec['params']:
        control = cmds.floatSliderGrp( label=p['label'], field=True
                                     , min=p['min'], max=p['max'], value=p['default']
                                     , step=0.001, dragCommand=partial(apply_solver, key)
                                     , changeCommand=complete, adjustableColumn=0 )  # pylint: disable=E1111
        SLIDER_CONTROLS[key][p['name']] = control
    cmds.setParent('..')

    build_preset_row(key)
    cmds.setParent('..')  # rowLayout
    cmds.setParent('..')  # frameLayout


def ui():
    """Create and display the SpringyKeys user interface.

    .. note::
        One shared Delta Time control at the top, then a collapsible frame per
        solver (see SOLVER_ORDER), each with its tuning sliders and a bank of
        three presets. Left-click a preset to apply that solver to the
        selection; right-click to save or clear. Presets persist across Maya
        sessions via optionVar.
    """
    global DT_SLIDER
    global RESTING_STATE

    if cmds.window("springOverlapWin", exists=True):
        cmds.deleteUI("springOverlapWin")

    # Forget any remembered size so the fit-to-content below is what shows.
    try:
        if cmds.windowPref("springOverlapWin", exists=True):
            cmds.windowPref("springOverlapWin", remove=True)
    except (RuntimeError, TypeError):
        pass

    window = cmds.window("springOverlapWin", title="SpringyKeys", iconName='springykeys', widthHeight=(760, 200))  # pylint: disable=E1111

    cmds.columnLayout( adjustableColumn=True )

    # Shared Delta Time (frame time) used by all spring-based solvers
    framerate = round(1.0 / DELTA_TIME, 2)
    DT_SLIDER = cmds.floatSliderGrp( label=f'Delta time ({framerate}fps) ', field=True
                                   , min=0.001, max=1.0, value=DELTA_TIME, step=0.001
                                   , dragCommand=update_dt, changeCommand=complete
                                   , adjustableColumn=0 )  # pylint: disable=E1111
    cmds.separator( style='in', height=8 )

    for key in SOLVER_ORDER:
        build_solver_frame(key)

    cmds.setParent('..')  # main columnLayout

    for key in SOLVER_ORDER:
        for i in range(PRESET_COUNT):
            refresh_preset_button(key, i)

    # Seed the undo-sync state with the sliders' starting values
    RESTING_STATE = capture_slider_state()
    SLIDER_UNDO_STACK.clear()
    SLIDER_REDO_STACK.clear()

    # Mirror slider values when the user undoes/redoes our operations. Parented
    # to the window so the jobs are removed when it closes. Wrapped defensively
    # so an unsupported event never blocks the window from opening.
    try:
        cmds.scriptJob(event=['Undo', on_undo], parent=window)
        cmds.scriptJob(event=['Redo', on_redo], parent=window)
    except RuntimeError:
        cmds.warning('SpringyKeys: undo/redo slider syncing is unavailable in this Maya version.')

    cmds.showWindow(window)

    # Open at the size of the content (avoids empty space below the frames).
    try:
        cmds.window(window, e=True, resizeToFitChildren=True)
    except (RuntimeError, TypeError):
        pass


if __name__ == "__main__":
    ui()
