"""
Runs springy filter function on selected keyframes in the graph editor

This tool applies physics-based spring damping to animation curves, providing
two methods: simple critical damping and full spring physics simulation.
"""

import math
from maya import cmds, mel


# Globals ------------------------------------------------------------------- #

UNDO_OPEN = False
KEY_DATA = {}
SELECTION_FINGERPRINT = {}
DAMPING_RATIO = 0.4
HALFLIFE = 0.25
DELTA_TIME = 1 / 30.0
DAMPING_FACTOR = 0.1
TIMELINE = mel.eval('string $tmpString=$gPlayBackSlider') # pylint: disable=E1111
GRAPH_EDITOR = 'graphEditor1GraphEd'


# Private ------------------------------------------------------------------- #

def is_equal(lst: list):
    """Check if all items in a list are equal.

    :param list lst: List of values to compare
    :return: True if all items are equal or list is empty, False otherwise
    :rtype: bool

    Examples:
        >>> is_equal([1, 1, 1])
        True
        >>> is_equal([1, 2, 3])
        False
        >>> is_equal([])
        True
    """
    return not lst or lst.count(lst[0]) == len(lst)


def get_current_selection_fingerprint():
    """Get fingerprint of current Graph Editor keyframe selection.

    :return: Dictionary mapping curve names to (start_time, end_time) tuples.
             Empty dict if no valid selection.
    :rtype: dict

    .. note::
        Only includes curves with multiple selected keyframes.
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

    :param dict SELECTION_FINGERPRINT: List of values to compare
    :return: True if selection has changed, False if unchanged
    :rtype: bool
    """
    current_fingerprint = get_current_selection_fingerprint()
    return current_fingerprint != SELECTION_FINGERPRINT


def get_selected_keyframe_data():
    """Extract keyframe data from currently selected curves in Graph Editor.

    :return: Dictionary mapping curve names to their keyframe data:
             {'curve_name': {'times': [float, ...], 'values': [float, ...]}}
             Returns None if no valid selection or data found
    :rtype: dict or None

    :raises RuntimeError: If Graph Editor widget cannot be found

    .. note::
        - Filters out single keyframe selections (need at least 2 for interpolation)
        - Skips flat curves (all values equal)
        - Uses continuous time range between first and last selected keys
    """
    # get the key selection
    if not cmds.animCurveEditor(GRAPH_EDITOR, exists=True):
        cmds.error("{} not found.".format(GRAPH_EDITOR))
        return # Cannot find graph editor?

    if not cmds.animCurveEditor(GRAPH_EDITOR, q=True, areCurvesSelected=True):
        cmds.warning("No keys selected to operate on.")
        return

    selected_curves = cmds.keyframe(q=True, selected=True, name=True) or []
    if not selected_curves: return None # How did the last check not catch this?

    # The Data dictionary
    key_data = {}

    for curve in selected_curves:

        selected_index = cmds.keyframe(curve, q=True, selected=True, indexValue=True)
        if len(selected_index) == 1: continue # Bounce

        selected_times = cmds.keyframe(curve, q=True, selected=True)

        time_range = []
        time_range.extend( [float(x) for x in range(int(selected_times[0])
                         , int(selected_times[-1]))])

        # Extend 1 frame in either direction to act as pivot?
        # time_range.insert(selected_times[0] - 1, 0)
        # time_range.append(selected_times[-1] + 1)

        value_range = []
        for time in time_range:
            value = cmds.keyframe(curve, q=True, time=(time, ), eval=True, valueChange=True)
            value_range.extend(value)

        if is_equal(value_range):
            continue # Ignore flat curves

        key_data[curve] = { "times" : time_range
                          , "values" : value_range
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
    # Do the magic, do the magic!
    for time, value in zip(times, values):
        cmds.keyframe(curve, e=True, time=(time,), valueChange=value)


def damping_ratio_to_stiffness(ratio: float, damping: float):
    """Convert damping ratio and damping coefficient to spring stiffness.

    :param float ratio: Damping ratio (typically 0.0 to 1.0)
    :param float damping: Damping coefficient

    :return: Spring stiffness value
    :rtype: float

    .. note::
        Used in spring physics calculations. Higher stiffness = faster response.
    """
    result = (damping / (ratio * 2.0))**2
    return result


def damping_ratio_to_damping(ratio: float, stiffness: float):
    """Convert damping ratio and stiffness to damping coefficient.

    :param float ratio: Damping ratio (typically 0.0 to 1.0)
    :param float stiffness: Spring stiffness value

    :return: Damping coefficient
    :rtype: float

    .. note::
        Used in spring physics calculations. Higher damping = more resistance.
    """
    result = ratio * 2.0 * (stiffness ** 0.5)
    return result


def halflife_to_damping(halflife: float, eps: float=1e-5):
    """Convert halflife duration to damping coefficient.

    :param float halflife: Time for amplitude to reduce by half
    :param float eps: Small epsilon to prevent division by zero
    :return: Damping coefficient
    :rtype: float

    .. note::
        Shorter halflife = faster decay = higher damping coefficient.
        Uses natural logarithm constant for exponential decay calculation.
    """
    result = (4.0 * 0.69314718056) / (halflife + eps)
    return result


def fast_atan(x: float):
    """Fast approximation of arctangent function.

    :param float x: Input value
    :return: Arctangent of x in radians
    :rtype: float

    .. note::
        Polynomial approximation for performance in real-time applications.
        Accuracy trade-off for speed in spring physics calculations.
    """
    z = abs(x)

    if z > 1.0:
        w = 1.0 / z
    else:
        w = z

    y = (math.pi / 4.0)*w - w*(w - 1)*(0.2447 + 0.0663*w)

    if z > 1.0:
        new_y = math.pi / 2.0 - y
    else:
        new_y = y

    return math.copysign(new_y, x)


def fast_negexp(x: float):
    """Fast approximation of negative exponential function.

    :param float x: Input value
    :return: Approximation of e^(-x)
    :rtype: float

    .. note::
        Rational function approximation for performance.
        Used in spring damping calculations where exact precision
        is less critical than speed.
    """
    return 1.0 / (1.0 + x + 0.48*x*x + 0.235*x*x*x)


def lerp(a: float, b: float, t: float):
    """Linear interpolation between two values.

    :param float a: Start value
    :param float b: End value
    :param float t: Interpolation factor (0.0 to 1.0)
    :return: Interpolated value
    :rtype: float

    .. note::
        a=0.0 returns a, a=1.0 returns b, a=0.5 returns midpoint.
    """
    return (1.0 - t) * a + t * b


def damper(x: float, g: float, factor: float):
    """Apply simple damping interpolation.

    :param float x: Current value
    :param floatg: Goal/target value
    :param float factor: Damping factor (0.0 to 1.0)
    :return: Damped value
    :rtype: float

    .. note::
        Simple linear interpolation toward goal. Higher factor = faster approach.
        Used by the Critical Damping Ratio slider.
    """
    return lerp(x, g, factor)


def update_factor(*args):
    """Update keyframes using simple critical damping method.

    :param args: Variable arguments from Maya slider callback. args[0] contains the slider value.
    :type args: tuple

    .. note::
        Callback function for Critical Damping Ratio slider.
        Applies progressive smoothing using linear interpolation.
        Requires KEY_DATA to be populated by begin().
    """
    begin()
    factor = args[0]

    if not KEY_DATA:
        return
    curves = KEY_DATA.keys()

    for curve in curves:
        current_y = KEY_DATA[curve]["values"][0]
        new_values = []
        for value in KEY_DATA[curve]["values"]:
            current_y = damper(current_y, value, factor)
            new_values.append(current_y)

        apply_values(curve, KEY_DATA[curve]["times"], new_values)


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
    """Calculate next spring position using exact mathematical solution.

    :param float x: Current position
    :param float v: Current velocity
    :param float x_goal: Target position
    :param floatv_goal: Target velocity (usually 0.0)
    :param float damping_ratio: Damping ratio (0.0=undamped, 1.0=critical, >1.0=overdamped)
    :param float halflife: Time for oscillation amplitude to halve
    :param float dt: Time step (delta time)
    :param float eps: Small epsilon for numerical stability
    :return: (new_position, new_velocity) after time step dt
    :rtype: tuple

    .. note::
        Implements exact analytical solution for spring-damper system.
        Handles three cases:

        - Critical damping (ratio â‰ˆ 1.0): Fastest approach without overshoot
        - Under damping (ratio < 1.0): Oscillatory with decay
        - Over damping (ratio > 1.0): Slow approach without overshoot
    """
    g = x_goal
    q = v_goal
    d = halflife_to_damping(halflife) # Damping
    s = damping_ratio_to_stiffness(damping_ratio, d) # Stiffness
    c = g + (d*q) / (s + eps)
    y = d / 2.0

    ## Start

    if abs(s - (d * d) / 4.0) < eps:  # Critically Damped
        j0 = x - c
        j1 = y + j0 * y

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

    else:  # Over Damped (s - (d*d) / 4.0 < 0.0)
        y0 = (d + math.sqrt(d * d - 4 * s)) / 2.0
        y1 = (d - math.sqrt(d * d - 4 * s)) / 2.0
        j1 = (c * y0 - x * y0 - v) / (y1 - y0)
        j0 = x - j1 - c

        ey0dt = fast_negexp(y0 * dt)
        ey1dt = fast_negexp(y1 * dt)

        new_x = j0 * ey0dt + j1 * ey1dt + c
        new_v = -y0 * j0 * ey0dt - y1 * j1 * ey1dt

    return new_x, new_v


def update_spring_keys(*args):
    """Update keyframes using full spring physics simulation.

    :param args: Variable arguments from Maya slider callback (unused)
    :type args: tuple

    .. note::
        Callback function for Damping Ratio and Halflife sliders.
        Applies spring-damper physics to each keyframe sequentially,
        using the previous frame's result as input for the next.
        Requires KEY_DATA to be populated by begin().
    """
    begin()
    damping_ratio = cmds.floatSliderGrp(SLIDER_DAMPING, q=True, value=True)
    halflife = cmds.floatSliderGrp(SLIDER_HALFLIFE, q=True, value=True)

    if not KEY_DATA:
        return
    curves = KEY_DATA.keys()

    for curve in curves:
        new_values = []
        current_x = KEY_DATA[curve]["values"][0]
        current_v = 0.0
        for value in KEY_DATA[curve]["values"]:
            current_x, current_v = spring_damper_exact_ratio(   current_x,
                                                                current_v,
                                                                value,
                                                                0.0,
                                                                damping_ratio,
                                                                halflife,
                                                                DELTA_TIME
                                                              )
            new_values.append(current_x)

        apply_values(curve, KEY_DATA[curve]["times"], new_values)


def update_deltatime(*args):
    """Update delta time and refresh spring calculations.

    :param args: Variable arguments from Maya slider callback. args[0] contains the new delta time value.
    :type args: tuple

    .. note::
        Callback function for Delta Time slider. Updates the global
        DELTA_TIME variable and recalculates spring physics.
        Also updates the slider label to show equivalent framerate.
    """
    begin()
    global DELTA_TIME
    DELTA_TIME = args[0]
    update_spring_keys()
    framerate = round(1/DELTA_TIME, 2)
    cmds.floatSliderGrp(SLIDER_DT, e=True, label=f'Delta time ({framerate}fps) ')


def begin():
    """Initialize processing session and capture keyframe data.

    .. note::
        - Only captures new snapshot if selection has changed
        - Opens Maya undo chunk only when starting sliding
        - Captures current keyframe data into global KEY_DATA
    """
    global KEY_DATA
    global SELECTION_FINGERPRINT
    global UNDO_OPEN


    # Check if we need a fresh snapshot
    if selection_changed(SELECTION_FINGERPRINT):
        # Start fresh
        KEY_DATA = get_selected_keyframe_data()
        SELECTION_FINGERPRINT = get_current_selection_fingerprint()

    if not UNDO_OPEN:
        cmds.undoInfo(openChunk=True)
        UNDO_OPEN = True


def complete(*args):
    """Completion callback for slider interactions.

    :param args: Variable arguments from Maya slider callback (unused)
    :type args: tuple

    .. note::
        Called when slider interaction is complete.
    """
    global UNDO_OPEN

    # Close previous undo chunk if we had one
    if UNDO_OPEN:
        cmds.undoInfo(closeChunk=True)
        UNDO_OPEN = False



def ui():
    """Create and display the SpringyKeys user interface.

    .. note::
        Creates a window with four sliders:

        - Critical Damping Ratio: Simple interpolation-based smoothing
        - Damping Ratio: Controls oscillation behavior in spring physics
        - Halflife: Controls decay rate of spring oscillations
        - Delta Time: Simulation time step (affects responsiveness)

        Window is recreated if it already exists.
    """
    global SLIDER_FACTOR
    global SLIDER_DAMPING
    global SLIDER_HALFLIFE
    global SLIDER_DT
    # Check if window exists and delete it
    if cmds.window("springOverlapWin", exists=True):
        cmds.deleteUI("springOverlapWin")

    window = cmds.window("springOverlapWin", title="SpringyKeys", iconName='springykeys', widthHeight=(600, 106))  # pylint: disable=E1111
    cmds.columnLayout( adjustableColumn=True)
    SLIDER_FACTOR = cmds.floatSliderGrp( label='Critical Damping Ratio' , field=True, min=0.0, max=1.0, value=DAMPING_FACTOR, step=0.001, dragCommand=update_factor,  changeCommand=complete, adjustableColumn=0  )  # pylint: disable=E1111
    cmds.separator()
    SLIDER_DAMPING  = cmds.floatSliderGrp( label='Damping Ratio ', field=True, min=0.001, max=1.0, value=DAMPING_RATIO, step=0.001, dragCommand=update_spring_keys, changeCommand=complete, adjustableColumn=0 )  # pylint: disable=E1111
    SLIDER_HALFLIFE = cmds.floatSliderGrp( label='Halflife' , field=True, min=0.0, max=1.0, value=HALFLIFE, step=0.001, dragCommand=update_spring_keys,  changeCommand=complete, adjustableColumn=0  )  # pylint: disable=E1111
    SLIDER_DT = cmds.floatSliderGrp( label='Delta time (30fps) ' , field=True, min=0.0, max=1.0, value=DELTA_TIME, step=0.001, dragCommand=update_deltatime,  changeCommand=complete, adjustableColumn=0  )  # pylint: disable=E1111
    cmds.showWindow(window)


if __name__ == "__main__":
    ui()
