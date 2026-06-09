"""
Runs springy filter function on selected keyframes in the graph editor

This tool applies physics-based spring damping to animation curves, providing
two methods: simple critical damping and full spring physics simulation.
"""

import math
from functools import partial
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

# Undo/redo slider sync. Slider values are pure UI state and are not part of
# Maya's undo queue, so we track them ourselves and mirror them when the user
# undoes/redoes one of our operations (chunks tagged with CHUNK_NAME).
CHUNK_NAME = 'SpringyKeys'
RESTING_STATE = None       # slider values at rest (before the next operation)
SLIDER_UNDO_STACK = []     # resting states, one per applied operation
SLIDER_REDO_STACK = []

PRESET_COUNT = 3

# Each solver is its own preset "section": its presets store only that
# solver's sliders and applying one runs only that solver. Add a new solver
# by adding an entry here (and a branch in the section helpers below).
# 'buttons' is filled in by ui().
SECTIONS = {
    'critical': {
        'optionvar': 'springyKeys_critical_preset_{}',
        'labels': ['Critical Damping'],
        'buttons': [],
    },
    'spring': {
        'optionvar': 'springyKeys_spring_preset_{}',
        'labels': ['Damping Ratio', 'Halflife', 'Delta Time'],
        'buttons': [],
    },
}


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
                         , int(selected_times[-1]) + 1)])

        # Extend 1 frame in either direction to act as pivot?
        # time_range.insert(selected_times[0] - 1, 0)
        # time_range.append(selected_times[-1] + 1)

        value_range = []
        for time in time_range:
            value = cmds.keyframe(curve, q=True, time=(time, ), eval=True, valueChange=True)
            value_range.extend(value)

        if is_equal(value_range):
            continue # Ignore flat curves

        # Get the value at the frame before the first selected key
        # so we can compute the incoming velocity at the start
        pre_time = time_range[0] - 1
        pre_value = cmds.keyframe(curve, q=True, time=(pre_time,), eval=True, valueChange=True)
        pre_value = pre_value[0] if pre_value else value_range[0]

        key_data[curve] = { "times" : time_range
                          , "values" : value_range
                          , "pre_value" : pre_value
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
        values = KEY_DATA[curve]["values"]
        current_x = values[0]
        current_v = (current_x - KEY_DATA[curve]["pre_value"]) / DELTA_TIME

        # The first selected key keeps its original value. Its incoming
        # velocity is carried into the spring so the next steps continue
        # the existing momentum instead of snapping forward an extra step.
        new_values = [current_x]
        for value in values[1:]:
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
    # Clamp away from zero so the velocity calc and framerate label below
    # never divide by zero when the slider is dragged fully left.
    DELTA_TIME = max(args[0], 1e-3)
    update_spring_keys()
    framerate = round(1/DELTA_TIME, 2)
    cmds.floatSliderGrp(SLIDER_DT, e=True, label=f'Delta time ({framerate}fps) ')


def capture_slider_state():
    """Snapshot the current values of all sliders.

    :return: Slider values keyed by name (plus the DELTA_TIME global)
    :rtype: dict
    """
    return {
        'factor':   cmds.floatSliderGrp(SLIDER_FACTOR, q=True, value=True),
        'damping':  cmds.floatSliderGrp(SLIDER_DAMPING, q=True, value=True),
        'halflife': cmds.floatSliderGrp(SLIDER_HALFLIFE, q=True, value=True),
        'dt':       cmds.floatSliderGrp(SLIDER_DT, q=True, value=True),
    }


def restore_slider_state(state: dict):
    """Push a captured slider snapshot back onto the sliders.

    :param dict state: Snapshot produced by :func:`capture_slider_state`

    .. note::
        Also restores the DELTA_TIME global and the Delta Time label so the
        spring sim and framerate stay in sync with the sliders.
    """
    global DELTA_TIME
    cmds.floatSliderGrp(SLIDER_FACTOR, e=True, value=state['factor'])
    cmds.floatSliderGrp(SLIDER_DAMPING, e=True, value=state['damping'])
    cmds.floatSliderGrp(SLIDER_HALFLIFE, e=True, value=state['halflife'])
    cmds.floatSliderGrp(SLIDER_DT, e=True, value=state['dt'])

    DELTA_TIME = max(state['dt'], 1e-3)
    framerate = round(1.0 / DELTA_TIME, 2)
    cmds.floatSliderGrp(SLIDER_DT, e=True, label=f'Delta time ({framerate}fps) ')


def on_undo():
    """Mirror an undo of one of our operations back onto the sliders.

    .. note::
        Triggered by the Undo scriptJob event. Guarded by CHUNK_NAME so it
        only reacts when the operation just undone was a SpringyKeys apply,
        leaving unrelated undos alone.
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
    """Mirror a redo of one of our operations back onto the sliders.

    .. note::
        Triggered by the Redo scriptJob event. Guarded by CHUNK_NAME so it
        only reacts when the operation just redone was a SpringyKeys apply.
    """
    global RESTING_STATE
    if not SLIDER_REDO_STACK:
        return
    if cmds.undoInfo(q=True, undoName=True) != CHUNK_NAME:
        return  # The thing just redone was not ours

    state = SLIDER_REDO_STACK.pop()
    SLIDER_UNDO_STACK.append(capture_slider_state())
    restore_slider_state(state)
    RESTING_STATE = state


def begin():
    """Initialize processing session and capture keyframe data.

    .. note::
        - Only captures new snapshot if selection has changed
        - Opens Maya undo chunk only when starting sliding
        - Captures current keyframe data into global KEY_DATA
        - Records the pre-operation slider state for undo syncing
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
        # Remember the slider state as it was before this operation so an undo
        # can restore it. A fresh operation invalidates the redo history.
        if RESTING_STATE is not None:
            SLIDER_UNDO_STACK.append(RESTING_STATE)
            SLIDER_REDO_STACK.clear()
        cmds.undoInfo(openChunk=True, chunkName=CHUNK_NAME)
        UNDO_OPEN = True


def complete(*args):
    """Completion callback for slider interactions.

    :param args: Variable arguments from Maya slider callback (unused)
    :type args: tuple

    .. note::
        Called when slider interaction is complete. Records the new slider
        state as the resting point for the next operation's undo entry.
    """
    global UNDO_OPEN
    global RESTING_STATE

    # Close previous undo chunk if we had one
    if UNDO_OPEN:
        cmds.undoInfo(closeChunk=True)
        UNDO_OPEN = False
        RESTING_STATE = capture_slider_state()



# Presets ------------------------------------------------------------------ #

def section_slider_names(section: str):
    """Return the slider control names owned by a solver section.

    :param str section: Section key ('critical' or 'spring')
    :return: Slider control names, in stored-value order
    :rtype: list
    """
    if section == 'critical':
        return [SLIDER_FACTOR]
    return [SLIDER_DAMPING, SLIDER_HALFLIFE, SLIDER_DT]


def get_section_values(section: str):
    """Read the current values of a section's sliders.

    :param str section: Section key ('critical' or 'spring')
    :return: One value per slider, in stored order
    :rtype: list
    """
    return [cmds.floatSliderGrp(s, q=True, value=True)
            for s in section_slider_names(section)]


def set_section_values(section: str, values: list):
    """Push saved values back onto a section's sliders.

    :param str section: Section key ('critical' or 'spring')
    :param list values: One value per slider, in stored order

    .. note::
        Setting a slider value in code does not fire its drag callback, so
        this only restores positions. For the spring section the global
        DELTA_TIME (which the sim reads instead of the slider) and the Delta
        Time label are refreshed so the framerate stays in sync.
    """
    global DELTA_TIME
    for slider, value in zip(section_slider_names(section), values):
        cmds.floatSliderGrp(slider, e=True, value=value)

    if section == 'spring':
        dt = values[-1]  # Delta Time is the last spring slider
        DELTA_TIME = max(dt, 1e-3)
        framerate = round(1.0 / DELTA_TIME, 2)
        cmds.floatSliderGrp(SLIDER_DT, e=True, label=f'Delta time ({framerate}fps) ')


def apply_section(section: str):
    """Run a section's solver on the current selection as one undo step.

    :param str section: Section key ('critical' or 'spring')

    .. note::
        Restoring slider values in code does not fire their drag callbacks,
        so the solver is invoked explicitly. ``complete`` closes the undo
        chunk opened by ``begin`` (inside the update functions).
    """
    if section == 'critical':
        update_factor(cmds.floatSliderGrp(SLIDER_FACTOR, q=True, value=True))
    else:
        update_spring_keys()
    complete()


def refresh_preset_button(section: str, index: int):
    """Update a preset button's label and tooltip to reflect its saved state.

    :param str section: Section key ('critical' or 'spring')
    :param int index: Zero-based preset slot

    .. note::
        Saved slots get a trailing ``*`` and a tooltip listing the stored
        values; empty slots prompt the user to right-click to save.
    """
    buttons = SECTIONS[section]['buttons']
    if index >= len(buttons):
        return

    button = buttons[index]
    name = SECTIONS[section]['optionvar'].format(index)

    if cmds.optionVar(exists=name):
        values = [float(x) for x in cmds.optionVar(q=name).split(',')]
        lines = ['{0}: {1:.3f}'.format(label, value)
                 for label, value in zip(SECTIONS[section]['labels'], values)]
        annotation = 'Preset {0} (saved)\n{1}\nLeft-click: apply   Right-click: edit'.format(
            index + 1, '\n'.join(lines))
        cmds.button(button, e=True, label=f'{index + 1} *', annotation=annotation)
    else:
        cmds.button(
            button, e=True, label=f'{index + 1}',
            annotation=f'Preset {index + 1} (empty)\nRight-click to save current values'
        )


def load_preset(section: str, index: int, *args):
    """Restore a preset and apply its solver to the selection (left-click).

    :param str section: Section key ('critical' or 'spring')
    :param int index: Zero-based preset slot
    :param args: Trailing Maya callback args (unused)
    """
    name = SECTIONS[section]['optionvar'].format(index)
    if not cmds.optionVar(exists=name):
        cmds.warning(f'Preset {index + 1} is empty. Right-click to save current values.')
        return

    values = [float(x) for x in cmds.optionVar(q=name).split(',')]
    set_section_values(section, values)
    apply_section(section)


def save_preset(section: str, index: int, *args):
    """Store a section's current slider values into a preset slot (right-click).

    :param str section: Section key ('critical' or 'spring')
    :param int index: Zero-based preset slot
    :param args: Trailing Maya callback args (unused)
    """
    name = SECTIONS[section]['optionvar'].format(index)
    values = get_section_values(section)
    cmds.optionVar(stringValue=(name, ','.join(repr(v) for v in values)))
    refresh_preset_button(section, index)


def clear_preset(section: str, index: int, *args):
    """Remove a preset slot's saved values (right-click menu).

    :param str section: Section key ('critical' or 'spring')
    :param int index: Zero-based preset slot
    :param args: Trailing Maya callback args (unused)
    """
    name = SECTIONS[section]['optionvar'].format(index)
    if cmds.optionVar(exists=name):
        cmds.optionVar(remove=name)
    refresh_preset_button(section, index)


def build_preset_row(section: str):
    """Build a horizontal row of preset buttons for a solver section.

    :param str section: Section key ('critical' or 'spring')

    .. note::
        Left-click applies the preset; right-click opens a save/clear menu.
        Created button names are recorded on the section for later refresh.
    """
    SECTIONS[section]['buttons'] = []
    cmds.rowLayout(numberOfColumns=PRESET_COUNT, columnAttach=[
        (i + 1, 'both', 2) for i in range(PRESET_COUNT)])
    for i in range(PRESET_COUNT):
        button = cmds.button( label=f'{i + 1}', width=44, height=24
                            , command=partial(load_preset, section, i) )  # pylint: disable=E1111
        cmds.popupMenu( parent=button, button=3 )
        cmds.menuItem( label='Save current values', command=partial(save_preset, section, i) )
        cmds.menuItem( label='Clear saved values', command=partial(clear_preset, section, i) )
        SECTIONS[section]['buttons'].append(button)
    cmds.setParent('..')


def ui():
    """Create and display the SpringyKeys user interface.

    .. note::
        Two solver sections, each with its own bank of preset buttons:

        - Critical Damping: simple interpolation-based smoothing
          (Critical Damping Ratio slider)
        - Spring Physics: full spring-damper simulation
          (Damping Ratio, Halflife, Delta Time sliders)

        Each section's presets store only that section's sliders. Left-click a
        preset to apply that solver to the selection; right-click to save or
        clear. Presets persist across Maya sessions via optionVar.

        Window is recreated if it already exists.
    """
    global SLIDER_FACTOR
    global SLIDER_DAMPING
    global SLIDER_HALFLIFE
    global SLIDER_DT
    global RESTING_STATE
    # Check if window exists and delete it
    if cmds.window("springOverlapWin", exists=True):
        cmds.deleteUI("springOverlapWin")

    window = cmds.window("springOverlapWin", title="SpringyKeys", iconName='springykeys', widthHeight=(760, 160))  # pylint: disable=E1111

    main_layout = cmds.columnLayout( adjustableColumn=True )

    # Critical Damping section: its slider on the left, its presets on the right
    cmds.frameLayout( label='Critical Damping', collapsable=False, marginWidth=4, marginHeight=4 )
    cmds.rowLayout( numberOfColumns=2, adjustableColumn=1
                  , columnAttach=[(1, 'both', 0), (2, 'both', 6)] )
    cmds.columnLayout( adjustableColumn=True )
    SLIDER_FACTOR = cmds.floatSliderGrp( label='Critical Damping Ratio' , field=True, min=0.0, max=1.0, value=DAMPING_FACTOR, step=0.001, dragCommand=update_factor,  changeCommand=complete, adjustableColumn=0  )  # pylint: disable=E1111
    cmds.setParent('..')
    build_preset_row('critical')
    cmds.setParent('..')  # rowLayout
    cmds.setParent('..')  # frameLayout

    # Spring Physics section: its sliders on the left, its presets on the right
    cmds.frameLayout( label='Spring Physics', collapsable=False, marginWidth=4, marginHeight=4 )
    cmds.rowLayout( numberOfColumns=2, adjustableColumn=1
                  , columnAttach=[(1, 'both', 0), (2, 'both', 6)] )
    cmds.columnLayout( adjustableColumn=True )
    SLIDER_DAMPING  = cmds.floatSliderGrp( label='Damping Ratio ', field=True, min=0.001, max=1.0, value=DAMPING_RATIO, step=0.001, dragCommand=update_spring_keys, changeCommand=complete, adjustableColumn=0 )  # pylint: disable=E1111
    SLIDER_HALFLIFE = cmds.floatSliderGrp( label='Halflife' , field=True, min=0.0, max=1.0, value=HALFLIFE, step=0.001, dragCommand=update_spring_keys,  changeCommand=complete, adjustableColumn=0  )  # pylint: disable=E1111
    SLIDER_DT = cmds.floatSliderGrp( label='Delta time (30fps) ' , field=True, min=0.001, max=1.0, value=DELTA_TIME, step=0.001, dragCommand=update_deltatime,  changeCommand=complete, adjustableColumn=0  )  # pylint: disable=E1111
    cmds.setParent('..')
    build_preset_row('spring')
    cmds.setParent('..')  # rowLayout
    cmds.setParent('..')  # frameLayout

    cmds.setParent('..')  # main columnLayout

    for section in SECTIONS:
        for i in range(PRESET_COUNT):
            refresh_preset_button(section, i)

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

    # Open at the size of the content. A columnLayout stacks its children and
    # reports their summed (natural) height regardless of the window size, so
    # matching the window to it avoids both clipping and empty space.
    try:
        content_height = cmds.columnLayout(main_layout, q=True, height=True)
        if content_height and content_height > 0:
            cmds.window(window, e=True, height=content_height)
    except RuntimeError:
        pass


if __name__ == "__main__":
    ui()
