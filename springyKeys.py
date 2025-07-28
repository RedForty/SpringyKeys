""" Runs springy filter function on selected keyframes in the graph editor """
import math
from maya import cmds, mel


SNAPSHOT = False
KEY_DATA = {}
DAMPING_RATIO = 0.4
HALFLIFE = 0.25
DELTA_TIME = 1 / 30.0
DAMPING_FACTOR = 0.1
TIMELINE = mel.eval('string $tmpString=$gPlayBackSlider') # pylint: disable=E1111
GRAPH_EDITOR = 'graphEditor1GraphEd'


# Private ------------------------------------------------------------------- #

def is_equal(lst): # pylint: disable=C0116
    """ Returns bool if every item in the list is equal """
    return not lst or lst.count(lst[0]) == len(lst)

def get_timeline_selection():
    if cmds.timeControl(TIMELINE, q=True, rangeVisible=True):
        return cmds.timeControl(TIMELINE, q=True, rangeArray=True)
    else:
        return []

def get_timeline_range():
    start_frame = cmds.playbackOptions(q=True, minTime=True)  # pylint: disable=E1111
    end_frame = cmds.playbackOptions(q=True, maxTime=True)    # pylint: disable=E1111
    return [start_frame, end_frame]

def get_work_time(node=None):

    keyframes = []

    timeline_selection = get_timeline_selection() # Selection takes priority
    if timeline_selection:
        timeline_selection = [x * 1.0 for x in range(int(min(timeline_selection)), int(max(timeline_selection))+1)]

    if node:
        keyframes = cmds.keyframe(node, q=True) or []
        keyframes = list(set(keyframes)) # All the keys

    time_range = get_timeline_range()
    start_frame      = min(time_range + keyframes)
    end_frame        = max(time_range + keyframes)
    keyframes = [x * 1.0 for x in range(int(start_frame), int(end_frame)+1)]

    if timeline_selection:
        new_keyframes = []
        for key in keyframes:
            if key in timeline_selection:
                new_keyframes.append(key)
        keyframes = new_keyframes # Crop to selection

    return keyframes

def get_selected_keyframe_data():
    ''' Returns a dictionary where the keys are curve names and the values are
        [times, values]
        ie, {'curve_name' : [times, values]}
    '''

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
        num_keys       = cmds.keyframe(curve, q=True, keyframeCount=True)


        time_range = []
        time_range.extend([float(x) for x in range(int(selected_times[0]), int(selected_times[-1]))])

        # Extend 1 frame in either direction to act as pivot?
        # time_range.insert(selected_times[0] - 1, 0)
        # time_range.append(selected_times[-1] + 1)

        value_range = []
        for time in time_range:
            value = cmds.keyframe(curve, q=True, time=(time, ), eval=True, valueChange=True)
            value_range.extend(value)

        if is_equal(value_range): continue # Ignore flat curves

        key_data[curve] = { "times" : time_range
                          , "values" : value_range
                          }

    if key_data:
        return key_data
    else:
        return None



def apply_values(curve, times, values):
    # Do the magic, do the magic!
    for time, value in zip(times, values):
        cmds.keyframe(curve, e=True, time=(time,), valueChange=value)


def damping_ratio_to_stiffness(ratio: float, damping: float):
    result = (damping / (ratio * 2.0))**2
    return result

def damping_ratio_to_damping(ratio: float, stiffness: float)    :
    result = ratio * 2.0 * (stiffness ** 0.5)
    return result

def halflife_to_damping(halflife: float, eps: float = 1e-5):
    result = (4.0 * 0.69314718056) / (halflife + eps)
    return result


def fast_atan(x: float):
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
    return 1.0 / (1.0 + x + 0.48*x*x + 0.235*x*x*x)

def lerp(x: float, y: float, a: float):
    return (1.0 - a) * x + a * y

def damper(x: float, g: float, factor: float):
    return lerp(x, g, factor)

def update_factor(*args):
    begin()
    slider_value = args[0]
    factor = slider_value

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
    eps: float = 1e-5,
    ):

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

def update_spring_keys(*args): # pylint: disable=W0613
    """ Process the sliders to update the keyframe values """
    begin()
    damping_ratio = cmds.floatSliderGrp(SLIDER_DAMPING, q=True, value=True)  # pylint: disable=E1111
    halflife = cmds.floatSliderGrp(SLIDER_HALFLIFE, q=True, value=True)      # pylint: disable=E1111

    if not KEY_DATA:
        return
    curves = KEY_DATA.keys()

    for curve in curves:
        new_values = []
        current_x = KEY_DATA[curve]["values"][0]
        current_v = 0.0
        for idx, time in enumerate(KEY_DATA[curve]["times"]):
            current_x, current_v = spring_damper_exact_ratio(   current_x,
                                                                current_v,
                                                                KEY_DATA[curve]["values"][idx],
                                                                0.0,
                                                                damping_ratio,
                                                                halflife,
                                                                DELTA_TIME
                                                              )
            new_values.append(current_x)

            # cmds.setKeyframe("pCube1.rotateY", time=(time, ), value=current_x)

        apply_values(curve, KEY_DATA[curve]["times"], new_values)


def update_deltatime(*args):
    global DELTA_TIME
    slider_value = args[0]
    DELTA_TIME = slider_value
    update_spring_keys()
    framerate = round(1/DELTA_TIME, 2)
    cmds.floatSliderGrp(SLIDER_DT, e=True, label=f'Delta time ({framerate}fps) ')



def begin():
    global SNAPSHOT
    if not SNAPSHOT:
        cmds.undoInfo(openChunk=True)
        global KEY_DATA
        KEY_DATA = get_selected_keyframe_data()
        SNAPSHOT = True

def complete(*args): # pylint: disable=W0613
    # end() # This resets the KEY_DATA snapshot
    pass

def end():
    global SNAPSHOT
    global KEY_DATA
    SNAPSHOT = False
    KEY_DATA = None
    cmds.undoInfo(closeChunk=True)


def ui():
    global SLIDER_FACTOR
    global SLIDER_DAMPING
    global SLIDER_HALFLIFE
    global SLIDER_DT
    # Check if window exists and delete it
    if cmds.window("springOverlapWin", exists=True):
        cmds.deleteUI("springOverlapWin")

    window = cmds.window("springOverlapWin", title="Spring Keys", iconName='springkeys', widthHeight=(600, 106))  # pylint: disable=E1111
    cmds.columnLayout( adjustableColumn=True, margins=6 )
    SLIDER_FACTOR = cmds.floatSliderGrp( label='Critical Damping Ratio' , field=True, min=0.0, max=1.0, value=DAMPING_FACTOR, step=0.001, dragCommand=update_factor,  changeCommand=complete, adjustableColumn=0  )  # pylint: disable=E1111
    cmds.separator()
    SLIDER_DAMPING  = cmds.floatSliderGrp( label='Damping Ratio ', field=True, min=0.001, max=1.0, value=DAMPING_RATIO, step=0.001, dragCommand=update_spring_keys, changeCommand=complete, adjustableColumn=0 )  # pylint: disable=E1111
    SLIDER_HALFLIFE = cmds.floatSliderGrp( label='Halflife' , field=True, min=0.0, max=1.0, value=HALFLIFE, step=0.001, dragCommand=update_spring_keys,  changeCommand=complete, adjustableColumn=0  )  # pylint: disable=E1111
    SLIDER_DT = cmds.floatSliderGrp( label='Delta time (30fps) ' , field=True, min=0.0, max=1.0, value=DELTA_TIME, step=0.001, dragCommand=update_deltatime,  changeCommand=complete, adjustableColumn=0  )  # pylint: disable=E1111
    cmds.showWindow(window)


if __name__ == "__main__":
    ui()
