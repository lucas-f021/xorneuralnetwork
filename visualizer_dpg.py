import math
import time
import dearpygui.dearpygui as dpg
from xornn import XORNet, INPUTS, EXPECTED, LEARNING_RATE

SNAPSHOT_INTERVAL = 50
EPOCHS = 5000

# --- pre-training ---

net = XORNet()
net.initialize()
snapshots = []

for epoch in range(EPOCHS):
    for i in range(4):
        net.forward_prop(INPUTS[i][0], INPUTS[i][1])
        net.back_prop(i)
    if epoch % SNAPSHOT_INTERVAL == 0:
        snapshots.append({
            'epoch': epoch,
            'w1': net.w1.copy(),
            'b1': net.b1.copy(),
            'w2': net.w2.copy(),
            'b2': float(net.b2),
            'loss': net.compute_loss(),
        })

print(f"training done -{len(snapshots)} snapshots")

_helper = XORNet()

def snap_forward(snap, x, y):
    _helper.w1 = snap['w1'].copy()
    _helper.b1 = snap['b1'].copy()
    _helper.w2 = snap['w2'].copy()
    _helper.b2 = snap['b2']
    _helper.forward_prop(x, y)
    return _helper.hidden.copy(), float(_helper.prediction)

# --- network diagram constants ---

CANVAS_W, CANVAS_H = 640, 640
NODE_R = 28
NODE_POS_PX = {
    'i0': (75,  180),
    'i1': (75,  460),
    'h0': (320, 180),
    'h1': (320, 460),
    'o0': (565, 320),
}
NODE_LABELS = {'i0': 'x0', 'i1': 'x1', 'h0': 'h0', 'h1': 'h1', 'o0': 'out'}

# (src, dst, weight_getter, label_pos_px, var_name)
# label positions sit above/beside their wire, clear of nodes and crossings
EDGES = [
    ('i0', 'h0', lambda s: s['w1'][0, 0], (175, 155), 'w1[0,0]'),
    ('i0', 'h1', lambda s: s['w1'][0, 1], (90,  222), 'w1[0,1]'),
    ('i1', 'h0', lambda s: s['w1'][1, 0], (90,  398), 'w1[1,0]'),
    ('i1', 'h1', lambda s: s['w1'][1, 1], (175, 438), 'w1[1,1]'),
    ('h0', 'o0', lambda s: s['w2'][0],    (415, 222), 'w2[0]'),
    ('h1', 'o0', lambda s: s['w2'][1],    (415, 390), 'w2[1]'),
]

def edge_color(w):
    return (33, 102, 172, 210) if w >= 0 else (214, 96, 77, 210)

def edge_thickness(w):
    return max(1.0, min(abs(w) * 3.0, 10.0))

# --- state ---

state = {'idx': 0, 'playing': False}
last_advance = time.time()

# --- DPG setup ---

dpg.create_context()

dpg.create_viewport(title='XOR Neural Network - Dear PyGui',
                    width=1350, height=820, resizable=True)

with dpg.window(tag='main', no_title_bar=True, no_move=True, no_scrollbar=True,
                no_resize=True, pos=(0, 0), width=1350, height=820):

    with dpg.group(horizontal=True):

        # ── LEFT: network canvas ──────────────────────────────────────
        with dpg.child_window(tag='left_panel', width=660, height=720, no_scrollbar=True, border=True):
            dpg.add_text('Network', color=(220, 220, 220))
            dpg.add_separator()

            canvas = dpg.add_drawlist(width=CANVAS_W, height=CANVAS_H)

            with dpg.draw_node(tag='net_layer', parent=canvas):

                # layer labels
                dpg.draw_text((40,  18), 'Input',  size=14, color=(210,210,210))
                dpg.draw_text((292, 18), 'Hidden', size=14, color=(210,210,210))
                dpg.draw_text((535, 18), 'Output', size=14, color=(210,210,210))

                # draw all lines first so labels render on top
                for idx, (src, dst, wfn, lpos, wname) in enumerate(EDGES):
                    p1 = NODE_POS_PX[src]
                    p2 = NODE_POS_PX[dst]
                    dpg.draw_line(p1, p2, color=(33,102,172,80), thickness=1,
                                  tag=f'edge_{idx}')

                for idx, (src, dst, wfn, lpos, wname) in enumerate(EDGES):
                    lx, ly = lpos
                    w = len(wname) * 7 + 6
                    dpg.draw_rectangle((lx - 3, ly - 3), (lx + w, ly + 30),
                                       fill=(18, 18, 18, 210), color=(90, 90, 110, 180))

                for idx, (src, dst, wfn, lpos, wname) in enumerate(EDGES):
                    lx, ly = lpos
                    dpg.draw_text((lx, ly),      wname,  size=14, color=(180,200,255), tag=f'wname_{idx}')
                    dpg.draw_text((lx, ly + 15), '+0.00', size=14, color=(255,255,255), tag=f'wval_{idx}')

                # static nodes (drawn on top of edges)
                for key, (nx, ny) in NODE_POS_PX.items():
                    dpg.draw_circle((nx, ny), NODE_R,
                                    color=(220,220,220,255), fill=(255,255,255,255),
                                    thickness=2, tag=f'node_circle_{key}')
                    label = NODE_LABELS[key]
                    tx = nx - (len(label) * 4)
                    dpg.draw_text((tx, ny - 7), label, size=14, color=(20,20,20),
                                  tag=f'node_label_{key}')

                # bias labels (below hidden + output nodes)
                for key in ('h0', 'h1', 'o0'):
                    nx, ny = NODE_POS_PX[key]
                    dpg.draw_text((nx - 22, ny + NODE_R + 4), 'b=0.00',
                                  size=13, color=(200,200,200), tag=f'bias_{key}')

                dpg.draw_text((10, 610), 'Epoch: 0', size=15, color=(220,220,220),
                              tag='epoch_label')

                # single correctness indicator (hidden until an input is selected)
                ox, oy = NODE_POS_PX['o0']
                dpg.draw_text((ox + NODE_R + 8, oy - 8), 'YES', size=15, color=(60,180,60),
                              tag='correct_val', show=False)

                # legend
                dpg.draw_line((430, 605), (465, 605), color=(33, 102, 172, 255), thickness=3)
                dpg.draw_text((470, 598), '= positive weight', size=13, color=(200,200,200))
                dpg.draw_line((430, 625), (465, 625), color=(214, 96, 77, 255),  thickness=3)
                dpg.draw_text((470, 618), '= negative weight', size=13, color=(200,200,200))

        # ── RIGHT: loss plot + test inputs ───────────────────────────
        with dpg.child_window(tag='right_panel', width=650, height=720, no_scrollbar=True, border=True):

            # loss plot
            dpg.add_text('Loss Curve', color=(220, 220, 220))
            dpg.add_separator()
            with dpg.plot(height=370, width=630, tag='loss_plot', no_mouse_pos=True):
                dpg.add_plot_axis(dpg.mvXAxis, label='Epoch', tag='x_axis')
                dpg.add_plot_axis(dpg.mvYAxis, label='Loss (MSE)', tag='y_axis')
                dpg.add_line_series(
                    [s['epoch'] for s in snapshots],
                    [s['loss']  for s in snapshots],
                    parent='y_axis', tag='loss_series')
                # vertical marker as 2-point line series
                dpg.add_line_series([0, 0], [0, 0.3],
                                    parent='y_axis', tag='epoch_marker',
                                    )
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=6)
            dpg.add_text('Pass Walkthrough', color=(220, 220, 220))
            dpg.add_separator()
            dpg.add_spacer(height=6)
            # shown when pass not active
            with dpg.group(tag='pass_launch_group'):
                dpg.add_button(label='Run Pass', tag='btn_pass', width=-1, height=36)
                dpg.add_spacer(height=8)
                dpg.add_text('Select an input, then press Run Pass.',
                             tag='math_placeholder', color=(130, 130, 130))
            # shown when pass is active
            with dpg.group(tag='pass_active_group', show=False):
                with dpg.group(horizontal=True):
                    dpg.add_button(label=' X ', tag='btn_pass_close', width=28, height=22)
                    dpg.add_spacer(width=6)
                    dpg.add_button(label=' < ', tag='btn_pass_prev', width=28, height=22)
                    dpg.add_button(label=' > ', tag='btn_pass_next', width=28, height=22)
                dpg.add_spacer(height=6)
                dpg.add_text('', tag='pass_title', color=(255, 220, 50))
                dpg.add_spacer(height=4)
                for _k in range(10):
                    dpg.add_text('', tag=f'pass_line_{_k}', color=(255, 255, 255))
    dpg.add_spacer(height=6)

    # ── Controls ─────────────────────────────────────────────────────
    with dpg.group(horizontal=True):
        dpg.add_button(label='Play',    tag='btn_play',    width=90, height=34)
        dpg.add_button(label='<- Step', tag='btn_back',    width=90, height=34)
        dpg.add_button(label='Step ->', tag='btn_fwd',     width=90, height=34)
        dpg.add_button(label='Restart', tag='btn_restart', width=90, height=34)
        dpg.add_spacer(width=20)
        dpg.add_text('Speed:')
        dpg.add_slider_int(tag='speed_slider', default_value=300,
                           min_value=50, max_value=600, width=200, format="")
        dpg.add_spacer(width=30)
        dpg.add_text('Show input:')
        dpg.add_combo(['none', '[0,0]', '[0,1]', '[1,0]', '[1,1]'],
                      tag='input_select', default_value='none', width=90)

# --- create red theme for epoch marker ---
with dpg.theme(tag='marker_theme'):
    with dpg.theme_component(dpg.mvLineSeries):
        dpg.add_theme_color(dpg.mvPlotCol_Line, (220, 50, 50, 200), category=dpg.mvThemeCat_Plots)
        dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots)

dpg.bind_item_theme('epoch_marker', 'marker_theme')

# --- update functions ---

_INPUT_OPTIONS = {'none': None, '[0,0]': 0, '[0,1]': 1, '[1,0]': 2, '[1,1]': 3}

def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))

# --- pass animation ---

pass_state = {'active': False, 'steps': [], 'step_idx': 0}

def _set_node_highlight(key, hi):
    if hi:
        dpg.configure_item(f'node_circle_{key}', fill=(255,220,50,255), color=(255,190,20,255))
    else:
        dpg.configure_item(f'node_circle_{key}', fill=(255,255,255,255), color=(220,220,220,255))

def _set_edge_highlight(idx, hi):
    if hi:
        dpg.configure_item(f'edge_{idx}', color=(255,220,50,230), thickness=5)
    else:
        snap = snapshots[state['idx']]
        w = EDGES[idx][2](snap)
        dpg.configure_item(f'edge_{idx}', color=edge_color(w), thickness=edge_thickness(w))

def _apply_pass_step(step):
    for k in NODE_POS_PX:
        _set_node_highlight(k, False)
    for i in range(len(EDGES)):
        _set_edge_highlight(i, False)
    for k in step['nodes']:
        _set_node_highlight(k, True)
    for i in step['edge_idxs']:
        _set_edge_highlight(i, True)
    dpg.configure_item('pass_title', default_value=step['title'])
    for k in range(10):
        line = step['lines'][k] if k < len(step['lines']) else ''
        dpg.configure_item(f'pass_line_{k}', default_value=line)

def _build_pass_steps(snap, i):
    x0, x1 = int(INPUTS[i][0]), int(INPUTS[i][1])
    exp = int(EXPECTED[i])
    w = snap['w1']; b1 = snap['b1']; w2 = snap['w2']; b2 = snap['b2']
    LR = LEARNING_RATE
    z0 = x0*w[0,0] + x1*w[1,0] + b1[0]; h0v = _sigmoid(z0)
    z1 = x0*w[0,1] + x1*w[1,1] + b1[1]; h1v = _sigmoid(z1)
    zo = h0v*w2[0] + h1v*w2[1] + b2;    pred = _sigmoid(zo)
    d_loss = 2*(pred - exp)
    d_sig  = pred*(1 - pred)
    dout   = d_loss * d_sig
    dh0 = dout * w2[0] * h0v*(1 - h0v)
    dh1 = dout * w2[1] * h1v*(1 - h1v)
    return [
        {'title': 'Step 1/6 - Read Inputs',
         'nodes': ['i0', 'i1'], 'edge_idxs': [],
         'lines': [f'x0 = {x0}', f'x1 = {x1}']},
        {'title': 'Step 2/6 - Forward: Hidden Layer',
         'nodes': ['i0', 'i1', 'h0', 'h1'], 'edge_idxs': [0, 1, 2, 3],
         'lines': [
            f'z_h0 = {x0}*({w[0,0]:+.2f}) + {x1}*({w[1,0]:+.2f}) + ({b1[0]:+.2f}) = {z0:+.4f}',
            f'h0   = sigmoid({z0:+.4f}) = {h0v:.4f}',
            '',
            f'z_h1 = {x0}*({w[0,1]:+.2f}) + {x1}*({w[1,1]:+.2f}) + ({b1[1]:+.2f}) = {z1:+.4f}',
            f'h1   = sigmoid({z1:+.4f}) = {h1v:.4f}',
         ]},
        {'title': 'Step 3/6 - Forward: Output',
         'nodes': ['h0', 'h1', 'o0'], 'edge_idxs': [4, 5],
         'lines': [
            f'z_out = {h0v:.3f}*({w2[0]:+.2f}) + {h1v:.3f}*({w2[1]:+.2f}) + ({b2:+.2f}) = {zo:+.4f}',
            f'pred  = sigmoid({zo:+.4f}) = {pred:.4f}',
            f'round = {round(pred)}   expected = {exp}',
            f'error = {pred:.4f} - {exp} = {pred - exp:+.4f}',
         ]},
        {'title': 'Step 4/6 - Backprop: Output Delta',
         'nodes': ['o0'], 'edge_idxs': [],
         'lines': [
            f'd_loss    = 2*(pred - exp) = 2*({pred:.4f} - {exp}) = {d_loss:+.4f}',
            f"sigmoid'  = pred*(1-pred) = {pred:.4f}*{1-pred:.4f} = {d_sig:.4f}",
            f'delta_out = d_loss * sigmoid\' = {d_loss:+.4f} * {d_sig:.4f} = {dout:+.4f}',
         ]},
        {'title': 'Step 5/6 - Backprop: Hidden Deltas',
         'nodes': ['h0', 'h1'], 'edge_idxs': [4, 5],
         'lines': [
            f'delta_h0 = delta_out * w2[0] * h0*(1-h0)',
            f'         = {dout:+.4f} * {w2[0]:+.2f} * {h0v*(1-h0v):.4f} = {dh0:+.4f}',
            '',
            f'delta_h1 = delta_out * w2[1] * h1*(1-h1)',
            f'         = {dout:+.4f} * {w2[1]:+.2f} * {h1v*(1-h1v):.4f} = {dh1:+.4f}',
         ]},
        {'title': 'Step 6/6 - Backprop: Update Weights',
         'nodes': [], 'edge_idxs': [0, 1, 2, 3, 4, 5],
         'lines': [
            f'new = old - learning_rate * gradient  (lr={LEARNING_RATE})',
            f'w2[0]:   {w2[0]:+.4f}  grad={LR*dout*h0v:+.4f}  ->  {w2[0] - LR*dout*h0v:+.4f}',
            f'w2[1]:   {w2[1]:+.4f}  grad={LR*dout*h1v:+.4f}  ->  {w2[1] - LR*dout*h1v:+.4f}',
            f'b2:      {b2:+.4f}  grad={LR*dout:+.4f}  ->  {b2 - LR*dout:+.4f}',
            f'w1[0,0]: {w[0,0]:+.4f}  grad={LR*dh0*x0:+.4f}  ->  {w[0,0] - LR*dh0*x0:+.4f}',
            f'w1[1,0]: {w[1,0]:+.4f}  grad={LR*dh0*x1:+.4f}  ->  {w[1,0] - LR*dh0*x1:+.4f}',
            f'w1[0,1]: {w[0,1]:+.4f}  grad={LR*dh1*x0:+.4f}  ->  {w[0,1] - LR*dh1*x0:+.4f}',
            f'w1[1,1]: {w[1,1]:+.4f}  grad={LR*dh1*x1:+.4f}  ->  {w[1,1] - LR*dh1*x1:+.4f}',
            f'b1[0]:   {b1[0]:+.4f}  grad={LR*dh0:+.4f}  ->  {b1[0] - LR*dh0:+.4f}',
            f'b1[1]:   {b1[1]:+.4f}  grad={LR*dh1:+.4f}  ->  {b1[1] - LR*dh1:+.4f}',
         ]},
    ]

def update_node_labels(snap):
    sel = dpg.get_value('input_select')
    i = _INPUT_OPTIONS.get(sel)
    show = (i is not None)
    dpg.show_item('correct_val') if show else dpg.hide_item('correct_val')

    if not show:
        for key, lbl in NODE_LABELS.items():
            nx, ny = NODE_POS_PX[key]
            dpg.configure_item(f'node_label_{key}', text=lbl,
                               pos=(nx - len(lbl) * 4, ny - 7))
    else:
        hidden, pred = snap_forward(snap, INPUTS[i][0], INPUTS[i][1])
        vals = {
            'i0': str(int(INPUTS[i][0])),
            'i1': str(int(INPUTS[i][1])),
            'h0': f'{hidden[0]:.2f}',
            'h1': f'{hidden[1]:.2f}',
            'o0': f'{pred:.2f}',
        }
        for key, txt in vals.items():
            nx, ny = NODE_POS_PX[key]
            dpg.configure_item(f'node_label_{key}', text=txt,
                               pos=(nx - len(txt) * 4, ny - 7))
        correct = round(pred) == int(EXPECTED[i])
        dpg.configure_item('correct_val',
                           text='YES' if correct else 'NO',
                           color=(60,180,60,255) if correct else (210,60,60,255))

def update_display():
    snap = snapshots[state['idx']]

    # edges
    for idx, (src, dst, wfn, lpos, wname) in enumerate(EDGES):
        w = wfn(snap)
        dpg.configure_item(f'edge_{idx}',
                           color=edge_color(w),
                           thickness=edge_thickness(w))
        dpg.configure_item(f'wval_{idx}', text=f'{w:+.2f}')

    # biases
    dpg.configure_item('bias_h0', text=f'b={snap["b1"][0]:.2f}')
    dpg.configure_item('bias_h1', text=f'b={snap["b1"][1]:.2f}')
    dpg.configure_item('bias_o0', text=f'b={snap["b2"]:.2f}')

    # epoch label + loss vline
    dpg.configure_item('epoch_label', text=f'Epoch: {snap["epoch"]}')
    loss_vals = [s['loss'] for s in snapshots]
    dpg.set_value('epoch_marker', [[snap['epoch'], snap['epoch']],
                                   [0, max(loss_vals)]])

    update_node_labels(snap)


def retrain():
    net.initialize()
    snapshots.clear()
    for epoch in range(EPOCHS):
        for i in range(4):
            net.forward_prop(INPUTS[i][0], INPUTS[i][1])
            net.back_prop(i)
        if epoch % SNAPSHOT_INTERVAL == 0:
            snapshots.append({
                'epoch': epoch,
                'w1': net.w1.copy(),
                'b1': net.b1.copy(),
                'w2': net.w2.copy(),
                'b2': float(net.b2),
                'loss': net.compute_loss(),
            })
    dpg.set_value('loss_series', [
        [s['epoch'] for s in snapshots],
        [s['loss']  for s in snapshots],
    ])
    dpg.fit_axis_data('x_axis')
    dpg.fit_axis_data('y_axis')

# --- button callbacks ---

def on_play(s, a, u):
    if pass_state['active']: _end_pass()
    state['playing'] = not state['playing']
    dpg.configure_item('btn_play', label='Pause' if state['playing'] else 'Play')

def on_fwd(s, a, u):
    if pass_state['active']: _end_pass()
    state['idx'] = min(state['idx'] + 1, len(snapshots) - 1)
    update_display()

def on_back(s, a, u):
    if pass_state['active']: _end_pass()
    state['idx'] = max(state['idx'] - 1, 0)
    update_display()

def on_restart(s, a, u):
    if pass_state['active']: _end_pass()
    state['playing'] = False
    dpg.configure_item('btn_play', label='Play')
    retrain()
    state['idx'] = 0
    update_display()

def on_input_select(s, a, u):
    if pass_state['active']: _end_pass()
    update_node_labels(snapshots[state['idx']])

def on_run_pass(s, a, u):
    sel = dpg.get_value('input_select')
    i = _INPUT_OPTIONS.get(sel)
    if i is None:
        i = 1  # default to [0,1]
        dpg.set_value('input_select', '[0,1]')
    snap = snapshots[state['idx']]
    pass_state['steps']    = _build_pass_steps(snap, i)
    pass_state['step_idx'] = 0
    pass_state['active']   = True
    state['playing'] = False
    dpg.configure_item('btn_play', label='Play')
    dpg.hide_item('pass_launch_group')
    dpg.show_item('pass_active_group')
    _apply_pass_step(pass_state['steps'][0])

def _end_pass():
    pass_state['active'] = False
    for _k in NODE_POS_PX:
        _set_node_highlight(_k, False)
    for _i in range(len(EDGES)):
        _set_edge_highlight(_i, False)
    dpg.hide_item('pass_active_group')
    dpg.show_item('pass_launch_group')
    update_node_labels(snapshots[state['idx']])

def on_pass_next(s, a, u):
    if not pass_state['active']:
        return
    pass_state['step_idx'] += 1
    if pass_state['step_idx'] >= len(pass_state['steps']):
        _end_pass()
    else:
        _apply_pass_step(pass_state['steps'][pass_state['step_idx']])

def on_pass_prev(s, a, u):
    if not pass_state['active']:
        return
    pass_state['step_idx'] = max(0, pass_state['step_idx'] - 1)
    _apply_pass_step(pass_state['steps'][pass_state['step_idx']])

def on_pass_close(s, a, u):
    _end_pass()

dpg.set_item_callback('btn_play',      on_play)
dpg.set_item_callback('btn_fwd',       on_fwd)
dpg.set_item_callback('btn_back',      on_back)
dpg.set_item_callback('btn_restart',   on_restart)
dpg.set_item_callback('input_select',  on_input_select)
dpg.set_item_callback('btn_pass',       on_run_pass)
dpg.set_item_callback('btn_pass_next',  on_pass_next)
dpg.set_item_callback('btn_pass_prev',  on_pass_prev)
dpg.set_item_callback('btn_pass_close', on_pass_close)

# --- resize callback ---

def on_viewport_resize():
    vw = dpg.get_viewport_client_width()
    vh = dpg.get_viewport_client_height()
    left_w  = vw // 2 - 5
    right_w = vw - left_w - 15
    panel_h = vh - 90
    dpg.configure_item('left_panel',  width=left_w,  height=panel_h)
    dpg.configure_item('right_panel', width=right_w, height=panel_h)
    dpg.configure_item('loss_plot',   width=right_w - 22,
                       height=max(370, panel_h - 280))
    avail_w = left_w - 20
    avail_h = panel_h - 50
    scale = min(avail_w / CANVAS_W, avail_h / CANVAS_H)
    dpg.configure_item(canvas, width=avail_w, height=avail_h)
    dpg.apply_transform('net_layer', dpg.create_scale_matrix([scale, scale]))

dpg.set_viewport_resize_callback(on_viewport_resize)

# --- render loop ---

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window('main', True)
on_viewport_resize()

update_display()

while dpg.is_dearpygui_running():
    if state['playing']:
        interval_s = (650 - dpg.get_value('speed_slider')) / 1000.0
        now = time.time()
        if now - last_advance >= interval_s:
            last_advance = now
            state['idx'] = min(state['idx'] + 1, len(snapshots) - 1)
            update_display()
            if state['idx'] >= len(snapshots) - 1:
                state['playing'] = False
                dpg.configure_item('btn_play', label='Play')

    dpg.render_dearpygui_frame()

dpg.destroy_context()
