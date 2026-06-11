import time
import dearpygui.dearpygui as dpg
from xornn import XORNet, INPUTS, EXPECTED

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

print(f"training done — {len(snapshots)} snapshots")

_helper = XORNet()

def snap_forward(snap, x, y):
    _helper.w1 = snap['w1']
    _helper.b1 = snap['b1']
    _helper.w2 = snap['w2']
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
EDGES = [
    ('i0', 'h0', lambda s: s['w1'][0, 0], (197, 157), 'w1[0,0]'),
    ('i0', 'h1', lambda s: s['w1'][0, 1], (45, 248), 'w1[0,1]'),
    ('i1', 'h0', lambda s: s['w1'][1, 0], (45, 372), 'w1[1,0]'),
    ('i1', 'h1', lambda s: s['w1'][1, 1], (197, 463), 'w1[1,1]'),
    ('h0', 'o0', lambda s: s['w2'][0],    (442, 228), 'w2[0]'),
    ('h1', 'o0', lambda s: s['w2'][1],    (442, 412), 'w2[1]'),
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
                    dpg.draw_text((lx, ly),      wname,  size=14, color=(180,200,255), tag=f'wname_{idx}')
                    dpg.draw_text((lx, ly + 15), '+0.00', size=14, color=(255,255,255), tag=f'wval_{idx}')

                # static nodes (drawn on top of edges)
                for key, (nx, ny) in NODE_POS_PX.items():
                    dpg.draw_circle((nx, ny), NODE_R,
                                    color=(220,220,220,255), fill=(255,255,255,255),
                                    thickness=2)
                    label = NODE_LABELS[key]
                    tx = nx - (len(label) * 4)
                    dpg.draw_text((tx, ny - 7), label, size=14, color=(20,20,20))

                # bias labels (below hidden + output nodes)
                for key in ('h0', 'h1', 'o0'):
                    nx, ny = NODE_POS_PX[key]
                    dpg.draw_text((nx - 22, ny + NODE_R + 4), 'b=0.00',
                                  size=13, color=(200,200,200), tag=f'bias_{key}')

                dpg.draw_text((10, 610), 'Epoch: 0', size=15, color=(220,220,220),
                              tag='epoch_label')

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
            dpg.bind_item_theme('epoch_marker',
                dpg.add_theme(tag='marker_theme') if False else 'marker_theme')

            dpg.add_spacer(height=8)
            dpg.add_separator()
            dpg.add_spacer(height=6)

            # test inputs table
            dpg.add_text('Test Inputs', color=(220, 220, 220))
            dpg.add_separator()
            dpg.add_spacer(height=4)

            COL_HEADERS = ['Input', 'h0', 'h1', 'out', 'correct?']
            with dpg.table(header_row=True, borders_innerH=True, borders_innerV=True,
                           borders_outerH=True, borders_outerV=True, width=620):
                for h in COL_HEADERS:
                    dpg.add_table_column(label=h)
                for i in range(4):
                    with dpg.table_row():
                        lbl = f'[{int(INPUTS[i][0])},{int(INPUTS[i][1])}]'
                        dpg.add_text(lbl)
                        dpg.add_text('0.00', tag=f'cell_h0_{i}')
                        dpg.add_text('0.00', tag=f'cell_h1_{i}')
                        dpg.add_text('0.00', tag=f'cell_out_{i}')
                        dpg.add_text('?',    tag=f'cell_pred_{i}')

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

# --- create red theme for epoch marker ---
with dpg.theme(tag='marker_theme'):
    with dpg.theme_component(dpg.mvLineSeries):
        dpg.add_theme_color(dpg.mvPlotCol_Line, (220, 50, 50, 200), category=dpg.mvThemeCat_Plots)
        dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots)

dpg.bind_item_theme('epoch_marker', 'marker_theme')

# --- update functions ---

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

    # test inputs
    for i in range(4):
        hidden, pred = snap_forward(snap, INPUTS[i][0], INPUTS[i][1])
        rounded = round(pred)
        correct = rounded == int(EXPECTED[i])
        marker = 'YES' if correct else 'NO'
        ok_col  = (60, 180, 60, 255)
        bad_col = (210, 60, 60, 255)
        dpg.configure_item(f'cell_h0_{i}',   default_value=f'{hidden[0]:.2f}')
        dpg.configure_item(f'cell_h1_{i}',   default_value=f'{hidden[1]:.2f}')
        dpg.configure_item(f'cell_out_{i}',  default_value=f'{pred:.2f}')
        dpg.configure_item(f'cell_pred_{i}', default_value=marker,
                           color=ok_col if correct else bad_col)

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

# --- button callbacks ---

def on_play(s, a, u):
    state['playing'] = not state['playing']
    dpg.configure_item('btn_play', label='Pause' if state['playing'] else 'Play')

def on_fwd(s, a, u):
    state['idx'] = min(state['idx'] + 1, len(snapshots) - 1)
    update_display()

def on_back(s, a, u):
    state['idx'] = max(state['idx'] - 1, 0)
    update_display()

def on_restart(s, a, u):
    state['playing'] = False
    dpg.configure_item('btn_play', label='Play')
    retrain()
    state['idx'] = 0
    update_display()

dpg.set_item_callback('btn_play',    on_play)
dpg.set_item_callback('btn_fwd',     on_fwd)
dpg.set_item_callback('btn_back',    on_back)
dpg.set_item_callback('btn_restart', on_restart)

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
