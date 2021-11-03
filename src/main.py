import taichi as ti
from wcsph import *

# Default run on CPU
# cuda performance has not been tested
# ti.init(arch=ti.gpu, device_memory_fraction=0.8, debug=True)
ti.init(arch=ti.gpu, device_memory_fraction=0.8)

if __name__ == '__main__':    
    SPH = WCSPH

    max_frame = 50000

    res = (400, 400)
    screen_to_world_ratio = 35
    alpha=0.5

    gui = ti.GUI('SPH', res, background_color=0x112F41)
    sph = SPH(ti.Vector([[0, res[1]], [0, res[0]]]), 
                    screen_to_world_ratio,
                    alpha=alpha)

    colors = [0xED553B, 0x068587, 0xEEEEF0, 0xFFFF00]

    add_r = 50
    add_color_id = 1
    frame = 0
    save_frame = 0
    while frame < max_frame:

        for e in gui.get_events():
            if gui.is_pressed(ti.GUI.LMB, ti.GUI.LEFT): # add object
                mouse_x, mouse_y = gui.get_cursor_pos()
                x_r = mouse_x * res[0]
                y_r = mouse_y * res[1]
                if 0 < x_r-add_r and x_r+add_r < res[0] and 0 < y_r-add_r and y_r+add_r < res[1]:
                    if np.random.rand() < 0.5: # add a cube
                        sph.add_cube(np.array([[x_r-add_r, x_r+add_r], 
                                            [y_r-add_r, y_r+add_r]]),
                                    ti.Vector([0.0, -5.0]), add_color_id, 300)
                    else:
                        sph.add_circle(np.array([x_r, y_r]), add_r, 
                                    ti.Vector([0.0, -5.0]), add_color_id, 300)
                    add_color_id = (add_color_id + 1) % len(colors)
                print(mouse_x, mouse_y)
            if gui.is_pressed('r'):
                sph = SPH(ti.Vector([[0, res[1]], [0, res[0]]]), 
                    screen_to_world_ratio,
                    alpha=alpha)
    
        for i in range(40):
            sph.step()
        particles = sph.get_positions()

        gui.circles(particles, 
                    radius=1.5,
                    palette=colors,
                    palette_indices=sph.get_color_indices())

        # if frame % 50 == 0:
        #     gui.show(f'{save_frame:06d}.png')
        #     save_frame += 1

        gui.show()
        frame += 1
    print('done')
