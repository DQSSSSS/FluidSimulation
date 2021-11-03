import numpy as np
import taichi as ti
from smooth import W_cubic

@ti.data_oriented
class WCSPH:

    def __init__(self,
                 bound, screen_to_world_ratio,
                 alpha):
        self.use_neighbor = True

        self.screen_to_world_ratio = screen_to_world_ratio
        self.dim = 2

        self.g = ti.Vector([0.0, -9.80], dt=ti.f32)  # Gravity
        self.alpha = alpha  # viscosity
        self.rho_0 = 1000.0  # reference density

        self.d_radius = bound[0, 1] / screen_to_world_ratio * 0.01  # Particle radius
        self.d_h = self.d_radius * 1.3  # Smooth length
        self.padding = 2 * self.d_radius
        self.m = self.d_radius**self.dim * self.rho_0
        print("d_radius:", self.d_radius)

        self.bound = bound / screen_to_world_ratio
        print(self.bound)

        self.gamma = 7.0
        self.c_0 = 200.0
        self.B = self.rho_0 * self.c_0 ** 2 / self.gamma

        # self.dt = 0.1 * self.d_h / self.c_0 / 2
        self.dt = 4.52 * 1e-5
        print(self.dt)

        self.n_particles = ti.field(ti.i32, shape=())
        self.x = ti.Vector.field(self.dim, dtype=ti.f32)
        self.v = ti.Vector.field(self.dim, dtype=ti.f32)
        self.rho = ti.field(dtype=ti.f32)
        self.p = ti.field(dtype=ti.f32)
        self.color_id = ti.field(dtype=ti.f32)

        self.d_v = ti.Vector.field(self.dim, dtype=ti.f32)
        self.d_rho = ti.field(dtype=ti.f32)

        ti.root.dense(ti.i, 2**15).place(
            self.x, self.v, self.p, self.rho,
            self.d_v, self.d_rho, self.color_id)
        
        # grid info
        if self.use_neighbor:
            self.GRID_MAX_NUM = 100
            self.grid_num = ti.field(dtype=ti.i32)
            self.grid_particle_indices = ti.field(dtype=ti.i32)
            self.max_grid_pos = ti.Vector(((np.array(self.bound)[:, 1] - np.array(self.bound)[:, 0]) / (2 * self.d_h) + 1).astype(np.int))
            ti.root.dense(ti.ij, self.max_grid_pos).place(self.grid_num)
            ti.root.dense(ti.ij, self.max_grid_pos).dense(ti.k, self.GRID_MAX_NUM).place(self.grid_particle_indices)
            print("max pos:", self.max_grid_pos)

        # self.initialize()
        # self.add_cube(np.array([[100, 300], [100, 300]]), ti.Vector([0.0, -5.0]), 0, 1000)
        self.add_circle(np.array([200, 200]), 200, ti.Vector([0.0, -5.0]), 0, 1000)

    @ti.kernel
    def add_cube(self, bound_ori:ti.ext_arr(), v:ti.template(), color_id:ti.i32, num:ti.i32):
        # num = 1000
        for i in range(self.n_particles[None], self.n_particles[None] + num):
            x = ti.Vector([0.0] * self.dim)
            for j in ti.static(range(self.dim)):
                l = bound_ori[j, 0] / self.screen_to_world_ratio
                r = bound_ori[j, 1] / self.screen_to_world_ratio
                x[j] = float(ti.random() * (r-l) + l)
            # print("i: ", self.x[i].n, self.x[i].m, x.n, x.m)
            self.x[i] = x
            self.rho[i] = 1000
            self.v[i] = v
            self.color_id[i] = color_id
        self.n_particles[None] += num

    @ti.kernel
    def add_circle(self, center:ti.ext_arr(), r_ori:ti.f32, v:ti.template(), color_id:ti.i32, num:ti.i32):
        assert self.dim == 2
        # num = 1000
        x_0 = center[0] / self.screen_to_world_ratio
        x_1 = center[1] / self.screen_to_world_ratio
        r = r_ori / self.screen_to_world_ratio
        for i in range(self.n_particles[None], self.n_particles[None] + num):
            r_p = ti.sqrt(ti.random()) * r
            theta = ti.random() * np.pi * 2
            x = ti.Vector([x_0 + r_p*ti.cos(theta), x_1 + r_p*ti.sin(theta)])
            # print("i: ", self.x[i].n, self.x[i].m, x.n, x.m)
            self.x[i] = x
            self.rho[i] = 1000
            self.v[i] = v
            self.color_id[i] = color_id
        self.n_particles[None] += num

    @ti.func
    def ask_grid_id(self, pos):
        for i in ti.static(range(self.dim)):
            pos[i] -= self.bound[i, 0]
        return (pos / (2 * self.d_h)).cast(int)

    @ti.kernel
    def build_neighbor(self):
        for I in ti.grouped(ti.ndrange(self.max_grid_pos[0], self.max_grid_pos[1])):
            self.grid_num[I] = 0
        for i in range(self.n_particles[None]):
            grid_id = self.ask_grid_id(self.x[i])
            self.grid_num[grid_id] += 1
            self.grid_particle_indices[grid_id, self.grid_num[grid_id]-1] = i
        pass    
    
    @ti.func
    def ask_dv_drho(self, i, j):
        d_v, d_rho = ti.Vector([0.0] * self.dim), 0.0
        r = self.x[i] - self.x[j]
        r_len = ti.max(r.norm(), 1e-5)

        # Compute Density change
        d_rho += self.m * W_cubic.W_grad(r, r_len, self.d_h) \
            * (self.v[i] - self.v[j]).dot(r / r_len)

        # viscosity force
        if (self.v[i] - self.v[j]).dot(r) < 0:
            vmu = -2.0 * self.alpha * self.d_radius * self.c_0 / (self.rho[i] + self.rho[j])
            d_v += -self.m * vmu * (self.v[i] - self.v[j]).dot(r) / (r_len**2 + 0.01 * self.d_radius**2) \
                * W_cubic.W_grad(r, r_len, self.d_h) * r / r_len
        
        # pressure force
        d_v += -self.m * (self.p[i] / self.rho[i] ** 2 + self.p[j] / self.rho[j] ** 2) \
            * W_cubic.W_grad(r, r_len, self.d_h) * r / r_len
        return d_v, d_rho

    @ti.kernel
    def pre_compute_neighbor(self):
        for i in range(self.n_particles[None]):
            d_v = ti.Vector([0.0] * self.dim)
            d_rho = 0.0
            now_grid_id = self.ask_grid_id(self.x[i])
            for off in ti.static(ti.grouped(ti.ndrange(*((-1, 2), ) * self.dim))):
                grid_id = now_grid_id + off
                flag = True
                for xx in ti.static(range(self.dim)):
                    if grid_id[xx] < 0 or grid_id[xx] >= self.max_grid_pos[xx]:
                        flag = False
                # if not flag:
                # print(i, now_grid_id, grid_id)
                if flag:
                    for t_j in range(self.grid_num[grid_id]):
                        j = self.grid_particle_indices[grid_id, t_j]
                        d_v_t, d_rho_t = self.ask_dv_drho(i, j)
                        d_v += d_v_t
                        d_rho += d_rho_t
            # gravity force
            d_v += self.g
            self.d_v[i] = d_v
            self.d_rho[i] = d_rho

    @ti.kernel
    def pre_compute(self):
        for i in range(self.n_particles[None]):
            d_v = ti.Vector([0.0] * self.dim)
            d_rho = 0.0
            for j in range(self.n_particles[None]):
                d_v_t, d_rho_t = self.ask_dv_drho(i, j)
                d_v += d_v_t
                d_rho += d_rho_t
            # gravity force
            d_v += self.g
            self.d_v[i] = d_v
            self.d_rho[i] = d_rho

    @ti.kernel
    def update(self):
        for i in range(self.n_particles[None]):
            self.v[i] += self.dt * self.d_v[i]
            self.x[i] += self.dt * self.v[i]
            self.rho[i] += self.dt * self.d_rho[i]
            self.p[i] = self.B * ((self.rho[i] / self.rho_0) ** self.gamma - 1.0)

            # limit the max v
            if self.v[i].norm() > 10:
                self.v[i] *= 1 / self.v[i].norm() * 10;

        for i in range(self.n_particles[None]):
            pos = self.x[i]
            for j in ti.static(range(self.dim)):
                l, r = self.bound[j, 0], self.bound[j, 1]
                if pos[j] < l + self.padding:
                    self.x[i][j] = l + self.padding
                    self.v[i][j] *= -0.3
                if pos[j] > r - self.padding:
                    self.x[i][j] = r - self.padding
                    self.v[i][j] *= -0.3 

    def step(self):
        if self.use_neighbor:
            self.build_neighbor()
            self.pre_compute_neighbor()
        else:
            self.pre_compute()
        self.update()

    @ti.kernel
    def copy_dynamic_nd(self, np_x: ti.ext_arr(), input_x: ti.template()):
        for i in range(self.n_particles[None]):
            for j in ti.static(range(self.dim)):
                np_x[i, j] = input_x[i][j]
    
    @ti.kernel
    def copy_dynamic(self, np_x: ti.ext_arr(), input_x: ti.template()):
        for i in range(self.n_particles[None]):
            np_x[i] = input_x[i]
    
    def get_positions(self):
        np_x = np.ndarray((self.n_particles[None], self.dim),
                          dtype=np.float32)
        self.copy_dynamic_nd(np_x, self.x)
        return np_x / np.array(self.bound)[:, 1]
        
    def get_color_indices(self):
        np_x = np.ndarray((self.n_particles[None], ), dtype=np.int32)
        self.copy_dynamic(np_x, self.color_id)
        return np_x

    def get_grid_lines(self):
        # [[s_x, s_y], ...] , [[t_x, t_y], ...]
        ans_s, ans_t = [], []
        for i in range(self.dim):
            for j in range(1, int(self.bound[i, 1] / (2 * self.d_h)) + 1):
                t = j * 2 * self.d_h
                t_lp = np.array(self.bound)[:, 0]
                t_rp = np.array(self.bound)[:, 1]
                t_lp[i] = t_rp[i] = t
                ans_s.append(t_lp)
                ans_t.append(t_rp)
        return np.array(ans_s) / np.array(self.bound)[:, 1], np.array(ans_t) / np.array(self.bound)[:, 1]