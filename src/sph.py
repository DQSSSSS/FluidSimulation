from os import X_OK
import taichi as ti
from smooth import *

PI = 3.1415926

@ti.data_oriented
class SPH:

    def __init__(self, 
                 n_particles,
                 dim=2, 
                 gravity=ti.Vector([0, -9.8]), 
                 gas_constant=1,
                 rest_density=1000000,
                 mass=1,
                 surface_intension_factor=0.1,
                 h = 0.02,
                 viscosity_mu=0.2,
                 normal_threshold=5.00
                  ) -> None:
        self.n_particles = n_particles        

        self.g = gravity
        self.sigma = surface_intension_factor
        self.rho_0 = rest_density
        self.p_k = gas_constant
        self.h = h
        self.dim = dim
        self.mu = viscosity_mu
        self.n_threshold = normal_threshold
        self.mass = mass

        self.x = ti.Vector.field(dim, dtype=float, shape=n_particles)  # position
        self.v = ti.Vector.field(dim, dtype=float, shape=n_particles)  # velocity
        self.rho = ti.field(dtype=float, shape=n_particles)  # density
        self.p = ti.field(dtype=float, shape=n_particles)  # pressure
        self.f = ti.Vector.field(dim, dtype=float, shape=n_particles)  # force
        self.m = ti.field(dtype=float, shape=n_particles) # mass
        self.is_surface = ti.field(dtype=int, shape=n_particles) # mark surface

        self.initialize()

    @ti.kernel
    def initialize(self):
        for i in range(self.n_particles):
            self.m[i] = self.mass
            self.x[i] = [
                (ti.random() - 0.5) * 0.5 + 0.5,
                (ti.random() - 0.5) * 0.5 + 0.5
            ]
            self.v[i] = [0.0, 0.0]

    @ti.kernel
    def step(self, dt:ti.f32):
        self.pre_compute()
        self.update(dt)
    
    @ti.func
    def pre_compute(self):
        
        # clear F
        for i in range(self.n_particles):
            self.f[i] = ti.Vector([0.0] * self.dim)

        # compute density and pressure
        for i in range(self.n_particles):
            rho = 0.0
            for j in range(self.n_particles):
                rho += self.m[j] * W_poly6.W(self.x[i]-self.x[j], self.h)
            self.rho[i] = rho
            self.p[i] = self.p_k * (rho - self.rho_0)
        
        # print(self.p[0])

        # compute pressure force
        for i in range(self.n_particles):
            f_p = ti.Vector([0.0] * self.dim)
            for j in range(self.n_particles):
                if i == j:
                    continue
                f_p += -self.m[j] * ((self.p[i] + self.p[j]) / self.rho[j] / 2) \
                    * W_spiky.W_grad(self.x[i] - self.x[j], self.h)

                # f_p += -self.rho[i] \
                #     * (self.p[i] / self.rho[i] ** 2 + self.p[j] / self.rho[j] ** 2) \
                #         * self.m[j] * W_spiky.W_grad(self.x[i] - self.x[j], self.h)

                # print(i, j, f_p, self.x[i] - self.x[j], self.h, W_spiky.W_grad(self.x[i] - self.x[j], self.h))
            self.f[i] += f_p

        print("pressure force", self.f[0])

        # compute viscosity
        for i in range(self.n_particles):
            tmp = ti.Vector([0.0] * self.dim)
            for j in range(self.n_particles):
                tmp += self.m[j] * ((self.v[j] - self.v[i]) / self.rho[j]) \
                    * W_viscosity.W_grad2(self.x[i] - self.x[j], self.h)
            self.f[i] += self.mu * tmp

        print("viscosity", self.f[0])

        # compute surface tension
        for i in range(self.n_particles):
            n = ti.Vector([0.0] * self.dim)
            for j in range(self.n_particles):
                n += self.m[j] / self.rho[j] \
                    * W_poly6.W_grad(self.x[i] - self.x[j], self.h)
            n_len = ti.sqrt(n.dot(n))

            self.is_surface[i] = 0
            if n_len > self.n_threshold: # surface
                self.is_surface[i] = 1
                tmp = ti.Vector([0.0] * self.dim)
                for j in range(self.n_particles):
                    tmp += self.m[j] / self.rho[j] \
                        * W_poly6.W_grad2(self.x[i] - self.x[j], self.h)
                self.f[i] += -self.sigma * n / n_len * tmp

        print("surface", self.f[0])

        # compute gravity force
        for i in range(self.n_particles):
            self.f[i] += self.rho[i] * self.g
        
        print("gravity", self.f[0])

        
    @ti.func
    def update(self, dt:ti.f32):
        
        pad = 0.1
        for i in range(self.n_particles):
            self.v[i] += self.f[i] / self.rho[i] * dt
            self.x[i] += self.v[i] * dt
            for j in ti.static(range(self.dim)):
                if self.x[i][j] < 0 + pad or self.x[i][j] > 1 - pad:
                    self.v[i][j] *= -0.3
                    if self.x[i][j] < 0 + pad:
                        self.x[i][j] = pad
                    else:
                        self.x[i][j] = 1 - pad
                
        # print(self.n[0])
        # print(self.f[0], self.x[0], self.v[0])
        pass

    def get_positions(self):
        return self.x.to_numpy()
    
    def get_colors(self):
        return [0x068587, 0xED553B]
    
    def get_color_indices(self):
        return self.is_surface.to_numpy()