import taichi as ti

PI = 3.1415926

# http://www.glowinggoo.com/sph/bin/kelager.06.pdf

@ti.data_oriented
class W_poly6:

    @staticmethod
    @ti.func
    def W(r_vec, h):
        r2 = r_vec.dot(r_vec)
        r2 = ti.max(r2, 1e-10)
        k = 0.0
        if r2 <= h ** 2:
            k = ((h ** 2) - r2) ** 3
        return 315 / (64 * PI * (h ** 9)) * k

    @staticmethod
    @ti.func
    def W_grad(r_vec, h):
        r2 = r_vec.dot(r_vec)
        r2 = ti.max(r2, 1e-10)
        k = ti.Vector([0.0 for i in range(r_vec.n)])
        if r2 <= h ** 2:
            k = ((h ** 2) - r2) ** 2 * r_vec
        return -945 / (32 * PI * (h ** 9)) * k

    @staticmethod
    @ti.func
    def W_grad2(r_vec, h):
        r2 = r_vec.dot(r_vec)
        r2 = ti.max(r2, 1e-10)
        k = 0.0
        if r2 <= h ** 2:
            k = ((h ** 2) - r2) * (3 * h ** 2 - 7 * r2)
        return -945 / (32 * PI * (h ** 9)) * k

@ti.data_oriented
class W_spiky:
    # pressure
        
    @staticmethod
    @ti.func
    def W(r_vec, h):
        r = ti.sqrt(r_vec.dot(r_vec))
        r = ti.max(r, 1e-5)
        k = 0.0
        if r <= h:
            k = (h - r) ** 3
        return 15 / (PI * (h ** 6)) * k
    
    @staticmethod
    @ti.func
    def W_grad(r_vec, h):
        r = ti.sqrt(r_vec.dot(r_vec))
        r = ti.max(r, 1e-5)
        k = ti.Vector([0.0 for i in range(r_vec.n)])
        if r <= h:
            k = (h - r) ** 2 / r * r_vec
        return -45 / (PI * (h ** 6)) * k

    @staticmethod
    @ti.func
    def W_grad2(r_vec, h):
        r = ti.sqrt(r_vec.dot(r_vec))
        r = ti.max(r, 1e-5)
        k = 0.0
        if r <= h:
            k = (h - r) * (h - 2 * r) / r
        return -90 / (PI * (h ** 6)) * k


@ti.data_oriented
class W_viscosity:
    # viscosity

    @staticmethod
    @ti.func
    def W(r_vec, h):
        r = ti.sqrt(r_vec.dot(r_vec))
        r = ti.max(r, 1e-5)
        k = 0.0
        if r <= h:
            k = -(r ** 3) / (2 * (h ** 3)) + (r / h) ** 2 + (h / 2 / r) - 1
        return 15 / (2 * PI * (h ** 3)) * k

    @staticmethod
    @ti.func
    def W_grad(r_vec, h):
        r = ti.sqrt(r_vec.dot(r_vec))
        r = ti.max(r, 1e-5)
        k = ti.Vector([0.0] * r_vec.n)
        if r <= h:
            k = r_vec * (-3 * r / (2 * (h**3)) + 2/h/h - (h / (2 * r**3)))
        return 15 / (2 * PI * (h ** 3)) * k

    @staticmethod
    @ti.func
    def W_grad2(r_vec, h):
        r = ti.sqrt(r_vec.dot(r_vec))
        r = ti.max(r, 1e-5)
        k = 0.0
        if r <= h:
            k = h - r
        return 45 / (PI * (h ** 6)) * k



if __name__ == '__main__':
    ti.init(arch=ti.cpu, debug=True)

    @ti.kernel
    def test():
        r0 = ti.Vector([1e-2, 0])
        r1 = ti.Vector([0.2, 0.2])
        r2 = ti.Vector([0.5, 0])
        r3 = ti.Vector([1.0, 0])
        h = 1
        print(W_poly6.W(r0, h), W_poly6.W(r1, h), W_poly6.W(r2, h), W_poly6.W(r3, h))
        print(W_poly6.W_grad(r0, h), W_poly6.W_grad(r1, h), W_poly6.W_grad(r2, h), W_poly6.W_grad(r3, h))
        print(W_poly6.W_grad2(r0, h), W_poly6.W_grad2(r1, h), W_poly6.W_grad2(r2, h), W_poly6.W_grad2(r3, h))

        print(W_spiky.W(r0, h), W_spiky.W(r1, h), W_spiky.W(r2, h), W_spiky.W(r3, h))
        print(W_spiky.W_grad(r0, h), W_spiky.W_grad(r1, h), W_spiky.W_grad(r2, h), W_spiky.W_grad(r3, h))
        print(W_spiky.W_grad2(r0, h), W_spiky.W_grad2(r1, h), W_spiky.W_grad2(r2, h), W_spiky.W_grad2(r3, h))

        print(W_viscosity.W(r0, h), W_viscosity.W(r1, h), W_viscosity.W(r2, h), W_viscosity.W(r3, h))
        print(W_viscosity.W_grad(r0, h), W_viscosity.W_grad(r1, h), W_viscosity.W_grad(r2, h), W_viscosity.W_grad(r3, h))
        print(W_viscosity.W_grad2(r0, h), W_viscosity.W_grad2(r1, h), W_viscosity.W_grad2(r2, h), W_viscosity.W_grad2(r3, h))


    test()

@ti.data_oriented
class W_cubic:
    # cubic

    @staticmethod
    @ti.func
    def W(r_vec, r, h):
        k = 10. / (7. * PI * h**r_vec.n)
        q = r / h
        res = 0.0
        if q <= 1.0:
            res = k * (1 - 1.5 * q**2 + 0.75 * q**3)
        elif q < 2.0:
            res = k * 0.25 * (2 - q)**3
        return res

    @staticmethod
    @ti.func
    def W_grad(r_vec, r, h):
        k = 10. / (7. * PI * h**r_vec.n)
        q = r / h
        res = 0.0
        if q < 1.0:
            res = (k / h) * (-3 * q + 2.25 * q**2)
        elif q < 2.0:
            res = -0.75 * (k / h) * (2 - q)**2
        return res
