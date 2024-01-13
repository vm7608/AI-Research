"""Vector.

This is a simple app to demonstrate the use of the Vector class.
"""

from vector import Vector

if __name__ == "__main__":
    v1 = Vector(2, 1)
    v2 = Vector(-2, 4)
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(v1.values == [2, 1])

    v = Vector(1, 2, 3)
    v[0] = 4
    print(v.values == [4, 2, 3])
    print(v1 + v2 == Vector(0, 5))

    print(v1 * 2 == Vector(4, 2))

    b1 = Vector(2, 0)
    b2 = Vector(0, 2)

    r = Vector(3, 4)

    print(r.change_basis(b1, b2))
