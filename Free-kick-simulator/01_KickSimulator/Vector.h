#pragma once
#include <cmath>
#include <iostream>

class Vector {
public:
    double x, y, z;

    Vector() : x(0), y(0), z(0) {}
    Vector(double x, double y, double z) : x(x), y(y), z(z) {}

    Vector operator+(const Vector& v) const {
        return Vector(x + v.x, y + v.y, z + v.z);
    }
    Vector operator-(const Vector& v) const {
        return Vector(x - v.x, y - v.y, z - v.z);
    }
    Vector operator*(double s) const {
        return Vector(x * s, y * s, z * s);
    }
    friend Vector operator*(double s, const Vector& v) {
        return Vector(v.x * s, v.y * s, v.z * s);
    }
    Vector operator-() const {
        return Vector(-x, -y, -z);
    }
    friend std::ostream& operator<<(std::ostream& os, const Vector& v) {
        os << "x: " << v.x << ", y: " << v.y << ", z: " << v.z;
        return os;
    }
    double dot(const Vector& v) const {
        return x * v.x + y * v.y + z * v.z;
    }
    Vector cross(const Vector& v) const {
        return Vector(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        );
    }
    void normalize() {
        double len = std::sqrt(x * x + y * y + z * z);
        if (len > 0) {
            x /= len;
            y /= len;
            z /= len;
        }
    }
    double norm() const {
        return std::sqrt(x * x + y * y + z * z);
    }
};
