#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <random>
#include <stack>
#include "omp.h"
#include <iomanip>
#include <immintrin.h>
#include <fstream>
#include <algorithm>
#include "../01_KickSimulator/Vector.h"
#include "../01_KickSimulator/generate_free_kick_waypoints.h"

using namespace std;

#define EPS 1e-4
int MAX_DEPTH = 5;

enum ANTI_ALIASING {
    AA_NONE,
    AA_RANDOM_2,
    AA_REGULAR_2,
    AA_RANDOM_4,
    AA_REGULAR_4
};

class RGB {
public:
    int r, g, b;

    RGB() : r(0), g(0), b(0) {}
    RGB(int _r, int _g, int _b) : r(_r), g(_g), b(_b) {}

    // Addition
    RGB operator+(const RGB& other) const {
        return RGB(r + other.r, g + other.g, b + other.b);
    }

    RGB& operator+=(const RGB& other) {
        r += other.r; g += other.g; b += other.b;
        return *this;
    }

    // Subtraction
    RGB operator-(const RGB& other) const {
        return RGB(r - other.r, g - other.g, b - other.b);
    }

    RGB& operator-=(const RGB& other) {
        r -= other.r; g -= other.g; b -= other.b;
        return *this;
    }

    // Component-wise multiplication (RGB * RGB)
    RGB operator*(const RGB& other) const {
        return RGB(r * other.r, g * other.g, b * other.b);
    }

    RGB& operator*=(const RGB& other) {
        r *= other.r; g *= other.g; b *= other.b;
        return *this;
    }

    // Scalar multiplication
    RGB operator*(double scalar) const {
        return RGB(r * scalar, g * scalar, b * scalar);
    }

    RGB& operator*=(double scalar) {
        r *= scalar; g *= scalar; b *= scalar;
        return *this;
    }

    // Scalar division
    RGB operator/(double scalar) const {
        return (scalar != 0) ? RGB(r / scalar, g / scalar, b / scalar) : RGB();
    }

    RGB& operator/=(double scalar) {
        if (scalar != 0) {
            r /= scalar; g /= scalar; b /= scalar;
        }
        return *this;
    }

    // Equality and Inequality
    bool operator==(const RGB& other) const {
        return (r == other.r && g == other.g && b == other.b);
    }

    bool operator!=(const RGB& other) const {
        return !(*this == other);
    }

    void clamp() {
        r = min(255, max(0, r));
        g = min(255, max(0, g));
        b = min(255, max(0, b));
    }

    // Output stream overload
    friend std::ostream& operator<<(std::ostream& os, const RGB& color) {
        os << "RGB(" << color.r << ", " << color.g << ", " << color.b << ")";
        return os;
    }
};

class LightSource {
public:
    Vector position;
    RGB intensity;

    LightSource(Vector _position, RGB _intensity) : position(_position), intensity(_intensity) {}
};

class Ray {
public:
    Vector origin;
    Vector direction;
    Ray(Vector _o, Vector _d) : origin(_o), direction(_d) {}
};

// Rodrigues' rotation formula function
Vector rotate(Vector p, Vector axis, double angle) {
    axis.normalize();
    return p * cos(angle) + axis.cross(p) * sin(angle) + 
           axis * (axis.dot(p)) * (1 - cos(angle));
}

class Pixel {
public:
    int x, y;
    RGB color;

    Pixel(int _x, int _y, RGB _color = RGB()) : x(_x), y(_y), color(_color) {}
};

class Material {
public:
    virtual RGB getObjectColor() const = 0;
    virtual double getAmbientCoefficient() const = 0;
    virtual double getDiffusionCoefficient() const = 0;
    virtual double getSpecularCoefficient() const = 0;
    virtual double getShininessCoefficient() const = 0;
    virtual double getRefractionIndex() const = 0;
    virtual ~Material() {}
};

class Shape {
public:
    virtual Vector getSurfaceNormal(Vector& pointOnSurface) = 0;
    virtual double solveRoot(Ray ray) = 0;
    virtual Material* getMaterial() const = 0;
    virtual ~Shape() {}
};

class Sphere : public Shape {
public:
    Vector center;
    double R;
    Material* material;

    Sphere(Vector _center, double _R, Material* _material) : center(_center), R(_R), material(_material) {}

    Material* getMaterial() const override { return material; }

    Vector getSurfaceNormal(Vector& pointOnSurface) override {
        return pointOnSurface - center;
    }

    double solveRoot(Ray ray) override {
        Vector oc = ray.origin - center;
        double a = ray.direction.dot(ray.direction);
        double b = 2.0 * oc.dot(ray.direction);
        double c = oc.dot(oc) - R * R;
        double discriminant = b * b - 4 * a * c;

        if (discriminant < 0) {
            return -std::numeric_limits<double>::min();  // No intersection
        }

        double inv2a = 0.5 / a;  // Precompute reciprocal to avoid division
        double sqrtD = sqrt(discriminant);
        double t1 = (-b - sqrtD) * inv2a;
        double t2 = (-b + sqrtD) * inv2a;

        
        if (t1 > 0) {
            if (t2 > 0) return (t1 < t2) ? t1 : t2;
            return t1;
        } 
        return (t2 > 0) ? t2 : -std::numeric_limits<double>::min();        
    }
};

// Define the quartic function and its derivative
double quarticFunction(double t, double a, double b, double c, double d, double e) {
    return a * t * t * t * t + b * t * t * t + c * t * t + d * t + e;
}

double quarticDerivative(double t, double a, double b, double c, double d) {
    return 4 * a * t * t * t + 3 * b * t * t + 2 * c * t + d;
}

// Newton's method for finding real roots of the quartic function
double newtonSolveQuartic(double a, double b, double c, double d, double e, double initialGuess, int maxIterations = 50, double tolerance = 1e-6) {
    double t = initialGuess;
    
    for (int i = 0; i < maxIterations; i++) {
        double f_t = quarticFunction(t, a, b, c, d, e);
        double f_prime_t = quarticDerivative(t, a, b, c, d);

        if (std::abs(f_prime_t) < tolerance) {
            break; // Avoid division by zero
        }

        double t_next = t - f_t / f_prime_t;
        
        if (std::abs(t_next - t) < tolerance) {
            return t_next; // Converged to a root
        }
        
        t = t_next;
    }

    return std::numeric_limits<double>::max(); // If no root was found
}

class Torus : public Shape {
public:
    Vector center;
    double R; // Major radius
    double r; // Minor radius
    Material* material;

    Torus(Vector _center, double _R, double _r, Material* _material) : center(_center), R(_R), r(_r), material(_material) {}

    Material* getMaterial() const override { return material; }

    Vector getSurfaceNormal(Vector& pointOnSurface) override {
        Vector local = pointOnSurface - center;
        double phi = atan2(local.y, local.x);
        Vector closestPoint = Vector(R * cos(phi), R * sin(phi), 0);
        Vector normal = (pointOnSurface - (center + closestPoint));
        normal.normalize();
        return normal;
    }

    double solveRoot(Ray ray) override {
        // Transform ray into torus coordinate system
        Vector o = ray.origin - center;
        Vector d = ray.direction;

        double dDotD = d.dot(d);
        double oDotD = o.dot(d);
        double oDotO = o.dot(o);
        double sumR = R * R + r * r;

        // Quartic equation coefficients
        double a = dDotD * dDotD;
        double b = 4.0 * dDotD * oDotD;
        double c = 4.0 * oDotD * oDotD + 2.0 * dDotD * (oDotO - sumR) + 4.0 * R * R * d.z * d.z;
        double d_coeff = 4.0 * (oDotO - sumR) * oDotD + 8.0 * R * R * o.z * d.z;
        double e = (oDotO - sumR) * (oDotO - sumR) - 4.0 * R * R * (r * r - o.z * o.z);

        // Use Newton’s method to solve
        double initialGuess = 1.0;
        double root = newtonSolveQuartic(a, b, c, d_coeff, e, initialGuess);

        if (root == std::numeric_limits<double>::max() || root < 0) {
            return -std::numeric_limits<double>::min();
        }

        return root;
    }
};

class Plane : public Shape {
public:
    Vector point;    // A point on the plane (e.g., the origin point on the floor)
    Vector normal;   // The normal vector of the plane
    Material* material; // Material for the plane

    Plane(Vector _point, Vector _normal, Material* _material) : point(_point), normal(_normal), material(_material) {}

    Material* getMaterial() const override { return material; }

    Vector getSurfaceNormal(Vector& pointOnSurface) override {
        return normal; // The normal is constant for a plane
    }

    double solveRoot(Ray ray) override {
        double denominator = 1/ray.direction.dot(normal);
        
        if (fabs(denominator) < EPS) {
            return -std::numeric_limits<double>::max();  // No intersection, ray is parallel to the plane
        }
        
        double t = (point - ray.origin).dot(normal) * denominator;
        
        if (t > EPS) {
            return t;
        }
        
        return -std::numeric_limits<double>::max();  // No valid intersection
    }
};

class KleinBottle : public Shape {
public:
    Vector center;
    double R; // Major radius
    double r; // Tube radius
    Material* material;

    KleinBottle(Vector _center, double _R, double _r, Material* _material)
        : center(_center), R(_R), r(_r), material(_material) {}

    Material* getMaterial() const override { return material; }

    Vector parametric(double u, double v) const {
        double x = (R + cos(u / 2) * sin(v) - sin(u / 2) * sin(2 * v)) * cos(u);
        double y = (R + cos(u / 2) * sin(v) - sin(u / 2) * sin(2 * v)) * sin(u);
        double z = sin(u / 2) * sin(v) + cos(u / 2) * sin(2 * v);
        return center + Vector(x, y, z);
    }

    Vector getSurfaceNormal(Vector& pointOnSurface) override {
        double u = atan2(pointOnSurface.y - center.y, pointOnSurface.x - center.x);
        double v = atan2(pointOnSurface.z - center.z, sqrt(pow(pointOnSurface.x - center.x, 2) + pow(pointOnSurface.y - center.y, 2)));

        double epsilon = 1e-8;
        Vector du = (parametric(u + epsilon, v) - parametric(u - epsilon, v)) * (0.5 / epsilon);
        Vector dv = (parametric(u, v + epsilon) - parametric(u, v - epsilon)) * (0.5 / epsilon);
        Vector normal = du.cross(dv);
        normal.normalize();

        return normal;
    }

    double solveRoot(Ray ray) override {
        const int maxIterations = 100;
        const double tolerance = 1e-4;
        double t = 1.0;
    
        for (int i = 0; i < maxIterations; i++) {
            Vector p = ray.origin + ray.direction * t;
            double u = atan2(p.y - center.y, p.x - center.x);
            double v = atan2(p.z - center.z, sqrt(pow(p.x - center.x, 2) + pow(p.y - center.y, 2)));
            Vector q = parametric(u, v);
            Vector normal = getSurfaceNormal(q);
    
            double error = (q - p).norm();
            if (error < tolerance) {
                return t;
            }
    
            // Compute ray-plane intersection step
            double denom = normal.dot(ray.direction);
            if (fabs(denom) < 1e-6) { // Avoid division by zero (ray nearly parallel to surface)
                return -std::numeric_limits<double>::min();
            }
    
            double step = normal.dot(q - p) / denom;
            t += step;
    
            if (t < 0) return -std::numeric_limits<double>::min();
        }
        if (t < 1) return t;
        return -std::numeric_limits<double>::min();
    }    
};

class Triangle : public Shape {
public:
    Vector v1;
    Vector v2;
    Vector v3;
    Vector normal;   // The normal vector of the plane
    Vector n1;
    Vector n2;
    Vector n3;
    Material* material; // Material for the plane

    Triangle(Vector _v1, Vector _v2, Vector _v3, Material* _material) : v1(_v1), v2(_v2), v3(_v3), material(_material) {
        normal = (v2 - v1).cross(v3 - v1);
        normal.normalize();
    }

    Material* getMaterial() const override { return material; }

    Vector getSurfaceNormal(Vector& pointOnSurface) override {
        // Compute barycentric coordinates
        Vector u = v2 - v1;
        Vector v = v3 - v1;
        Vector w = pointOnSurface - v1;
    
        double d00 = u.dot(u);
        double d01 = u.dot(v);
        double d11 = v.dot(v);
        double d20 = w.dot(u);
        double d21 = w.dot(v);
    
        double denom = d00 * d11 - d01 * d01;
        if (fabs(denom) < EPS) return normal;
    
        double beta = (d11 * d20 - d01 * d21) / denom;
        double gamma = (d00 * d21 - d01 * d20) / denom;
        double alpha = 1.0 - beta - gamma;

        if (alpha < 0 || beta < 0 || gamma < 0) {
            return normal;  // fallback to face normal
        }
    
        Vector interpolatedNormal = n1 * alpha + n2 * beta + n3 * gamma;
        interpolatedNormal.normalize();
        return interpolatedNormal;
    }

    __attribute__((hot)) double solveRoot(Ray ray) override {
        double denominator = ray.direction.dot(normal);
        if (fabs(denominator) < EPS) {
            return -std::numeric_limits<double>::max();
        }

        denominator = 1/denominator;
    
        double t = (v1 - ray.origin).dot(normal) * denominator;
        if (t <= EPS) {
            return -std::numeric_limits<double>::max();
        }
    
        Vector Rp = ray.origin + ray.direction * t;
    
        Vector edge1 = v2 - v1;
        Vector edge2 = v3 - v2;
        Vector edge3 = v1 - v3;
    
        Vector C1 = Rp - v1;
        Vector C2 = Rp - v2;
        Vector C3 = Rp - v3;
    
        double cross1 = edge1.cross(C1).dot(normal);
        if (cross1 < -EPS) return -std::numeric_limits<double>::max();
    
        double cross2 = edge2.cross(C2).dot(normal);
        if (cross2 < -EPS) return -std::numeric_limits<double>::max();
    
        double cross3 = edge3.cross(C3).dot(normal);
        if (cross3 < -EPS) return -std::numeric_limits<double>::max();
    
        return t;
    }
};

class Screen {
public:
    Vector normal;
    Vector right;
    Vector up;
    int width;
    int height;
    double pov;
    std::vector<Pixel> pixels;

    Screen(Vector _n, int _w, int _h) : normal(_n), width(_w), height(_h) {
        pixels.reserve(width * height);

        pov = normal.norm();

        // Initialize pixels with default values
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                pixels.emplace_back(x, y, RGB(0, 0, 0));
            }
        }

        // Step 1: Define Simple Vectors (Before Rotation)
        Vector simple_normal(1, 0, 0);
        Vector simple_right(0, 1, 0);
        Vector simple_up(0, 0, 1);

        // Step 2: Compute Rotation Axis and Angle
        Vector rotation_axis = simple_normal.cross(normal);
        double rotation_angle = acos(simple_normal.dot(normal) / (simple_normal.norm() * normal.norm()));

        // Step 3: Rotate Vectors
        right = rotate(simple_right, rotation_axis, rotation_angle);
        up = rotate(simple_up, rotation_axis, rotation_angle);
        normal = rotate(simple_normal, rotation_axis, rotation_angle);
    }

    std::vector<Pixel>::iterator begin() { return pixels.begin(); }
    std::vector<Pixel>::iterator end() { return pixels.end(); }

    void writeToJPG(const std::string& filename, int quality = 90) {
        std::vector<unsigned char> imageData(width * height * 3); // RGB format

        // Convert grayscale pixel data to RGB
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = (y*width + x) * 3;  // RGB index
                RGB pixelColor = pixels[y*width + x].color;
                unsigned char red = static_cast<unsigned char>(pixelColor.r);
                unsigned char green = static_cast<unsigned char>(pixelColor.g);
                unsigned char blue = static_cast<unsigned char>(pixelColor.b);

                // Assign to all RGB channels
                imageData[index] = red;     // R
                imageData[index + 1] = green; // G
                imageData[index + 2] = blue; // B
            }
        }

        // Write the image to a JPG file
        stbi_write_jpg(filename.c_str(), width, height, 3, imageData.data(), quality);
    }
};


class Metallic : public Material {
public:
    // Members
    RGB color;
    double mA = 0.0005;
    double mD = 0.005;
    double mS = 1.0;
    double mSp = 50;
    double eta = 999999;

    // Constructor
    Metallic(RGB _color) : color(_color) {}

    // Getters
    RGB getObjectColor() const override { return color; }
    double getAmbientCoefficient() const override { return mA; }
    double getDiffusionCoefficient() const override { return mD; }
    double getSpecularCoefficient() const override { return mS; }
    double getShininessCoefficient() const override { return mSp; }
    double getRefractionIndex() const override { return eta; }
};

class Glassy : public Material {
public:
    // Members
    RGB color;
    double mA = 0.00001;
    double mD = 0.00001;
    double mS = 0.5;
    double mSp = 300;
    double eta = 1.5;

    // Constructor
    Glassy(RGB _color) : color(_color) {}

    // Getters
    RGB getObjectColor() const override { return color; }
    double getAmbientCoefficient() const override { return mA; }
    double getDiffusionCoefficient() const override { return mD; }
    double getSpecularCoefficient() const override { return mS; }
    double getShininessCoefficient() const override { return mSp; }
    double getRefractionIndex() const override { return eta; }
};

class CheckerboardMaterial : public Material {
public:
    RGB color1, color2;  // Two colors for the checkerboard pattern
    double squareSize;    // Size of each square on the checkerboard

    CheckerboardMaterial(RGB _color1, RGB _color2, double _squareSize)
        : color1(_color1), color2(_color2), squareSize(_squareSize) {}

    RGB getObjectColor() const override { return RGB(255, 255, 255); } // Default to white for shading

    double getAmbientCoefficient() const override { return 1.0; }
    double getDiffusionCoefficient() const override { return 1.0; }
    double getSpecularCoefficient() const override { return 0.0; }
    double getShininessCoefficient() const override { return 0.0; }
    double getRefractionIndex() const override { return 1.0; }

    // Determine the color at a given point on the checkerboard
    RGB getColorAtPoint(const Vector& point) const {
        int xIndex = abs(static_cast<int>(floor(point.x / squareSize))) % 2;
        int yIndex = abs(static_cast<int>(floor(point.y / squareSize))) % 2;
        // Alternate colors based on the coordinates
        return (xIndex == yIndex) ? color1 : color2;
    }
};

enum ReflectionMethod {
    FRESNEL,
    SCHLICK
};

RGB traceRay(Ray& ray, Screen& screen, vector<Shape*>& objects, int depth, LightSource& light, RGB& ambientLight, ReflectionMethod reflMethod) {
    if (depth > MAX_DEPTH) return RGB(0, 0, 0); // Prevent infinite recursion

    // Find the closest object the ray hits
    Shape* closestObject = nullptr;
    double minT = std::numeric_limits<double>::max();
    Material* hitMaterial = nullptr;
    Vector hitPoint;

    for (Shape* obj : objects) {
        double t = obj->solveRoot(ray);
        if (t > 0 && t < minT) {
            minT = t;
            closestObject = obj;
            hitMaterial = closestObject->getMaterial();
            hitPoint = ray.origin + (ray.direction * minT);
        }
    } 

    if (!closestObject) return RGB(0, 0, 0); // No intersection, return background

    if (hitMaterial) {
        if (dynamic_cast<CheckerboardMaterial*>(hitMaterial)) {
            // Apply checkerboard pattern
            CheckerboardMaterial* checkerboard = dynamic_cast<CheckerboardMaterial*>(hitMaterial);
            return checkerboard->getColorAtPoint(hitPoint);
        }
    }

    // Compute intersection point and normal
    Vector normal = closestObject->getSurfaceNormal(hitPoint);
    normal.normalize();

    // Get material properties
    RGB objectColor = hitMaterial->getObjectColor();
    double mA = hitMaterial->getAmbientCoefficient();
    double mD = hitMaterial->getDiffusionCoefficient();
    double mS = hitMaterial->getSpecularCoefficient();
    double mSp = hitMaterial->getShininessCoefficient();

    // Light vector
    Vector L = (light.position - hitPoint);
    L.normalize();

    // View vector
    Vector V = (ray.origin - hitPoint);
    V.normalize();

    // Reflection vector
    Vector R = (normal * (2 * normal.dot(L))) - L;
    R.normalize();

    // **Phong Lighting Model**
    RGB ambient = objectColor * ambientLight * mA;
    RGB diffuse = objectColor * light.intensity * mD * std::max(L.dot(normal), 0.0);
    RGB specular = RGB(1, 1, 1) * light.intensity * mS * pow(std::max(V.dot(R), 0.0), mSp);

    RGB phongColor = ambient + diffuse + specular;

    // **Shadow Check**
    Vector shadowOrigin = hitPoint + normal * EPS; // Prevent self-intersection
    Vector shadowDir = (light.position - shadowOrigin);
    shadowDir.normalize();
    Ray shadowRay(shadowOrigin, shadowDir);

    double t = closestObject->solveRoot(shadowRay);
    if (t > EPS && t < (light.position - shadowOrigin).norm()) {  
        phongColor = ambient;  // Only ambient light if blocked
    }

    // **Reflection & Refraction**
    double n1 = 1.0;   // Air
    double n2 = hitMaterial->getRefractionIndex();  // Glass, water, etc.

    RGB reflectionColor(0, 0, 0), refractionColor(0, 0, 0);

    if (n2 < 10) {
        bool inside = (ray.direction.dot(normal) > 0);
        if (inside) {
            std::swap(n1, n2);
            normal = -normal;
        }

        double eta = n1 / n2;
        double cosTheta1 = -normal.dot(ray.direction);
        if (inside) {
            normal = -normal;  // Flip normal when inside
            cosTheta1 = -cosTheta1;  // Ensure proper direction
        }
        double sin2Theta2 = eta * eta * (1 - cosTheta1 * cosTheta1);
        
        // **Total Internal Reflection (TIR)**
        if (sin2Theta2 > 1.0) {
            // Total Internal Reflection
            Vector reflectionDir = ray.direction - normal * (2 * ray.direction.dot(normal));
            reflectionDir.normalize();
            Ray reflectedRay(hitPoint + normal * EPS, reflectionDir);
            reflectionColor = traceRay(reflectedRay, screen, objects, depth + 1, light, ambientLight, reflMethod);
        } 
        else {
            // Refraction Calculation (Snell's Law)
            double cosTheta2 = sqrt(1 - sin2Theta2);
            Vector refractionDir = (eta * ray.direction) + normal * (eta * cosTheta1 - sqrt(fabs(1.0 - sin2Theta2)));
            refractionDir.normalize();
            Ray refractedRay(hitPoint - normal * EPS, refractionDir);
            refractionColor = traceRay(refractedRay, screen, objects, depth + 1, light, ambientLight, reflMethod);
        
            // Reflection Calculation
            Vector reflectionDir = ray.direction - normal * (2 * ray.direction.dot(normal));
            reflectionDir.normalize();
            Ray reflectedRay(hitPoint + normal * EPS, reflectionDir);
            reflectionColor = traceRay(reflectedRay, screen, objects, depth + 1, light, ambientLight, reflMethod);
        
            // Fresnel Reflectance
            double R = 1.0;
            double T = 0.0;
            if (reflMethod == FRESNEL) {
                double RsBase = (n1 * cosTheta1 - n2 * cosTheta2) / (n1 * cosTheta1 + n2 * cosTheta2);
                double Rs = RsBase * RsBase;
                double RpBase = (n1 * cosTheta2 - n2 * cosTheta1) / (n1 * cosTheta2 + n2 * cosTheta1);
                double Rp = RpBase * RpBase;
                R = Rs + Rp;
                T = 1.0 - R;
            }
            // Fresnel Reflectance approximated by Schlick's Method
            else if (reflMethod == SCHLICK) {
                double R0 = (n1 - n2) / (n1 + n2);
                R0 = R0 * R0;
                R = R0 + (1.0 - R0) * pow(1.0 - cosTheta1, 5);
                T = 1.0 - R;
            }
        
            RGB finalColor = phongColor + (reflectionColor * R) + (refractionColor * T);
            finalColor.clamp();
            return finalColor;
        }
    }

    RGB tmpColor = phongColor + reflectionColor;
    tmpColor.clamp();
    return tmpColor;
}

void processScreen(Screen& screen, Vector& origin, vector<Shape*>& objects, LightSource& light, RGB& ambientLight, ANTI_ALIASING aa, ReflectionMethod reflMethod) {
    #pragma omp parallel for
    for (auto& pixel : screen) {
        double aspectRatio = (double)screen.width / screen.height;
        stack<Ray> st;

        auto computePixelPosition = [&](double px, double py) -> Vector {
            Vector screenCenter = origin + screen.normal * screen.pov;
            Vector pixelPosition = screenCenter 
                + screen.right * ((px - screen.width / 2.0) / screen.width * aspectRatio) 
                + screen.up * ((screen.height / 2.0 - py) / screen.height); //Inverted y direction.
            return pixelPosition;
        };
        
        auto shootRay = [&](double px, double py) {
            Vector pixelPos = computePixelPosition(px, py);
            Ray ray(origin, pixelPos - origin);
            ray.direction.normalize();
            st.push(ray);
        };

        switch (aa) {
            case AA_NONE: {
                shootRay(pixel.x, pixel.y);
                break;
            }
        
            case AA_RANDOM_2: {
                for (int i = 0; i < 2; i++) {
                    double randX = pixel.x + (rand() / (double)RAND_MAX - 0.5);
                    double randY = pixel.y + (rand() / (double)RAND_MAX - 0.5);
                    shootRay(randX, randY);
                }
                break;
            }
        
            case AA_REGULAR_2: {
                for (int i = 0; i < 2; i++) {
                    double offsetX = (i * 0.5) - 0.25;
                    shootRay(pixel.x + offsetX, pixel.y);
                }
                break;
            }
        
            case AA_RANDOM_4: {
                for (int i = 0; i < 4; i++) {
                    double randX = pixel.x + (rand() / (double)RAND_MAX - 0.5);
                    double randY = pixel.y + (rand() / (double)RAND_MAX - 0.5);
                    shootRay(randX, randY);
                }
                break;
            }
        
            case AA_REGULAR_4: {
                for (int i = 0; i < 2; i++) {
                    for (int j = 0; j < 2; j++) {
                        double offsetX = (i * 0.5) - 0.25;
                        double offsetY = (j * 0.5) - 0.25;
                        shootRay(pixel.x + offsetX, pixel.y + offsetY);
                    }
                }
                break;
            }
        }

        // Loop through normal rays
        int rayCount = st.size();
        RGB accumulatedColor(0, 0, 0);

        // Process each ray
        while (!st.empty()) {
            Ray ray = st.top();
            st.pop();

            RGB color = traceRay(ray, screen, objects, 0, light, ambientLight, reflMethod);
            accumulatedColor += color;
        }

        // Average the accumulated color over the samples
        pixel.color = accumulatedColor / rayCount;
        pixel.color.clamp(); // Ensure RGB values stay within valid range
    }
}  

vector<Shape*> loadOBJ(const string& filename, Material* material) {
    vector<Vector> vertices;
    vector<Vector> vertexNormals; // Accumulated vertex normals
    vector<tuple<int, int, int>> faces;
    vector<Shape*> triangles;

    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open OBJ file: " << filename << endl;
        return triangles;
    }

    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        string prefix;
        iss >> prefix;

        if (prefix == "v") {
            double x, y, z;
            iss >> x >> y >> z;
            vertices.emplace_back(x, y, z);
            vertexNormals.emplace_back(0, 0, 0); // Initialize with zero vector
        } else if (prefix == "f") {
            int v1, v2, v3;
            iss >> v1 >> v2 >> v3;
            faces.emplace_back(v1 - 1, v2 - 1, v3 - 1); // Store as 0-based index
        }
    }

    // Accumulate normals
    for (const auto& [i1, i2, i3] : faces) {
        Vector& p1 = vertices[i1];
        Vector& p2 = vertices[i2];
        Vector& p3 = vertices[i3];

        Vector faceNormal = (p2 - p1).cross(p3 - p1);

        vertexNormals[i1] = vertexNormals[i1] + faceNormal;
        vertexNormals[i2] = vertexNormals[i2] + faceNormal;
        vertexNormals[i3] = vertexNormals[i3] + faceNormal;
    }

    // Normalize vertex normals
    for (auto& n : vertexNormals) {
        n.normalize();
    }

    // Create triangles with vertex normals
    for (const auto& [i1, i2, i3] : faces) {
        Triangle* tri = new Triangle(vertices[i1], vertices[i2], vertices[i3], material);
        tri->n1 = vertexNormals[i1];
        tri->n2 = vertexNormals[i2];
        tri->n3 = vertexNormals[i3];
        tri->normal = (tri->v2 - tri->v1).cross(tri->v3 - tri->v1);
        tri->normal.normalize();
        triangles.push_back(tri);
    }

    file.close();
    return triangles;
}

void renderMovingSphere(int count, Vector camPos, Vector screenNormal, vector<Shape*>& objects, Material* sphereMat, LightSource& light, RGB ambientLight, ANTI_ALIASING aa, ReflectionMethod reflMethod, double kick_start_z) {
    auto waypoints = generate_free_kick_waypoints(count, kick_start_z); // Pass kick_start_z
    Sphere* movingSphere = new Sphere(Vector(0,0,0), 0.11, sphereMat); // Changed radius from 1.0 to 0.11
    objects.push_back(movingSphere);
    int numFrames = waypoints.size();
    for (int i = 0; i < numFrames; ++i) {
        // Apply coordinate mapping: waypoint (depth, height, width) to world (depth, width, height)
        movingSphere->center.x = waypoints[i].x; // X (depth) maps directly
        movingSphere->center.y = waypoints[i].z; // Y (world width) gets Z from waypoint (width)
        movingSphere->center.z = waypoints[i].y; // Z (world height) gets Y from waypoint (height)
        
        Screen screen(screenNormal, 2560, 1440);
        screen.pov = 0.9; // Decrease pov for a wider field of view
        processScreen(screen, camPos, objects, light, ambientLight, aa, reflMethod);
        std::ostringstream filenameStream;
        filenameStream << "output/frame_" << std::setw(4) << std::setfill('0') << i << ".jpg";
        screen.writeToJPG(filenameStream.str());
    }
    // Remove the movingSphere from the objects vector before deleting it
    objects.erase(std::remove(objects.begin(), objects.end(), movingSphere), objects.end());
    delete movingSphere;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Not enough parameters." << endl;
        exit(-1);
    }
    ReflectionMethod reflMethod = (ReflectionMethod)atoi(argv[1]);
    MAX_DEPTH = (int)atoi(argv[2]);
    
    Vector origin(0.0, 0.0, 0.0);
    double radius = 3.0;  
    double glassRadius = radius;  
    double smallSphereRadius = 0.25 * 2;  
    Vector glassCenter(10, 0, 0);  // Large sphere's position (Y stays constant)

    // Materials
    Glassy glassy(RGB(255, 255, 255));
    Metallic redMetal(RGB(255, 0, 0));
    Metallic blueMetal(RGB(0, 0, 255));
    Metallic greenMetal(RGB(0, 255, 0));
    Metallic yellowMetal(RGB(255, 255, 0));
    Metallic whiteMetal(RGB(255, 255, 255));
    CheckerboardMaterial checkerboard(RGB(34, 139, 34), RGB(0, 100, 0), radius); // Grassy green

    // Shapes
    // std::vector<Sphere*> spheres; // Commented out as it's related to orbiting spheres
    std::vector<Shape*> objects; // Only use spheres and/or plane

    // cout << "Loaded teapot: " << objects.size() << " triangles have been loaded"  << endl;

    {
        // Rotate left a bit
        Vector rotationAxis(0, 1, 0); 
        double angle = M_PI / 2;
        for (Shape* shape : objects) {
            Triangle* tri = dynamic_cast<Triangle*>(shape);
            if (tri) {
                // Rotate
                tri->v1 = rotate(tri->v1, rotationAxis, angle);
                tri->v2 = rotate(tri->v2, rotationAxis, angle);
                tri->v3 = rotate(tri->v3, rotationAxis, angle);

                // Recompute normal
                tri->normal = (tri->v2 - tri->v1).cross(tri->v3 - tri->v1);
                tri->normal.normalize();
            }
        }
    }
    {
        // Rotate up and Translate teapot into view
        Vector teapotOffset(10, 0, -1);  // In front of the camera, above the floor
        Vector rotationAxis(1, 0, 0); 
        double angle = M_PI/2;
        for (Shape* shape : objects) {
            Triangle* tri = dynamic_cast<Triangle*>(shape);
            if (tri) {
                // Rotate
                tri->v1 = rotate(tri->v1, rotationAxis, angle);
                tri->v2 = rotate(tri->v2, rotationAxis, angle);
                tri->v3 = rotate(tri->v3, rotationAxis, angle);

                // Translate
                tri->v1 = tri->v1 + teapotOffset;
                tri->v2 = tri->v2 + teapotOffset;
                tri->v3 = tri->v3 + teapotOffset;

                // Recompute normal
                tri->normal = (tri->v2 - tri->v1).cross(tri->v3 - tri->v1);
                tri->normal.normalize();
            }
        }
    }

    /*
    // **Small spheres orbiting in a perfect horizontal circle**
    double orbitRadius = radius + smallSphereRadius + 0.5;  // Distance from the big sphere
    Sphere* redSphere = new Sphere(glassCenter + Vector(orbitRadius, 0, 0), smallSphereRadius, &redMetal);
    Sphere* blueSphere = new Sphere(glassCenter + Vector(-orbitRadius, 0, 0), smallSphereRadius, &blueMetal);
    Sphere* yellowSphere = new Sphere(glassCenter + Vector(0, 0, -orbitRadius), smallSphereRadius, &yellowMetal);

    spheres.push_back(redSphere);
    spheres.push_back(blueSphere);
    spheres.push_back(yellowSphere);

    for (Sphere* sphere : spheres) {
        objects.push_back(sphere);
    }
    */

    // **Ground Plane**
    Plane floor(Vector(0, 0, -1), Vector(0, 0, 1), &checkerboard);
    objects.push_back(&floor);

    // **Light source**
    LightSource light(Vector(5, 5, 0), RGB(255, 255, 255));
    RGB ambientLight(255, 255, 255); // Increased from (255, 255, 225)

    /*
    // **Horizontal Orbital Motion Setup**
    int numFrames = 1;  // Number of frames for one full orbit
    double angleStep = 2 * M_PI / numFrames;  // Angle increment per frame

    for (int i = 0; i < numFrames; i++) {
        double angle = i * angleStep;

        // **Move small spheres only in the X-Z plane, keeping Y fixed**
        redSphere->center = glassCenter + Vector(orbitRadius * cos(angle), orbitRadius * sin(angle), 0);
        blueSphere->center = glassCenter + Vector(orbitRadius * cos(angle + M_PI_2), orbitRadius * sin(angle + M_PI_2), 0); 
        yellowSphere->center = glassCenter + Vector(orbitRadius * cos(angle + 3 * M_PI_2), orbitRadius * sin(angle + 3 * M_PI_2), 0);  

        // **Render frame**
        Screen screen(Vector(1.0, 0.0, 0.0), 500, 500);
        screen.pov = 0.9; // Decrease pov for a wider field of view
        processScreen(screen, origin, objects, light, ambientLight, AA_REGULAR_4, reflMethod);

        // **Save frame**
        std::ostringstream filenameStream;
        filenameStream << "output/frame_" << std::setw(4) << std::setfill('0') << i << ".jpg";
        std::string filename = filenameStream.str();
        screen.writeToJPG(filename);
    }
    */

    // **Cleanup dynamically allocated objects**
    /*
    for (Sphere* sphere_to_delete : spheres) { // `spheres` is the list of small orbiting ones
        delete sphere_to_delete;
    }
    spheres.clear(); // Clear the list of pointers
    */
    objects.clear(); // Clear all objects from the previous scene (dangling small spheres and the floor)
    objects.push_back(&floor); // Re-add the floor for the renderMovingSphere calls. 'floor' is still in scope.

    // Randomize kick_start_z for the free kick
    std::random_device rd_kick_z;
    std::mt19937 gen_kick_z(rd_kick_z());
    // Assuming goal width is 7.32, let's say kick can be from -3.0 to 3.0 for some variation
    std::uniform_real_distribution<> kick_z_dist(-10.0, 10.0);
    double random_kick_start_z = kick_z_dist(gen_kick_z);

    // Define and add the defensive wall
    // Wall position: 9.15m from kick (KICK_X = 25.0) -> Wall_X = 25.0 - 9.15 = 15.85
    // Wall Y-center is now halfway between random_kick_start_z and goal center (0.0)
    const double wall_X = 15.85;
    const double wall_Y_center = random_kick_start_z / 2.0; // Halfway between kick_start_z and 0
    const double wall_width = 2.0;
    const double wall_height = 1.8;
    const double floor_Z = -1.0;

    Vector v_wall_bottom_left(wall_X, wall_Y_center - wall_width / 2.0, floor_Z);
    Vector v_wall_bottom_right(wall_X, wall_Y_center + wall_width / 2.0, floor_Z);
    Vector v_wall_top_right(wall_X, wall_Y_center + wall_width / 2.0, floor_Z + wall_height);
    Vector v_wall_top_left(wall_X, wall_Y_center - wall_width / 2.0, floor_Z + wall_height);

    Triangle* wall_tri1 = new Triangle(v_wall_bottom_left, v_wall_bottom_right, v_wall_top_right, &whiteMetal);
    Triangle* wall_tri2 = new Triangle(v_wall_bottom_left, v_wall_top_right, v_wall_top_left, &whiteMetal);
    
    objects.push_back(wall_tri1);
    objects.push_back(wall_tri2);

    // now use renderMovingSphere for free kick and goalkeeper views
    // Pass random_kick_start_z to the first call (free kick view)
    // renderMovingSphere(100, origin, Vector(1.0, 0.0, 0.0), objects, &whiteMetal, light, ambientLight, AA_REGULAR_4, reflMethod, random_kick_start_z);

    Vector goalkeeper_pos(0.0, 0.0, 1.8);
    Vector screen_normal(1.0, 0.0, 0.0);
    // For goalkeeper view, we might want a consistent kick Z or another random one.
    // Using the same random_kick_start_z for consistency in this example.
    renderMovingSphere(500, goalkeeper_pos, screen_normal, objects, &whiteMetal, light, ambientLight, AA_REGULAR_4, reflMethod, random_kick_start_z);

    delete wall_tri1;
    delete wall_tri2;

    return 0;
}