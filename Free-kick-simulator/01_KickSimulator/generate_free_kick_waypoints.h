#pragma once
#include <vector>
#include <random>
#include <cmath>
#include "Vector.h"

// Modified to accept kick_start_z
inline std::vector<Vector> generate_free_kick_waypoints(int N, double kick_start_z) {
    // Field and goal parameters
    const double GOAL_X = 0.0;
    const double GOAL_WIDTH = 7.32;
    const double GOAL_HEIGHT = 2.44;
    const double GOAL_Z_LEFT = -GOAL_WIDTH / 2;
    const double GOAL_Z_RIGHT = GOAL_WIDTH / 2;
    
    // Free kick start position
    const double KICK_X = 25.0; // Distance from goal line
    const double KICK_Y = 0.11 / 2.0; // Ball radius, so it starts on the ground 
    // KICK_Z is now taken from the kick_start_z parameter

    // Random engine
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> ydist(1.0, GOAL_HEIGHT - 1.0); // Defines Y range for target

    // Target inside goal. X is on the goal plane. Y is height. Z is width.
    double TARGET_X = GOAL_X;
    double TARGET_Y = ydist(gen); // Target random height: 1m from ground, 1m from crossbar

    // TARGET_Z logic: randomly target within 1m strip inside either goalpost
    std::uniform_int_distribution<> side_dist(0, 1); // 0 for left post region, 1 for right post region
    int target_side = side_dist(gen);
    double TARGET_Z;

    if (target_side == 0) { // Target 1m region from left post, inwards
        // Generates Z in [GOAL_Z_LEFT, GOAL_Z_LEFT + 1.0)
        std::uniform_real_distribution<> z_left_region_dist(GOAL_Z_LEFT, GOAL_Z_LEFT + 1.0);
        TARGET_Z = z_left_region_dist(gen);
    } else { // Target 1m region from right post, inwards
        // Generates Z in [GOAL_Z_RIGHT - 1.0, GOAL_Z_RIGHT)
        std::uniform_real_distribution<> z_right_region_dist(GOAL_Z_RIGHT - 1.0, GOAL_Z_RIGHT);
        TARGET_Z = z_right_region_dist(gen);
    }

    // Control point for Bezier (arc above ground)
    // p0 is the kick origin, p2 is the target
    Vector p0(KICK_X, KICK_Y, kick_start_z); // Use passed kick_start_z
    Vector p2(TARGET_X, TARGET_Y, TARGET_Z);

    // p1 is the control point
    double CONTROL_X = (p0.x + p2.x) / 2.0;
    // Adjust CONTROL_Y for a proper vertical arc.
    // (KICK_Y + TARGET_Y) / 2.0 would be mid-height, add an offset for the arc.
    double CONTROL_Y = (p0.y + p2.y) / 2.0 + 2.5; // Added 2.5m for vertical arc peak
    // CONTROL_Z for side swerve, as per user's file
    double CONTROL_Z = (p0.z + p2.z) / 2.0 + 2.0; 

    Vector p1(CONTROL_X, CONTROL_Y, CONTROL_Z);

    auto bezier = [](double t, const Vector& P0, const Vector& P1, const Vector& P2) {
        return P0 * ((1-t)*(1-t)) + P1 * (2*(1-t)*t) + P2 * (t*t);
    };

    std::vector<Vector> waypoints;
    for (int i = 0; i < N; ++i) {
        double t = (N <= 1) ? 1.0 : static_cast<double>(i) / static_cast<double>(N - 1); // Avoid division by zero if N=1
        Vector pos = bezier(t, p0, p1, p2);
        waypoints.push_back(pos);
    }
    return waypoints;
}
