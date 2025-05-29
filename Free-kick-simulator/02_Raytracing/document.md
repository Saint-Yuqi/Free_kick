# Exercises - Week 4

INFO: Even though i talk about data parallelism in C++ at the end, the experiments are already using OpenMP and Vectorisation (AVX) - See makefile compile command:
```
g++ main.cpp -O3 -pg -mavx -DUSE_AVX -ftree-vectorize -fopt-info-vec-optimized -fopenmp -o main 
```

## Exercise 1 - Compare Fresnel's Equations with Schlick's Approximation

As seen in the figure, Schlick's Approximation generally captures the lower-frequency components of the Fresnel function's behavior. However, it lacks accuracy in representing higher-frequency components, such as sharp peaks. Additionally, in the right plot, it fails to exhibit the necessary asymptotic behavior, particularly in cases of total internal reflection. For this reason, Schlick's Approximation needs to be evaluated constructively using conditional statements.
![alt text](benchmark_fresnel.png)

Regarding the performance, a small benchmark has been conducted. For the experimental setup, a scene has been constructed where a big glass sphere is placed on the center of a rotating system. Additionally, four small metallic balls with different colors are orbiting the big glass sphere. The total execution time of the program has been measured in relation to the depth of the recursive rays.
In this scene, it is expected that many recursive rays are needed because of the refractive properties of the glass. For this benchmark, only one frame has been calculated. As seen on the figure, both methods seem to not have a big influence on the total execution time of the program. Perhaps the performacne gain would only be visible on bigger scenes, where multiple refractive objects are presented in the scene.
In order to study exactly how these methods are performing, one should only measure the execution time of the functions and not the total program.
![alt text](performance_benchmark.png)

## Exercise 2 - Check if intersection point is on triangle
To try to find a way to check whether an intersection point is on a triangle, i relied on geometry to try to find a relation. My idea was the following: Define Vectors a,b,c which represent the vectors from the intersection point to the vertices. Additionally, define the vectors a',b' and c' which represent the vectors from the midpoint of each edge to the perpendicular vertice. My assumption was that the intersection point would only lie on the triangle if the norm of the a,b,c vectors would be smaller than to their corresponding prime vectors a',b',c'. See next figure to get the visual idea.
![alt text](IMG_7261.JPG)

The idea worked partially. the triangle seems to show curves, which could mean that the midpoint approach may not be correct. See next figure  
![alt text](output/triangle_with_own_brain.jpg)

After researching on the internet, one correct approach would involve vector geometry, in particular calculating the cross product of the edges with the a,b,c vectors and dot product with the normal. If the dot product is negative, then the point would be outside of the triangle - very clever!
Here is the image after using the edge approach  
![alt text](output/triangle_with_looking_up.jpg)

## Exercise 3 - Paragraph about data parallelism in C++

When one reaches the limits of single core executions, the next step is to parallelise these executions.

One programming paradigm that can be followed is data parallelism.

With data parallelism, hardware would execute the same intended instruction on different data points.

In C++ there are several data parallel possibilities to explore such as OpenMP, Vectorisation (AVX) or when focusing on the GPU, OpenACC or CUDA for example.

However, there are clear distinctions between these approaches. For example:

While OpenMP and Vectorisation act in a data parallel way, their level of execution is different. OpenMP is more high-level (Focusing on threads) and Vectorisation focuses on registers inside a core.

With Vectorisation one can achieve higher speedups with just one core whereas with OpenMP one would need multiple cores. A key advantage is that both can be combined, thus resulting in a much higher speedup leveraging speedups inside a core plus the usage of multiple cores.

In my opinion, one fundamental algebraic principle should be followed intuitively to maximize parallelism within a program's logic: The instruction code should not in generall depend on multiple data accesses with different indexes, because the execution order may not be guaranteed, thus resulting in race conditions or inconsistent results.
