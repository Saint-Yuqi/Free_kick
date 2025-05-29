all:
	g++ main.cpp -O3 -mavx -DUSE_AVX -ftree-vectorize -fopt-info-vec-optimized -fopenmp -o main 
fresnel:
	./main 0
schlick:
	./main 1
profiling:
	gprof ./main gmon.out > profile.txt
create_sphere_animation:
	ffmpeg -framerate 30 -pattern_type glob -i 'output/sphere/*.jpg' -q:v 2 sphere.gif
create_torus_animation:
	ffmpeg -framerate 30 -pattern_type glob -i 'output/torus/*.jpg' -q:v 2 torus.gif
benchmark:
	@echo "param1,param2,time" > benchmark_results.csv
	@for i in 0 1; do \
	    for j in 1 2 3 4 5 6 7 8 9 10; do \
	        START=$$(date +%s%N); \
	        ./main $$i $$j > /dev/null; \
	        END=$$(date +%s%N); \
	        TIME=$$(echo "scale=6; ($$END - $$START) / 1000000000" | bc); \
	        echo "$$i,$$j,$$TIME" >> benchmark_results.csv; \
	    done \
	done
clean:
	rm -rf output/*
