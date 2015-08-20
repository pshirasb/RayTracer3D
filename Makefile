CC = g++
CFLAGS = -g -O2 
LIBS = -lm 
SRCS = ./src/raytracer.cpp ./src/util.cpp ./src/light_source.cpp ./src/scene_object.cpp ./src/bmp_io.cpp
SRCDIR = ./src
BUILDIR = ./build


all: 
	$(CC) $(CFLAGS) -o raytracer $(SRCS) $(LIBS)

clean:
	-rm -f core *.o
	-rm raytracer
	



