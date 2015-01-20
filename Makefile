CC = g++
CFLAGS = -g -O2 
LIBS = -lm 
SRCS = ./src/raytracer.cpp ./src/util.cpp ./src/light_source.cpp ./src/scene_object.cpp ./src/bmp_io.cpp
#OBJS = ./build/raytracer.o ./build/util.o ./build/light_source.o ./build/scene_object.o ./build/bmp_io.o
SRCDIR = ./src
BUILDIR = ./build


all: 
	$(CC) $(CFLAGS) -o raytracer $(SRCS) $(LIBS)

clean:
	-rm -f core *.o
	-rm raytracer
	



