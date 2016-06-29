CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`
ALLFLAGS = $(CFLAGS) $(LIBS) -std=c++11 -lboost_filesystem -lboost_system -lpython2.7 -lboost_python -I/usr/include/python2.7/

all: relocalizer

relocalizer: relocalize.o
	g++ -g relocalize.o -o relocalizer $(ALLFLAGS)

relocalize.o: relocalize.cpp relocalize.h
	g++ -g -c relocalize.cpp relocalize.h $(ALLFLAGS)

clean:
	rm *.o relocalizer
