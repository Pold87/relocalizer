CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`
ALLFLAGS = $(CFLAGS) $(LIBS) -std=c++11 -lboost_filesystem -lboost_system

all: relocalize_folder

relocalize_folder: relocalize_folder.o
	g++ -g relocalize_folder.o -o relocalize_folder relocalize.cpp $(ALLFLAGS)

relocalize_folder.o: relocalize.cpp relocalize.h
	g++ -g -c relocalize_folder.cpp relocalize.cpp relocalize.h $(ALLFLAGS)

show_matches:
	g++ -g show_matches.cpp -o show_matches  $(ALLFLAGS)

clean:
	rm *.o relocalize_folder
