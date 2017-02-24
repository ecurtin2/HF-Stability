all: $(patsubst %.cpp, %.exe, $(wildcard *.cpp))
all: $(patsubst %.cpp, %.log, $(wildcard *.cpp))
all: $(patsubst %.cpp, %, $(wildcard *.cpp))

SRC_DIR = ../src/

%.exe: %.cpp Makefile
	g++ $< -o $@ -I$(SRC_DIR) -std=c++11

%.log: %.exe
	./$< > $@

%: %.log
	./makeexample.sh $@

clean:
	rm -f *.exe
	rm -f *.log
	rm -f *_example*
        

