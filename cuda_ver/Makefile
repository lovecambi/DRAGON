
INCLUDES        = -I /usr/local/cuda/bin
LIBS            = -L /usr/local/cuda/lib

default: remote
remote: 
	nvcc -std=c++11 -O2 -o test test.cu -lcublas -lcurand
mac:
	nvcc -std=c++11 -O2 $(LIBS) $(INCLUDES) test.cu -o test -lcublas -lcurand
clean:
	rm -f *.o test
	
.PHONY: clean
