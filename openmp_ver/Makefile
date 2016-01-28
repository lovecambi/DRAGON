CC		 = /usr/local/bin/clang-omp++ -O3 -std=c++11
INCLUDES        = -I /usr/local/include
LIBS            = /usr/local/lib/libiomp5.dylib
default: remote
remote: 
	icpc -O3 -std=c++11 -o test test.cpp -fopenmp
mac:
	$(CC) $(LIBS) $(INCLUDES) test.cpp -o test -fopenmp
clean:
	rm -f *.o test
	
.PHONY: clean
