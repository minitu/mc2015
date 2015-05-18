TARGET=mat_mul
OBJS=mat_mul.o timers.o

CC=gcc
CFLAGS=-g -O2 -Wall
LDFLAGS=-lOpenCL -lm

all: $(TARGET)

$(TARGET):$(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

clean:
	rm -rf $(TARGET) $(OBJS) task*

run: $(TARGET)
	thorq --add --mode single --device gpu ./$(TARGET)

test: $(TARGET)
	thorq --add --mode single --device gpu --timeout 100 ./$(TARGET)

testp: $(TARGET)
	thorq --add --mode single --device gpu --timeout 100 ./$(TARGET) -p
