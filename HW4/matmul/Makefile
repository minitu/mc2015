TARGET=mat_mul
OBJS=mat_mul.o timers.o

CC=gcc
CFLAGS=-g -O2 -Wall
LDFLAGS=-pthread -lm

all: $(TARGET)

$(TARGET):$(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

clean:
	rm -rf $(TARGET) $(OBJS) task*

run: $(TARGET)
	thorq --add ./$(TARGET) -t 4 -v
