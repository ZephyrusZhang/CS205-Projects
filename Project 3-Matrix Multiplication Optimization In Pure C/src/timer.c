#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "../include/timer.h"

void lock(TIMER *timer) {
    gettimeofday(&(timer->start), NULL);
}

void unlock(TIMER *timer) {
    gettimeofday(&(timer->end), NULL);
    timer->time_used = (timer->end.tv_sec - timer->start.tv_sec) * 1000000 + (timer->end.tv_usec - timer->start.tv_usec);
    printf("---> Time consumed: %.6fs\n", timer->time_used / 1000000.0);
}

