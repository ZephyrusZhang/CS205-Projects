#ifndef PROJECT_3_TIMER_H
#define PROJECT_3_TIMER_H

#define TIME_START lock(&timer);
#define TIME_END(Procedure) printf(Procedure); \
unlock(&timer);

typedef struct Timer {
    struct timeval start;
    struct timeval end;
    long time_used;
} TIMER;

void lock(TIMER *timer);

void unlock(TIMER *timer);

#endif //PROJECT_3_TIMER_H
