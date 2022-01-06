#ifndef CNN_FACEDETECTION_MACRO_HPP
#define CNN_FACEDETECTION_MACRO_HPP

#if defined(NDEBUG)
#ifndef ONLY4DEBUG
#define ONLY4DEBUG(statement)
#endif
#else
#ifndef ONLY4DEBUG
#define ONLY4DEBUG(statement) statement
#endif
#endif

#if defined(DEBUG)
#ifndef ASSERT
#define ASSERT(expr, error_message) \
if (!(expr)) \
{ \
    fprintf(stderr, "\033[31mfile: %s => line: %d => func: %s => Assertion \'%s\' failed => \'%s\'\033[0m\n", \
    __FILE__, __LINE__, __FUNCTION__, #expr, error_message); \
    exit(EXIT_FAILURE); \
}
#endif
#else
#ifndef ASSERT
#define ASSERT(expr, error_message) \
if (!(expr)) \
    exit(EXIT_FAILURE);
#endif
#endif

#ifndef force_cast
#define force_cast(type) (type)
#endif

#ifndef DECORATOR_REDIRECT
#define DECORATOR_REDIRECT(filename, statement) \
freopen(filename, "w", stdout);        \
statement                                  \
freopen("/dev/tty", "w", stdout);
#endif

#ifndef REDIRECT2
#define REDIRECT2(file) freopen(file, "w", stdout);
#endif

#ifndef REDIRECT2STDOUT
#define REDIRECT2STDOUT freopen("/dev/tty", "w", stdout);
#endif

#ifndef EXIT
#define EXIT exit(EXIT_FAILURE);
#endif

#ifndef SWAP
#define SWAP(a, b) { (a) = (b) - (a); (b) = (b) - (a); (a) = (a) + (b); }
#endif

#ifndef DEBUG_PRINT
#define DEBUG_PRINT cout << "<================Here================>" << endl;
#endif

#ifndef CNN_ALWAYS_INLINE
#ifdef __GNUC__
#define CNN_ALWAYS_INLINE inline __attribute__((always_inline))
#endif
#endif

#endif //CNN_FACEDETECTION_MACRO_HPP