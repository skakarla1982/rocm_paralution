
#ifndef HIPSPARSE_EXPORT_H
#define HIPSPARSE_EXPORT_H

#ifdef HIPSPARSE_STATIC_DEFINE
#  define HIPSPARSE_EXPORT
#  define HIPSPARSE_NO_EXPORT
#else
#  ifndef HIPSPARSE_EXPORT
#    ifdef hipsparse_EXPORTS
        /* We are building this library */
#      define HIPSPARSE_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define HIPSPARSE_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef HIPSPARSE_NO_EXPORT
#    define HIPSPARSE_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef HIPSPARSE_DEPRECATED
#  define HIPSPARSE_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef HIPSPARSE_DEPRECATED_EXPORT
#  define HIPSPARSE_DEPRECATED_EXPORT HIPSPARSE_EXPORT HIPSPARSE_DEPRECATED
#endif

#ifndef HIPSPARSE_DEPRECATED_NO_EXPORT
#  define HIPSPARSE_DEPRECATED_NO_EXPORT HIPSPARSE_NO_EXPORT HIPSPARSE_DEPRECATED
#endif

#define DEFINE_NO_DEPRECATED 0
#if DEFINE_NO_DEPRECATED
# define HIPSPARSE_NO_DEPRECATED
#endif

#endif
