/*
 * Workaround for running TensorFlow 2.21 on an older NVIDIA driver.
 *
 * TF 2.21 resolves CUDA driver entry points with dlsym().
 * CudaExecutor::Init() queries CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED (132)
 * unconditionally and treats CUDA_ERROR_INVALID_VALUE from old drivers as a
 * fatal error.  Intercept dlsym("cuDeviceGetAttribute") when called from
 * TensorFlow and report "not supported" for attributes the driver does not
 * know.  Calls coming from the WSL libcuda wrapper itself are passed through
 * to avoid recursion.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>

typedef int CUresult;
typedef int CUdevice;
typedef int CUdevice_attribute;

static CUresult (*real_cuDeviceGetAttribute)(int *, CUdevice_attribute, CUdevice);
static void *(*real_dlsym_fn)(void *, const char *);
static CUresult compat_cuDeviceGetAttribute(int *, CUdevice_attribute, CUdevice);

__attribute__((constructor)) static void init_shim(void) {
    void *libc = dlopen("libc.so.6", RTLD_LAZY | RTLD_NOLOAD);
    if (libc == NULL)
        libc = dlopen("libc.so.6", RTLD_LAZY);
    if (libc != NULL)
        real_dlsym_fn = (void *(*)(void *, const char *))dlvsym(libc, "dlsym", "GLIBC_2.34");
    if (real_dlsym_fn == NULL)
        real_dlsym_fn = (void *(*)(void *, const char *))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
}

static int caller_is_tensorflow(void *caller) {
    Dl_info info;
    if (dladdr(caller, &info) == 0 || info.dli_fname == NULL)
        return 0;
    return strstr(info.dli_fname, "libtensorflow") != NULL ||
           strstr(info.dli_fname, "lib_pywrap_tf") != NULL;
}

static CUresult compat_cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
    if (real_cuDeviceGetAttribute == NULL) {
        void *h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL);
        if (h == NULL)
            return 100; /* CUDA_ERROR_NO_DEVICE */
        real_cuDeviceGetAttribute =
            (CUresult (*)(int *, CUdevice_attribute, CUdevice))real_dlsym_fn(h, "cuDeviceGetAttribute");
        if (real_cuDeviceGetAttribute == NULL)
            return 500; /* CUDA_ERROR_NOT_FOUND */
    }

    CUresult r = real_cuDeviceGetAttribute(pi, attrib, dev);

    /* Attributes >= 132 were added after CUDA 12.0; old drivers return
     * CUDA_ERROR_INVALID_VALUE for them.  Report them as unsupported
     * instead of failing device initialization. */
    if (r == 1 && attrib >= 132 && pi != NULL) {
        *pi = 0;
        return 0;
    }
    return r;
}

void *dlsym(void *handle, const char *name) {
    if (real_dlsym_fn == NULL)
        init_shim();
    if (strcmp(name, "cuDeviceGetAttribute") == 0 &&
        caller_is_tensorflow(__builtin_return_address(0)))
        return (void *)&compat_cuDeviceGetAttribute;
    return real_dlsym_fn(handle, name);
}
