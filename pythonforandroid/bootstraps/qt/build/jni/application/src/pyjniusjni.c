#include <pthread.h>
#include <jni.h>
#include "android/log.h"

#define LOG_TAG "Python_android"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

JNIEnv* Android_JNI_GetEnv(void);
static void Android_JNI_ThreadDestroyed(void*);

static pthread_key_t mThreadKey;
static JavaVM* mJavaVM;

int Android_JNI_SetupThread(void)
{
    Android_JNI_GetEnv();
    return 1;
}

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved)
{
    JNIEnv *env;
    mJavaVM = vm;
    LOGI("JNI_OnLoad called");
    if ((*mJavaVM)->GetEnv(mJavaVM, (void**) &env, JNI_VERSION_1_4) != JNI_OK) {
        LOGE("Failed to get the environment using GetEnv()");
        return -1;
    }

    if (pthread_key_create(&mThreadKey, Android_JNI_ThreadDestroyed) != 0) {
        __android_log_print(ANDROID_LOG_ERROR, "pyjniusjni", "Error initializing pthread key");
    }
    Android_JNI_SetupThread();

    return JNI_VERSION_1_4;
}

JNIEnv* Android_JNI_GetEnv(void)
{
    JNIEnv *env;
    int status = (*mJavaVM)->AttachCurrentThread(mJavaVM, &env, NULL);
    if(status < 0) {
        LOGE("failed to attach current thread");
        return 0;
    }

    pthread_setspecific(mThreadKey, (void*) env);

    return env;
}

static void Android_JNI_ThreadDestroyed(void* value)
{
    JNIEnv *env = (JNIEnv*) value;
    if (env != NULL) {
        (*mJavaVM)->DetachCurrentThread(mJavaVM);
        pthread_setspecific(mThreadKey, NULL);
    }
}

void *WebView_AndroidGetJNIEnv()
{
    return Android_JNI_GetEnv();
}
