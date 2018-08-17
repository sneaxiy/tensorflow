load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_binary",
)

cc_import(
    name = "profiler",
    hdrs = ["gperftools/profiler.h"],
    shared_library = "libprofiler.so",
    #interface_library = "libprofiler.so",
    #system_provided = 1
)

tf_cc_binary(
    name = "example",
    srcs = ["example.cc"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:tensorflow",
        ":profiler"
    ],
    copts = ["-I/usr/local/include", "-L/usr/local/lib"]
)
