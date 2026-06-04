import re
from pathlib import Path

import sh
from pythonforandroid.logger import error
from pythonforandroid.recipe import PythonRecipe, current_directory, info_main, shprint, warning


class TFLiteRuntimeRecipe(PythonRecipe):
    version = "2.18.0"
    url = "https://github.com/tensorflow/tensorflow/archive/refs/tags/v{version}.zip"
    depends = ["numpy"]
    hostpython_prerequisites = ["pybind11==2.11.1"]
    site_packages_name = "tflite-runtime"
    call_hostpython_via_targetpython = False

    def should_build(self, arch):
        name = self.folder_name.replace("-", "_")
        if self.ctx.has_package(name, arch):
            init_file = Path(self.ctx.get_site_packages_dir(arch)) / name / "__init__.py"
            if init_file.exists():
                installed_init = init_file.read_text()
                if (
                    "__version__ = '{}'".format(self.version) in installed_init
                    or '__version__ = "{}"'.format(self.version) in installed_init
                ):
                    info_main("tflite-runtime {} already exists in site-packages".format(self.version))
                    return False
            info_main("tflite-runtime exists in site-packages, but not version {}; rebuilding".format(self.version))
            return True
        info_main("{} apparently isn't already in site-packages".format(name))
        return True

    def _replace_once(self, text, old, new, label):
        if new in text:
            return text
        if old not in text:
            warning("tflite-runtime 2.18 patch skipped: {} anchor not found".format(label))
            return text
        return text.replace(old, new, 1)

    def _strip_android_cases(self, text):
        lines = text.splitlines(keepends=True)
        stripped = []
        i = 0
        while i < len(lines):
            if lines[i].startswith("  android)"):
                i += 1
                while i < len(lines):
                    if lines[i].strip() == ";;":
                        i += 1
                        break
                    i += 1
                continue
            stripped.append(lines[i])
            i += 1
        return "".join(stripped)

    def _strip_p4a_cache_restore_blocks(self, text):
        lines = text.splitlines(keepends=True)
        stripped = []
        i = 0
        while i < len(lines):
            if lines[i] == 'P4A_TFLITE_CMAKE_CACHE_ENTRIES="\n':
                i += 1
                while i < len(lines) and lines[i] != '"\n':
                    i += 1
                if i < len(lines):
                    i += 1
                if i < len(lines) and lines[i].startswith('if [ -n "${P4A_TFLITE_CMAKE_CACHE_DIR}" ]'):
                    depth = 0
                    while i < len(lines):
                        stripped_line = lines[i].strip()
                        if stripped_line.startswith("if ") or stripped_line.startswith("for "):
                            depth += 1
                        elif stripped_line in ("fi", "done"):
                            depth -= 1
                        i += 1
                        if depth == 0:
                            break
                while i < len(lines) and lines[i].strip() == "":
                    i += 1
                continue
            stripped.append(lines[i])
            i += 1
        return "".join(stripped)

    def _strip_ml_dtypes_cache_flags(self, text):
        lines = text.splitlines(keepends=True)
        stripped = []
        i = 0
        while i < len(lines):
            if lines[i] == 'ML_DTYPES_FETCHCONTENT_FLAG=""\n':
                i += 1
                while i < len(lines):
                    current = lines[i].strip()
                    i += 1
                    if current == "fi":
                        break
                while i < len(lines) and lines[i].strip() == "":
                    i += 1
                continue
            stripped.append(lines[i])
            i += 1
        return "".join(stripped)

    def _patch_build_script(self, script_path):
        script = Path(script_path)
        text = script.read_text()

        text = text.replace(
            'PYBIND11_INCLUDE=$(${PYTHON} -c "import pybind11; print (pybind11.get_include())")',
            '# PYBIND11_INCLUDE=$(${PYTHON} -c "import pybind11; print (pybind11.get_include())")',
        )
        text = text.replace(
            'NUMPY_INCLUDE=$(${PYTHON} -c "import numpy; print (numpy.get_include())")',
            '# NUMPY_INCLUDE=$(${PYTHON} -c "import numpy; print (numpy.get_include())")',
        )
        text = text.replace(
            'echo "__git_version__ = \'$(git -C "${TENSORFLOW_DIR}" describe)\'" >> "${BUILD_DIR}/tflite_runtime/__init__.py"',
            'echo "__git_version__ = \'${PACKAGE_VERSION}\'" >> "${BUILD_DIR}/tflite_runtime/__init__.py"',
        )

        cache_block = r'''
P4A_TFLITE_CMAKE_CACHE_ENTRIES="
  FP16 FP16-download FP16-source
  FXdiv FXdiv-download FXdiv-source
  abseil-cpp cpuinfo clog eigen farmhash fft2d ml_dtypes protobuf
  flatbuffers flatbuffers-flatc gemmlowp neon2sse
  psimd psimd-download psimd-source
  pthreadpool pthreadpool-download pthreadpool-source
  ruy xnnpack
"
if [ -n "${P4A_TFLITE_CMAKE_CACHE_DIR}" ] && [ -d "${P4A_TFLITE_CMAKE_CACHE_DIR}" ]; then
  for cache_entry in ${P4A_TFLITE_CMAKE_CACHE_ENTRIES}; do
    if [ -e "${P4A_TFLITE_CMAKE_CACHE_DIR}/${cache_entry}" ] && [ ! -e "${cache_entry}" ]; then
      cp -a "${P4A_TFLITE_CMAKE_CACHE_DIR}/${cache_entry}" .
    fi
  done
  if [ ! -e clog/deps/clog ] && [ -e "${P4A_TFLITE_CMAKE_CACHE_DIR}/cpuinfo/deps/clog" ]; then
    mkdir -p clog/deps
    cp -a "${P4A_TFLITE_CMAKE_CACHE_DIR}/cpuinfo/deps/clog" clog/deps/
  fi
  for stamp_dir in \
    FXdiv-download/fxdiv-prefix/src/fxdiv-stamp \
    psimd-download/psimd-prefix/src/psimd-stamp; do
    stamp_name="$(basename "${stamp_dir}" -stamp)"
    if [ -d "${stamp_dir}" ] && [ ! -e "${stamp_dir}/${stamp_name}-update" ]; then
      touch "${stamp_dir}/${stamp_name}-update"
    fi
  done
fi
'''
        ml_dtypes_cache_flag = r'''
ML_DTYPES_FETCHCONTENT_FLAG=""
if [ -n "${P4A_TFLITE_CMAKE_CACHE_DIR}" ] && [ -f "${P4A_TFLITE_CMAKE_CACHE_DIR}/ml_dtypes/setup.py" ]; then
  ML_DTYPES_FETCHCONTENT_FLAG="-DFETCHCONTENT_SOURCE_DIR_ML_DTYPES=${P4A_TFLITE_CMAKE_CACHE_DIR}/ml_dtypes"
fi
'''
        text = self._strip_ml_dtypes_cache_flags(text)
        text = self._strip_p4a_cache_restore_blocks(text)
        text = self._replace_once(
            text,
            'mkdir -p "${BUILD_DIR}/cmake_build"\ncd "${BUILD_DIR}/cmake_build"\n',
            'mkdir -p "${BUILD_DIR}/cmake_build"\ncd "${BUILD_DIR}/cmake_build"\n' + cache_block + ml_dtypes_cache_flag,
            "cmake cache restore",
        )

        host_tools_block = r'''# Build host tools
if [[ "${TENSORFLOW_TARGET}" != "native" ]]; then
  echo "Building for host tools."
  HOST_BUILD_DIR="${BUILD_DIR}/cmake_build_host"
  mkdir -p "${HOST_BUILD_DIR}/flatbuffers-flatc/bin"

  FLATBUFFERS_SOURCE_DIR=""
  for flatbuffers_candidate in \
    "${P4A_TFLITE_CMAKE_CACHE_DIR}/flatbuffers" \
    "${HOST_BUILD_DIR}/flatbuffers" \
    "${BUILD_DIR}/cmake_build/flatbuffers"; do
    if [ -f "${flatbuffers_candidate}/CMakeLists.txt" ]; then
      FLATBUFFERS_SOURCE_DIR="${flatbuffers_candidate}"
      break
    fi
  done
  if [ -z "${FLATBUFFERS_SOURCE_DIR}" ]; then
    echo "Missing cached flatbuffers source for host flatc." >&2
    echo "Please populate P4A_TFLITE_CMAKE_CACHE_DIR/flatbuffers, e.g. from https://github.com/google/flatbuffers" >&2
    exit 1
  fi

  rm -rf "${HOST_BUILD_DIR}/flatbuffers-build"
  mkdir -p "${HOST_BUILD_DIR}/flatbuffers-build"
  pushd "${HOST_BUILD_DIR}/flatbuffers-build"
  env -u CC -u CXX -u CPP -u LD -u AR -u RANLIB -u STRIP \
      -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS \
      -u CMAKE_TOOLCHAIN_FILE -u CMAKE_SYSTEM_NAME -u ANDROID_PLATFORM -u ANDROID_ABI \
    cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DFLATBUFFERS_BUILD_TESTS=OFF \
      -DFLATBUFFERS_BUILD_FLATLIB=OFF \
      -DFLATBUFFERS_STATIC_FLATC=OFF \
      -DFLATBUFFERS_BUILD_FLATHASH=OFF \
      "${FLATBUFFERS_SOURCE_DIR}"
  env -u CC -u CXX -u CPP -u LD -u AR -u RANLIB -u STRIP \
      -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS \
      -u CMAKE_TOOLCHAIN_FILE -u CMAKE_SYSTEM_NAME -u ANDROID_PLATFORM -u ANDROID_ABI \
    cmake --build . --verbose -j ${BUILD_NUM_JOBS} -t flatc
  cp flatc "${HOST_BUILD_DIR}/flatbuffers-flatc/bin/flatc"
  popd
fi
'''
        original_host_tools_block = r'''# Build host tools
if [[ "${TENSORFLOW_TARGET}" != "native" ]]; then
  echo "Building for host tools."
  HOST_BUILD_DIR="${BUILD_DIR}/cmake_build_host"
  mkdir -p "${HOST_BUILD_DIR}"
  pushd "${HOST_BUILD_DIR}"
  cmake "${TENSORFLOW_LITE_DIR}"
  cmake --build . --verbose -j ${BUILD_NUM_JOBS} -t flatbuffers-flatc
  popd
fi
'''
        android_env_host_tools_block = original_host_tools_block.replace(
            '  cmake "${TENSORFLOW_LITE_DIR}"\n',
            '  env -u CMAKE_TOOLCHAIN_FILE -u CMAKE_SYSTEM_NAME -u ANDROID_PLATFORM -u ANDROID_ABI \\\n'
            '    cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 "${TENSORFLOW_LITE_DIR}"\n',
        )
        if host_tools_block not in text:
            if original_host_tools_block in text:
                text = text.replace(original_host_tools_block, host_tools_block, 1)
            elif android_env_host_tools_block in text:
                text = text.replace(android_env_host_tools_block, host_tools_block, 1)
            else:
                warning("tflite-runtime 2.18 patch skipped: host flatc block anchor not found")

        android_case = r'''  android)
    BUILD_FLAGS=${BUILD_FLAGS:-"${WRAPPER_INCLUDES}"}
    cmake \
      -DCMAKE_SYSTEM_NAME=Android \
      -DANDROID_ARM_NEON=ON \
      -DCMAKE_CXX_FLAGS="${BUILD_FLAGS}" \
      -DCMAKE_SHARED_LINKER_FLAGS="${CMAKE_SHARED_LINKER_FLAGS}" \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DCMAKE_TOOLCHAIN_FILE="${CMAKE_TOOLCHAIN_FILE}" \
      -DANDROID_PLATFORM="${ANDROID_PLATFORM}" \
      -DANDROID_ABI="${ANDROID_ABI}" \
      -DTFLITE_ENABLE_XNNPACK=OFF \
      -DTFLITE_HOST_TOOLS_DIR="${HOST_BUILD_DIR}" \
      -DFETCHCONTENT_SOURCE_DIR_FLATBUFFERS="${P4A_TFLITE_CMAKE_CACHE_DIR}/flatbuffers" \
      "-DFETCHCONTENT_SOURCE_DIR_ABSEIL-CPP=${P4A_TFLITE_CMAKE_CACHE_DIR}/abseil-cpp" \
      -DFETCHCONTENT_SOURCE_DIR_EIGEN="${P4A_TFLITE_CMAKE_CACHE_DIR}/eigen" \
      -DFETCHCONTENT_SOURCE_DIR_FARMHASH="${P4A_TFLITE_CMAKE_CACHE_DIR}/farmhash" \
      -DFETCHCONTENT_SOURCE_DIR_FFT2D="${P4A_TFLITE_CMAKE_CACHE_DIR}/fft2d" \
      -DFETCHCONTENT_SOURCE_DIR_GEMMLOWP="${P4A_TFLITE_CMAKE_CACHE_DIR}/gemmlowp" \
      -DFETCHCONTENT_SOURCE_DIR_NEON2SSE="${P4A_TFLITE_CMAKE_CACHE_DIR}/neon2sse" \
      -DFETCHCONTENT_SOURCE_DIR_CPUINFO="${P4A_TFLITE_CMAKE_CACHE_DIR}/cpuinfo" \
      -DFETCHCONTENT_SOURCE_DIR_XNNPACK="${P4A_TFLITE_CMAKE_CACHE_DIR}/xnnpack" \
      -DFETCHCONTENT_SOURCE_DIR_RUY="${P4A_TFLITE_CMAKE_CACHE_DIR}/ruy" \
      -DFETCHCONTENT_SOURCE_DIR_PROTOBUF="${P4A_TFLITE_CMAKE_CACHE_DIR}/protobuf" \
      ${ML_DTYPES_FETCHCONTENT_FLAG} \
      "${TENSORFLOW_LITE_DIR}"
    ;;
'''
        text = self._strip_android_cases(text)
        text = self._replace_once(text, "  *)\n    BUILD_FLAGS=", android_case + "  *)\n    BUILD_FLAGS=", "android cmake case")

        text = text.replace(
            '${PYTHON} setup.py bdist bdist_wheel',
            '${PYTHON} setup.py bdist',
        )

        save_cache_block = r'''
if [ -n "${P4A_TFLITE_CMAKE_CACHE_DIR}" ]; then
  mkdir -p "${P4A_TFLITE_CMAKE_CACHE_DIR}"
  for cache_entry in ${P4A_TFLITE_CMAKE_CACHE_ENTRIES}; do
    if [ -e "${cache_entry}" ]; then
      cp -a "${cache_entry}" "${P4A_TFLITE_CMAKE_CACHE_DIR}/"
    fi
  done
fi

'''
        text = self._replace_once(
            text,
            '# Build debian package.\n',
            save_cache_block + '# Build debian package.\n',
            "cmake cache save",
        )

        script.write_text(text)

    def _patch_cmake_lists(self, cmake_path):
        cmake_file = Path(cmake_path)
        text = cmake_file.read_text()

        text = self._replace_once(
            text,
            'project(tensorflow-lite C CXX)\n',
            'project(tensorflow-lite C CXX)\nif("${CMAKE_SYSTEM_NAME}" STREQUAL "Android")\n  find_library(ANDROID_LOG_LIB log)\nendif()\n',
            "android log library lookup",
        )
        old_logging_block = (
            'if(NOT "${CMAKE_SYSTEM_NAME}" STREQUAL "iOS")\n'
            '  list(FILTER TFLITE_SRCS EXCLUDE REGEX ".*minimal_logging_ios\\\\.cc$")\n'
            'endif()\n'
        )
        if old_logging_block in text:
            text = text.replace(
                old_logging_block,
                old_logging_block
                + 'if("${CMAKE_SYSTEM_NAME}" STREQUAL "Android")\n'
                + '  list(FILTER TFLITE_SRCS EXCLUDE REGEX ".*minimal_logging_default\\\\.cc$")\n'
                + 'endif()\n',
                1,
            )
        elif "minimal_logging_android.cc" not in text:
            warning("tflite-runtime 2.18 patch skipped: android minimal logging source anchor not found")

        if "    ${ANDROID_LOG_LIB}\n" not in text:
            text = self._replace_once(
                text,
                "    ${TFLITE_TARGET_DEPENDENCIES}\n)",
                "    ${TFLITE_TARGET_DEPENDENCIES}\n    ${ANDROID_LOG_LIB}\n)",
                "android log link",
            )

        cmake_file.write_text(text)

    def _patch_ml_dtypes_cmake(self, cmake_path):
        cmake_file = Path(cmake_path)
        text = cmake_file.read_text()

        git_fetch_block = '''  GIT_REPOSITORY https://github.com/jax-ml/ml_dtypes
  # Sync with tensorflow/third_party/py/ml_dtypes/workspace.bzl
  GIT_TAG 24084d9ed2c3d45bf83b7a9bff833aa185bf9172
  # It's not currently possible to shallow clone with a GIT TAG
  # as cmake attempts to git checkout the commit hash after the clone
  # which doesn't work as it's a shallow clone hence a different commit hash.
  # https://gitlab.kitware.com/cmake/cmake/-/issues/17770
  # GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
'''
        zip_fetch_url = '  URL https://github.com/jax-ml/ml_dtypes/archive/24084d9ed2c3d45bf83b7a9bff833aa185bf9172.zip\n'
        zip_fetch_block = '''  URL https://github.com/jax-ml/ml_dtypes/archive/24084d9ed2c3d45bf83b7a9bff833aa185bf9172.zip
  LICENSE_FILE "LICENSE"
  LICENSE_URL "https://github.com/jax-ml/ml_dtypes/raw/24084d9ed2c3d45bf83b7a9bff833aa185bf9172/LICENSE"
'''
        if zip_fetch_block not in text:
            if git_fetch_block in text:
                text = text.replace(git_fetch_block, zip_fetch_block, 1)
            elif zip_fetch_url in text:
                text = text.replace(zip_fetch_url, zip_fetch_block, 1)
            else:
                warning("tflite-runtime 2.18 patch skipped: ml_dtypes fetch anchor not found")

        cmake_file.write_text(text)

    def _patch_gemmlowp_cmake(self, cmake_path):
        cmake_file = Path(cmake_path)
        text = cmake_file.read_text()

        text = self._replace_once(
            text,
            'set(BUILD_TESTING ${BUILD_TESTING_TMP})\n',
            'set(BUILD_TESTING ${BUILD_TESTING_TMP})\n'
            'if(NOT TARGET gemmlowp)\n'
            '  add_library(gemmlowp INTERFACE)\n'
            '  target_include_directories(gemmlowp INTERFACE "${gemmlowp_SOURCE_DIR}")\n'
            'endif()\n',
            "gemmlowp interface target",
        )

        cmake_file.write_text(text)

    def _patch_interpreter_wrapper(self, wrapper_path):
        wrapper_file = Path(wrapper_path)
        text = wrapper_file.read_text()

        text = text.replace(
            '#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"\n',
            "",
        )
        text = text.replace(
            '  TfLiteDelegate* xnnpack_delegate = nullptr;\n'
            '  if (default_delegate_latest_features) {\n'
            '    auto opts = TfLiteXNNPackDelegateOptionsDefault();\n'
            '    opts.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;\n'
            '    opts.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_SUBGRAPH_RESHAPING;\n'
            '    opts.num_threads = num_threads;\n'
            '    xnnpack_delegate = TfLiteXNNPackDelegateCreate(&opts);\n'
            '  }\n',
            '  (void)default_delegate_latest_features;\n',
        )
        text = text.replace(
            '  if (default_delegate_latest_features) {\n'
            '    builder.AddDelegate(xnnpack_delegate);\n'
            '  }\n',
            "",
        )

        wrapper_file.write_text(text)

    def build_arch(self, arch):
        if arch.arch == "x86_64":
            warning("******** tflite-runtime x86_64 will not be built *******")
            warning("Use x86 not x86_64")
            return

        env = self.get_recipe_env(arch)
        self.install_hostpython_prerequisites()

        root_dir = self.get_build_dir(arch.arch)
        root = Path(root_dir)
        script_dir = root / "tensorflow" / "lite" / "tools" / "pip_package"
        build_dir = script_dir / "gen" / "tflite_pip" / "python3"
        cmake_cache_dir = Path(self.get_download_cache_dir()) / "cmake_build"

        self._patch_build_script(script_dir / "build_pip_package_with_cmake.sh")
        self._patch_cmake_lists(root / "tensorflow" / "lite" / "CMakeLists.txt")
        self._patch_ml_dtypes_cmake(root / "tensorflow" / "lite" / "tools" / "cmake" / "modules" / "ml_dtypes.cmake")
        self._patch_gemmlowp_cmake(root / "tensorflow" / "lite" / "tools" / "cmake" / "modules" / "gemmlowp.cmake")
        self._patch_interpreter_wrapper(root / "tensorflow" / "lite" / "python" / "interpreter_wrapper" / "interpreter_wrapper.cc")

        python_include_dir = self.ctx.python_recipe.include_root(arch.arch)
        pybind11_include_dir = sh.Command(self.real_hostpython_location)(
            "-c", "import pybind11; print(pybind11.get_include())"
        ).strip()
        numpy_include_dir = Path(self.ctx.get_site_packages_dir(arch)) / "numpy" / "_core" / "include"
        if not numpy_include_dir.exists():
            numpy_include_dir = Path(self.ctx.get_site_packages_dir(arch)) / "numpy" / "core" / "include"
        includes = " -I{} -I{} -I{}".format(
            python_include_dir,
            numpy_include_dir,
            pybind11_include_dir,
        )

        build_script = script_dir / "build_pip_package_with_cmake.sh"
        toolchain = Path(self.ctx.ndk_dir) / "build" / "cmake" / "android.toolchain.cmake"

        with current_directory(root_dir):
            env.update({
                "TENSORFLOW_TARGET": "android",
                "CMAKE_TOOLCHAIN_FILE": str(toolchain),
                "ANDROID_PLATFORM": str(self.ctx.ndk_api),
                "ANDROID_ABI": arch.arch,
                "WRAPPER_INCLUDES": includes,
                "CMAKE_SHARED_LINKER_FLAGS": env["LDFLAGS"],
                "CMAKE_POLICY_VERSION_MINIMUM": "3.5",
                "P4A_TFLITE_CMAKE_CACHE_DIR": str(cmake_cache_dir),
            })
            try:
                info_main("tflite-runtime {} is building...".format(self.version))
                info_main("Expect this to take at least 5 minutes...")
                sh.Command(str(build_script))(_env=env)
            except sh.ErrorReturnCode as e:
                error(str(e.stderr))
                exit(1)

        info_main("Installing tflite-runtime into site-packages")
        with current_directory(str(build_dir)):
            hostpython = sh.Command(self.hostpython_location)
            install_dir = self.ctx.get_python_install_dir(arch.arch)
            env["PACKAGE_VERSION"] = self.version
            env["PROJECT_NAME"] = "tflite_runtime"
            shprint(
                hostpython,
                "setup.py",
                "install",
                "-O2",
                "--root={}".format(install_dir),
                "--install-lib=.",
                _env=env,
            )


recipe = TFLiteRuntimeRecipe()
