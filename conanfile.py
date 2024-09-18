from conan.tools.microsoft import is_msvc, msvc_runtime_flag
from conan.tools.build import check_min_cppstd
from conan.tools.scm import Version
from conan.tools import files
from conan import ConanFile
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.gnu import PkgConfigDeps
from conan.errors import ConanInvalidConfiguration
from conan import tools
import os

required_conan_version = ">=1.55.0"

class LintDBConan(ConanFile):
    name = "LintDB"
    description = "LintDB is a late interaction database for vector retrieval."
    topics = ("vector", "ann", "search", "retrieval")
    license = "Apache-2.0"

    settings = "os", "arch", "compiler", "build_type"

    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "build_tests": [True, False],
        "build_benchmarks": [True, False],
        "build_python": [True, False],
        "build_server": [True, False],
    }

    default_options = {
        "shared": False,
        "fPIC": True,
        "build_tests": True,
        "build_benchmarks": True,
        "build_python": True,
        "build_server": False,
    }

    exports_sources = {
        "lintdb/*",
        "third_party/*",
        "tests/*",
        "CMakelists.txt",
        "*.cmake",
        "conanfile.py",
    }

    @property
    def _minimum_cpp_standard(self):
        return 17

    @property
    def _minimum_compilers_version(self):
        return {
            "gcc": "7",
            "clang": "15",
            "apple-clang": "10",
        }

    def requirements(self):
        self.requires("rocksdb/9.5.2")
        self.requires("glog/0.4.0")
        self.requires("jsoncpp/1.9.5")
        self.requires("ms-gsl/4.0.0")
        self.requires("bitsery/5.2.4")
        self.requires("drogon/1.9.6")
        self.requires("taywee-args/6.4.6")

        if self.options.build_tests:
            self.requires("gtest/1.15.0")

        if self.options.build_benchmarks:
            self.requires("benchmark/1.9.0")

    def validate(self):
        if self.settings.os == "Windows":
            raise ConanInvalidConfiguration("Windows not supported")

        if self.settings.compiler.get_safe("cppstd"):
            check_min_cppstd(self, self._minimum_cpp_standard)
        min_version = self._minimum_compilers_version.get(str(self.settings.compiler))
        if not min_version:
            self.output.warn(
                "{} recipe lacks information about the {} compiler support.".format(
                    self.name, self.settings.compiler
                )
            )
        else:
            if Version(self.settings.compiler.version) < min_version:
                raise ConanInvalidConfiguration(
                    "{} requires C++{} support. The current compiler {} {} does not support it.".format(
                        self.name,
                        self._minimum_cpp_standard,
                        self.settings.compiler,
                        self.settings.compiler.version,
                    )
                )

    def configure(self):
        self.options["rocksdb/*"].use_rtti = True

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["CMAKE_POSITION_INDEPENDENT_CODE"] = self.options.get_safe(
            "fPIC", True
        )

        cxx_std_flag = tools.build.cppstd_flag(self)
        cxx_std_value = (
            cxx_std_flag.split("=")[1]
            if cxx_std_flag
            else "c++{}".format(self._minimum_cpp_standard)
        )
        tc.variables["CXX_STD"] = cxx_std_value

        tc.variables["BUILD_TESTS"] = self.options.build_tests
        tc.variables["BUILD_BENCHMARKS"] = self.options.build_benchmarks
        tc.variables["BUILD_PYTHON"] = self.options.build_python
        tc.variables["BUILD_SERVER"] = self.options.build_server

        tc.generate()

        deps = CMakeDeps(self)
        deps.generate()

        pc = PkgConfigDeps(self)
        pc.generate()

    def build(self):
        # files.apply_conandata_patches(self)
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()
        files.rmdir(self, os.path.join(self.package_folder, "lib", "cmake"))
        files.rmdir(self, os.path.join(self.package_folder, "lib", "pkgconfig"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "lintdb")
        self.cpp_info.set_property("cmake_target_name", "lintdb::LintDB")
        self.cpp_info.set_property("pkg_config_name", "liblintdb")

        self.cpp_info.components["liblintdb"].libs = ["lintdb"]