# cuAgg

A high-performance GPU aggregation kernel for GNNs.

## Getting Started

**Downloading Repo**

This project uses Professor [Scott Beamer](https://scottbeamer.net/)'s
[GAP Benchmark Suite](https://github.com/sbeamer/gapbs/). To clone this repo,
use this command.

```bash
git clone --recurse-submodules git@github.com:karimsaraipour/cuAgg.git
```

If you already cloned the repo without the submodules, don't worry! Just run
this command.

```bash
git submodule update --init --recursive
```

**Installing CMake**

```bash
. cmake_install.sh
```

You may need to run `chmod u+x cmake_install.sh` to give execute permission to
the script.

**Building Project**

Make & enter a build folder for CMake.

```bash
mkdir build && cd build
```

Run CMake build generation. Re-run this when new source files get added.

```bash
cmake ..
```

Build project.

```bash
make
```

The excutables can be found in the build folder.

**Contributing to Project**

The compile commands are all described in `CMakeLists.txt`.

To add your own executable (`.cpp` or `.cu` file with a `int main(int, char**)`
function), add the following line.

```cmake
add_executable(<binary name> <path/to/source.cpp or .cu>)
```

If it uses graph or aggregation related code, also add this command
(`graph` for graph; `agg` for aggregation).

```cmake
target_link_libraries(<binary name> graph agg)
```

To add code related to the graph or aggregation code, modify this line (lines
18 and/or 19).

```cmake
add_library(<graph or agg> ... <your new source file1> <file2> <etc>)
```
