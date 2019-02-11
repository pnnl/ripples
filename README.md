# ReadMe

## List of dependencies

The following is the complete list of dependencies and the version that have been tested:
- A compiler with C++17 and OpenMP support.
- [Spdlog (v1.1.0)](https://github.com/gabime/spdlog)
- [JSON](https://github.com/nlohmann/json)
- [TRNG4](https://github.com/rabauke/trng4)
- MPI library (only to support distributed systems).

Please use your package manager of choice or follow the instructions of each software package to get dependencies installed on your system.

## Build Instruction

This project uses [WAF](https://waf.io) as its build system.  The build is a two step process: configure and build.

The configure step can be invoked with:
```shell
./waf configure
```
The build system offers options that can be used to help the configuration step locate dependencies (i.e., they are installed in unconventional paths).  A complete list of the options can be obtained with:
```shell
./waf configure --help
```

After the configuration step suceed, the build step can be executed running:
```shell
./waf build
```

## Usage

To execute the tool that implements IMM, run from the root of this repository:
```shell
./build/tools/imm --help
```
the command will print the help menu.  The only graph input format that is accepet right now is an edge list stored in a text file.  Support for more input formats is planned.
