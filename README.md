# ReadMe

## List of dependencies

The following is the complete list of dependencies and the version that have been tested:
- A compiler with C++14 and OpenMP support.
- [Spdlog (v1.1.0)](https://github.com/gabime/spdlog)
- [JSON](https://github.com/nlohmann/json)
- [TRNG4](https://github.com/rabauke/trng4)

Please use your package manager of choice or follow their instructions to get them installed on your system.

## Buldind Instruction

To configure and build the project run the following commands from the root of the project:

```shell
./waf configure
./waf build
```

If you have installed some of the dependencies in uncoventional paths (like the OpenMP library on MacOS X):
```shell
./waf configure --help
```
will list the options to help find libraries where they are installed.  For example, you use brew and you have installed llvm from it:
```shell
./waf configure --openmp-root=/usr/local/opt/llvm
```
Similar options are present for all the dependencies.

## Usage

To execute the tool that implements IMM, run from the root of this repository:
```shell
./build/tools/imm --help
```
the command will print the help menu.  The only graph input format that is accepet right now is an edge list stored in a text file.  Support for more input formats is planned.
