# CMAKE generated file: DO NOT EDIT!
# Generated by "NMake Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE
NULL=nul
!ENDIF
SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = C:\Users\Enzo\AppData\Local\JetBrains\Toolbox\apps\CLion\ch-0\211.7442.42\bin\cmake\win\bin\cmake.exe

# The command to remove a file.
RM = C:\Users\Enzo\AppData\Local\JetBrains\Toolbox\apps\CLion\ch-0\211.7442.42\bin\cmake\win\bin\cmake.exe -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\Enzo\Documents\Github\Country_guesser\mlp_library

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\Enzo\Documents\Github\Country_guesser\mlp_library\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles\mlp_library.dir\depend.make

# Include the progress variables for this target.
include CMakeFiles\mlp_library.dir\progress.make

# Include the compile flags for this target's objects.
include CMakeFiles\mlp_library.dir\flags.make

CMakeFiles\mlp_library.dir\mlp.cpp.obj: CMakeFiles\mlp_library.dir\flags.make
CMakeFiles\mlp_library.dir\mlp.cpp.obj: ..\mlp.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\Enzo\Documents\Github\Country_guesser\mlp_library\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/mlp_library.dir/mlp.cpp.obj"
	C:\PROGRA~2\MICROS~3\2019\COMMUN~1\VC\Tools\MSVC\1427~1.291\bin\Hostx86\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\mlp_library.dir\mlp.cpp.obj /FdCMakeFiles\mlp_library.dir\ /FS -c C:\Users\Enzo\Documents\Github\Country_guesser\mlp_library\mlp.cpp
<<

CMakeFiles\mlp_library.dir\mlp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mlp_library.dir/mlp.cpp.i"
	C:\PROGRA~2\MICROS~3\2019\COMMUN~1\VC\Tools\MSVC\1427~1.291\bin\Hostx86\x64\cl.exe > CMakeFiles\mlp_library.dir\mlp.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\Enzo\Documents\Github\Country_guesser\mlp_library\mlp.cpp
<<

CMakeFiles\mlp_library.dir\mlp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mlp_library.dir/mlp.cpp.s"
	C:\PROGRA~2\MICROS~3\2019\COMMUN~1\VC\Tools\MSVC\1427~1.291\bin\Hostx86\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\mlp_library.dir\mlp.cpp.s /c C:\Users\Enzo\Documents\Github\Country_guesser\mlp_library\mlp.cpp
<<

# Object files for target mlp_library
mlp_library_OBJECTS = \
"CMakeFiles\mlp_library.dir\mlp.cpp.obj"

# External object files for target mlp_library
mlp_library_EXTERNAL_OBJECTS =

mlp_library.exe: CMakeFiles\mlp_library.dir\mlp.cpp.obj
mlp_library.exe: CMakeFiles\mlp_library.dir\build.make
mlp_library.exe: CMakeFiles\mlp_library.dir\objects1.rsp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\Enzo\Documents\Github\Country_guesser\mlp_library\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable mlp_library.exe"
	C:\Users\Enzo\AppData\Local\JetBrains\Toolbox\apps\CLion\ch-0\211.7442.42\bin\cmake\win\bin\cmake.exe -E vs_link_exe --intdir=CMakeFiles\mlp_library.dir --rc=C:\PROGRA~2\WI3CF2~1\10\bin\100190~1.0\x86\rc.exe --mt=C:\PROGRA~2\WI3CF2~1\10\bin\100190~1.0\x86\mt.exe --manifests -- C:\PROGRA~2\MICROS~3\2019\COMMUN~1\VC\Tools\MSVC\1427~1.291\bin\Hostx86\x64\link.exe /nologo @CMakeFiles\mlp_library.dir\objects1.rsp @<<
 /out:mlp_library.exe /implib:mlp_library.lib /pdb:C:\Users\Enzo\Documents\Github\Country_guesser\mlp_library\cmake-build-debug\mlp_library.pdb /version:0.0 /machine:x64 /debug /INCREMENTAL /subsystem:console  kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib 
<<

# Rule to build all files generated by this target.
CMakeFiles\mlp_library.dir\build: mlp_library.exe

.PHONY : CMakeFiles\mlp_library.dir\build

CMakeFiles\mlp_library.dir\clean:
	$(CMAKE_COMMAND) -P CMakeFiles\mlp_library.dir\cmake_clean.cmake
.PHONY : CMakeFiles\mlp_library.dir\clean

CMakeFiles\mlp_library.dir\depend:
	$(CMAKE_COMMAND) -E cmake_depends "NMake Makefiles" C:\Users\Enzo\Documents\Github\Country_guesser\mlp_library C:\Users\Enzo\Documents\Github\Country_guesser\mlp_library C:\Users\Enzo\Documents\Github\Country_guesser\mlp_library\cmake-build-debug C:\Users\Enzo\Documents\Github\Country_guesser\mlp_library\cmake-build-debug C:\Users\Enzo\Documents\Github\Country_guesser\mlp_library\cmake-build-debug\CMakeFiles\mlp_library.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles\mlp_library.dir\depend

