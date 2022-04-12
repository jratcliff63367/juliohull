JulioHull was written by <jerezjulio0@gmail.com>

Julio is best known for creating the Newton Physics Engine : http://newtondynamics.com/forum/newton.php

This code was packaged up in this form by John W. Ratcliff (jratcliffscarab@gmail.com)

JulioHull is extremely fast, robust, and has high numerical precision.

JulioHull is delivered as a header file only library

Here is how you use it:

In one of your CPP files add the line: #define ENABLE_JULIO_HULL_IMPLEMENTATION 1

and then include "JulioHull.h"

To build the test application for windows go to the app directory and type:

cmake CMakeLists.txt

This will produce a solution file called 'TestHull.sln' that you can load, build, and run.

To build the test application for linux go to the app directory and type:

cmake CMakeLists.txt
cmake --build .

Much thanks for Julio for making this library publicly available!
