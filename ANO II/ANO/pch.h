// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file

#ifndef PCH_H
#define PCH_H

// TODO: add headers that you want to pre-compile here
#include "targetver.h"

#include <stdio.h>

#include <opencv2/opencv.hpp>

#include <stdlib.h>
#include <string>
#include "pch.h"
#include <iostream>

#include <time.h>
#include <math.h>
#include <string.h>
#include <vector>


#define LAMBDA 1.0
#define ETA 0.1

#define SQR( x ) ( ( x ) * ( x ) )

#endif //PCH_H
