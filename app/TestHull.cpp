#include <stdio.h>

#define TEST_FOR_MEMORY_LEAKS 0 // set to 1, on Windows only, to enable memory leak checking on application exit
#if TEST_FOR_MEMORY_LEAKS
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#else
#include <stdlib.h>
#endif

#include <string.h>
#include <stdint.h>

#define ENABLE_JULIO_HULL_IMPLEMENTATION 1
#include "JulioHull.h"
#include "wavefront.h"
#include "ScopedTime.h"

#include <thread>
#include <string>
#include <vector>

#ifdef _MSC_VER
#pragma warning(disable:4100 4996)
#include <conio.h>
#endif

// Evaluates if this is true or false, returns true if it 
// could be evaluated. Stores the result into 'value'

int main(int argc,const char **argv)
{
#if TEST_FOR_MEMORY_LEAKS
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
#endif
	if ( argc < 2 )
	{
		printf("Usage: TestVHACD <wavefront.obj> (options)\n");
		printf("\n");
		printf("-v <maxHullVertCount>   : Maximum number of vertices in the output convex hull. Default value is 64\n");
	}
	else
	{
		uint32_t maxHullVertexCount = 64;

		const char *inputFile = argv[1];

		WavefrontObj w;
		uint32_t tcount = w.loadObj(inputFile);
		if ( tcount == 0 )
		{
			printf("Failed to load valid mesh from wavefront OBJ file:%s\n", inputFile);
		}
		else
		{
			for (int i=2; i<argc; i+=2)
			{
				const char *option = argv[i];
				const char *value = argv[i+1];
				if ( strcmp(option,"-v") == 0 )
				{
					int32_t r = atoi(value);
					if ( r >= 8 && r <= 2048 )
					{
						printf("Maximum hull vertices set to:%d\n", r);
						maxHullVertexCount = uint32_t(r);
					}
					else
					{
						printf("Invalid maximum hull vertices, must be between 8 and 20484\n");
					}
				}
			}

			juliohull::JulioHull *jh = juliohull::JulioHull::create();
			double *points = new double[w.mVertexCount*3];
			for (uint32_t i=0; i<w.mVertexCount*3; i++)
			{
				points[i] = w.mVertices[i];
			}
			printf("Computing convex hull\n");
			uint32_t tcount = jh->computeConvexHull(w.mVertexCount,points,maxHullVertexCount);
			if ( tcount )
			{
				FILE *fph = fopen("hull.obj", "wb");
				if ( fph )
				{
					printf("Saving ConvexHull with %d triangles to 'hull.obj'\n", tcount);
					uint32_t baseIndex = 1;
					uint32_t vcount;
					const double *vertices = jh->getVertices(vcount);
					const uint32_t *indices = jh->getIndices(tcount);
					for (uint32_t j=0; j<vcount; j++)
					{
						const double *pos = &vertices[j*3];
						fprintf(fph,"v %0.9f %0.9f %0.9f\n", pos[0], pos[1], pos[2]);
					}
					for (uint32_t j=0; j<tcount; j++)
					{
						uint32_t i1 = indices[j*3+0]+baseIndex;
						uint32_t i2 = indices[j*3+1]+baseIndex;
						uint32_t i3 = indices[j*3+2]+baseIndex;
						fprintf(fph,"f %d %d %d\n", i1, i2, i3);
					}
					fclose(fph);
				}
			}
			else
			{
				printf("Failed to create a convex hull for the input mesh.\n");
			}

			delete []points;
			jh->release();


		}
	}
	return 0;
}
