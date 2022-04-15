/* Copyright (c) 2022 Julio Jerez (jerezjulio0@gmail.com)
 All rights reserved.
 
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 3. The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once

#include <stdint.h>

// JulioHull was written by <jerezjulio0@gmail.com>
//
// Julio is best known for creating the Newton Physics Engine : http://newtondynamics.com/forum/newton.php
//
// This code was packaged up in this form by John W. Ratcliff (jratcliffscarab@gmail.com)
//
// JulioHull is extremely fast, robust, and has high numerical precision.
//
// JulioHull is delivered as a header file only library
//
// Here is how you use it:
//
// In one of your CPP files add the line: #define ENABLE_JULIO_HULL_IMPLEMENTATION 1
//
// and then include "JulioHull.h"

// Prototype for the JulioHull class
namespace juliohull
{
class JulioHull
{
public:
	/**
	* Create an instance of the JulioHull class
	* 
	* @return : Returns a pointer to an instance of the JulioHull class, call 'release' to destruct it.
	*/
	static JulioHull *create(void);

	/**
	* Computes a convex hull around the provided vertices. Returns the number of triangles produced.
	* 
	* @param vertexCount : Number of input vertices
	* @param vertices : The vertices in double format (x1,y1,z1,  x2,y2,z2, ..) form.
	* @param maxHullVertices : The maximum number of vertices allowed in the convex hull
	* @param distanceTolerance : A tolerance value to consider two vertices the same
	* 
	* @return : Returns the number of triangles produced, if zero no hull created.
	*/
	virtual uint32_t computeConvexHull(uint32_t vertexCount,
								       const double *vertices,
									   uint32_t maxHullVertices,
									   double distanceTolerance=0.0001) = 0;

	/**
	* Returns the convex hull vertices and number of vertices
	* 
	* @param vcount : A reference to return number of vertices in the convex hull
	* 
	* @return : A pointer to the set of vertices
	*/
	virtual const double *getVertices(uint32_t &vcount) = 0;

	/***
	* Returns the convex hull triangle indices and triangle count
	* 
	* @param tcount  : A reference to return number of triangles in the convex hull
	* 
	* @return : A pointer to the triangle indices; 3 per triangle
	*/
	virtual const uint32_t *getIndices(uint32_t &tcount) = 0;

	/**
	* Release this instance of JulioJull
	*/
	virtual void release(void) = 0;

protected:
	virtual ~JulioHull(void)
	{
	}
};

} // end of juliohull namespace



#if ENABLE_JULIO_HULL_IMPLEMENTATION
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <limits.h>

#include <chrono>
#include <iostream>
#include <vector>
#include <queue>
#include <mutex>
#include <atomic>
#include <thread>
#include <algorithm>
#include <condition_variable>
#include <unordered_set>
#include <unordered_map>


#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4100 4189 4456 4701 4702 4127 4996)
#endif

namespace juliohull
{

//***********************************************************************************************
// ConvexHull generation code by Julio Jerez <jerezjulio0@gmail.com>
//***********************************************************************************************

namespace nd 
{
	namespace juliohull 
	{
		//!    Vector dim 3.
		template <typename T>
		class Vect3 {
		public:
			T& operator[](size_t i) { return m_data[i]; }
			const T& operator[](size_t i) const { return m_data[i]; }
			T& getX();
			T& getY();
			T& getZ();
			const T& getX() const;
			const T& getY() const;
			const T& getZ() const;
			void Normalize();
			T GetNorm() const;
			void operator=(const Vect3& rhs);
			void operator+=(const Vect3& rhs);
			void operator-=(const Vect3& rhs);
			void operator-=(T a);
			void operator+=(T a);
			void operator/=(T a);
			void operator*=(T a);
			Vect3 operator^(const Vect3& rhs) const;
			T operator*(const Vect3& rhs) const;
			Vect3 operator+(const Vect3& rhs) const;
			Vect3 operator-(const Vect3& rhs) const;
			Vect3 operator-() const;
			Vect3 operator*(T rhs) const;
			Vect3 operator/(T rhs) const;
			bool operator<(const Vect3& rhs) const;
			bool operator>(const Vect3& rhs) const;
			Vect3();
			Vect3(T a);
			Vect3(T x, T y, T z);
			Vect3(const Vect3& rhs);
			/*virtual*/ ~Vect3(void);

			// Compute the center of this bounding box and return the diagonal length
			T GetCenter(const Vect3 &bmin, const Vect3 &bmax)
			{
				getX() = (bmin.getX() + bmax.getX())*0.5;
				getY() = (bmin.getY() + bmax.getY())*0.5;
				getZ() = (bmin.getZ() + bmax.getZ())*0.5;
				T dx = bmax.getX() - bmin.getX();
				T dy = bmax.getY() - bmin.getY();
				T dz = bmax.getZ() - bmin.getZ();
				T diagonal = T(sqrt(dx*dx + dy*dy + dz*dz));
				return diagonal;
			}

			// Update the min/max values relative to this point
			void UpdateMinMax(Vect3 &bmin, Vect3 &bmax) const
			{
				if (getX() < bmin.getX())
				{
					bmin.getX() = getX();
				}
				if (getY() < bmin.getY())
				{
					bmin.getY() = getY();
				}
				if (getZ() < bmin.getZ())
				{
					bmin.getZ() = getZ();
				}
				if (getX() > bmax.getX())
				{
					bmax.getX() = getX();
				}
				if (getX() > bmax.getX())
				{
					bmax.getX() = getX();
				}
				if (getY() > bmax.getY())
				{
					bmax.getY() = getY();
				}
				if (getZ() > bmax.getZ())
				{
					bmax.getZ() = getZ();
				}
			}

			// Returns the squared distance between these two points
			T GetDistanceSquared(const Vect3 &p) const
			{
				T dx = getX() - p.getX();
				T dy = getY() - p.getY();
				T dz = getZ() - p.getZ();
				return dx*dx + dy*dy + dz*dz;
			}

			T GetDistance(const Vect3 &p) const
			{
				return sqrt(GetDistanceSquared(p));
			}

			// Returns the raw vector data as a pointer
			T* GetData(void)
			{
				return m_data;
			}
		private:
			T m_data[3];
		};
		//!    Vector dim 2.
		template <typename T>
		class Vec2 {
		public:
			T& operator[](size_t i) { return m_data[i]; }
			const T& operator[](size_t i) const { return m_data[i]; }
			T& getX();
			T& getY();
			const T& getX() const;
			const T& getY() const;
			void Normalize();
			T GetNorm() const;
			void operator=(const Vec2& rhs);
			void operator+=(const Vec2& rhs);
			void operator-=(const Vec2& rhs);
			void operator-=(T a);
			void operator+=(T a);
			void operator/=(T a);
			void operator*=(T a);
			T operator^(const Vec2& rhs) const;
			T operator*(const Vec2& rhs) const;
			Vec2 operator+(const Vec2& rhs) const;
			Vec2 operator-(const Vec2& rhs) const;
			Vec2 operator-() const;
			Vec2 operator*(T rhs) const;
			Vec2 operator/(T rhs) const;
			Vec2();
			Vec2(T a);
			Vec2(T x, T y);
			Vec2(const Vec2& rhs);
			/*virtual*/ ~Vec2(void);

		private:
			T m_data[2];
		};

		template <typename T>
		const bool Colinear(const Vect3<T>& a, const Vect3<T>& b, const Vect3<T>& c);
		template <typename T>
		const T ComputeVolume4(const Vect3<T>& a, const Vect3<T>& b, const Vect3<T>& c, const Vect3<T>& d);
	}
}

namespace nd
{
	namespace juliohull
	{
		template <typename T>
		inline Vect3<T> operator*(T lhs, const Vect3<T> & rhs)
		{
			return Vect3<T>(lhs * rhs.getX(), lhs * rhs.getY(), lhs * rhs.getZ());
		}
		template <typename T>
		inline T & Vect3<T>::getX()
		{
			return m_data[0];
		}
		template <typename T>
		inline  T &    Vect3<T>::getY()
		{
			return m_data[1];
		}
		template <typename T>
		inline  T &    Vect3<T>::getZ()
		{
			return m_data[2];
		}
		template <typename T>
		inline  const T & Vect3<T>::getX() const
		{
			return m_data[0];
		}
		template <typename T>
		inline  const T & Vect3<T>::getY() const
		{
			return m_data[1];
		}
		template <typename T>
		inline  const T & Vect3<T>::getZ() const
		{
			return m_data[2];
		}
		template <typename T>
		inline  void Vect3<T>::Normalize()
		{
			T n = sqrt(m_data[0] * m_data[0] + m_data[1] * m_data[1] + m_data[2] * m_data[2]);
			if (n != 0.0) (*this) /= n;
		}
		template <typename T>
		inline  T Vect3<T>::GetNorm() const
		{
			return sqrt(m_data[0] * m_data[0] + m_data[1] * m_data[1] + m_data[2] * m_data[2]);
		}
		template <typename T>
		inline  void Vect3<T>::operator= (const Vect3 & rhs)
		{
			this->m_data[0] = rhs.m_data[0];
			this->m_data[1] = rhs.m_data[1];
			this->m_data[2] = rhs.m_data[2];
		}
		template <typename T>
		inline  void Vect3<T>::operator+=(const Vect3 & rhs)
		{
			this->m_data[0] += rhs.m_data[0];
			this->m_data[1] += rhs.m_data[1];
			this->m_data[2] += rhs.m_data[2];
		}
		template <typename T>
		inline void Vect3<T>::operator-=(const Vect3 & rhs)
		{
			this->m_data[0] -= rhs.m_data[0];
			this->m_data[1] -= rhs.m_data[1];
			this->m_data[2] -= rhs.m_data[2];
		}
		template <typename T>
		inline void Vect3<T>::operator-=(T a)
		{
			this->m_data[0] -= a;
			this->m_data[1] -= a;
			this->m_data[2] -= a;
		}
		template <typename T>
		inline void Vect3<T>::operator+=(T a)
		{
			this->m_data[0] += a;
			this->m_data[1] += a;
			this->m_data[2] += a;
		}
		template <typename T>
		inline void Vect3<T>::operator/=(T a)
		{
			this->m_data[0] /= a;
			this->m_data[1] /= a;
			this->m_data[2] /= a;
		}
		template <typename T>
		inline void Vect3<T>::operator*=(T a)
		{
			this->m_data[0] *= a;
			this->m_data[1] *= a;
			this->m_data[2] *= a;
		}
		template <typename T>
		inline Vect3<T> Vect3<T>::operator^ (const Vect3<T> & rhs) const
		{
			return Vect3<T>(m_data[1] * rhs.m_data[2] - m_data[2] * rhs.m_data[1],
				m_data[2] * rhs.m_data[0] - m_data[0] * rhs.m_data[2],
				m_data[0] * rhs.m_data[1] - m_data[1] * rhs.m_data[0]);
		}
		template <typename T>
		inline T Vect3<T>::operator*(const Vect3<T> & rhs) const
		{
			return (m_data[0] * rhs.m_data[0] + m_data[1] * rhs.m_data[1] + m_data[2] * rhs.m_data[2]);
		}
		template <typename T>
		inline Vect3<T> Vect3<T>::operator+(const Vect3<T> & rhs) const
		{
			return Vect3<T>(m_data[0] + rhs.m_data[0], m_data[1] + rhs.m_data[1], m_data[2] + rhs.m_data[2]);
		}
		template <typename T>
		inline  Vect3<T> Vect3<T>::operator-(const Vect3<T> & rhs) const
		{
			return Vect3<T>(m_data[0] - rhs.m_data[0], m_data[1] - rhs.m_data[1], m_data[2] - rhs.m_data[2]);
		}
		template <typename T>
		inline  Vect3<T> Vect3<T>::operator-() const
		{
			return Vect3<T>(-m_data[0], -m_data[1], -m_data[2]);
		}

		template <typename T>
		inline Vect3<T> Vect3<T>::operator*(T rhs) const
		{
			return Vect3<T>(rhs * this->m_data[0], rhs * this->m_data[1], rhs * this->m_data[2]);
		}
		template <typename T>
		inline Vect3<T> Vect3<T>::operator/ (T rhs) const
		{
			return Vect3<T>(m_data[0] / rhs, m_data[1] / rhs, m_data[2] / rhs);
		}
		template <typename T>
		inline Vect3<T>::Vect3(T a)
		{
			m_data[0] = m_data[1] = m_data[2] = a;
		}
		template <typename T>
		inline Vect3<T>::Vect3(T x, T y, T z)
		{
			m_data[0] = x;
			m_data[1] = y;
			m_data[2] = z;
		}
		template <typename T>
		inline Vect3<T>::Vect3(const Vect3 & rhs)
		{
			m_data[0] = rhs.m_data[0];
			m_data[1] = rhs.m_data[1];
			m_data[2] = rhs.m_data[2];
		}
		template <typename T>
		inline Vect3<T>::~Vect3(void) {};

		template <typename T>
		inline Vect3<T>::Vect3() {}

		template<typename T>
		inline const bool Colinear(const Vect3<T> & a, const Vect3<T> & b, const Vect3<T> & c)
		{
			return  ((c.getZ() - a.getZ()) * (b.getY() - a.getY()) - (b.getZ() - a.getZ()) * (c.getY() - a.getY()) == 0.0 /*EPS*/) &&
				((b.getZ() - a.getZ()) * (c.getX() - a.getX()) - (b.getX() - a.getX()) * (c.getZ() - a.getZ()) == 0.0 /*EPS*/) &&
				((b.getX() - a.getX()) * (c.getY() - a.getY()) - (b.getY() - a.getY()) * (c.getX() - a.getX()) == 0.0 /*EPS*/);
		}

		template<typename T>
		inline const T ComputeVolume4(const Vect3<T> & a, const Vect3<T> & b, const Vect3<T> & c, const Vect3<T> & d)
		{
			return (a - d) * ((b - d) ^ (c - d));
		}

		template <typename T>
		inline bool Vect3<T>::operator<(const Vect3 & rhs) const
		{
			if (getX() == rhs[0])
			{
				if (getY() == rhs[1])
				{
					return (getZ() < rhs[2]);
				}
				return (getY() < rhs[1]);
			}
			return (getX() < rhs[0]);
		}
		template <typename T>
		inline  bool Vect3<T>::operator>(const Vect3 & rhs) const
		{
			if (getX() == rhs[0])
			{
				if (getY() == rhs[1])
				{
					return (getZ() > rhs[2]);
				}
				return (getY() > rhs[1]);
			}
			return (getX() > rhs[0]);
		}
		template <typename T>
		inline Vec2<T> operator*(T lhs, const Vec2<T> & rhs)
		{
			return Vec2<T>(lhs * rhs.getX(), lhs * rhs.getY());
		}
		template <typename T>
		inline T & Vec2<T>::getX()
		{
			return m_data[0];
		}
		template <typename T>
		inline  T &    Vec2<T>::getY()
		{
			return m_data[1];
		}
		template <typename T>
		inline  const T & Vec2<T>::getX() const
		{
			return m_data[0];
		}
		template <typename T>
		inline  const T & Vec2<T>::getY() const
		{
			return m_data[1];
		}
		template <typename T>
		inline  void Vec2<T>::Normalize()
		{
			T n = sqrt(m_data[0] * m_data[0] + m_data[1] * m_data[1]);
			if (n != 0.0) (*this) /= n;
		}
		template <typename T>
		inline  T Vec2<T>::GetNorm() const
		{
			return sqrt(m_data[0] * m_data[0] + m_data[1] * m_data[1]);
		}
		template <typename T>
		inline  void Vec2<T>::operator= (const Vec2 & rhs)
		{
			this->m_data[0] = rhs.m_data[0];
			this->m_data[1] = rhs.m_data[1];
		}
		template <typename T>
		inline  void Vec2<T>::operator+=(const Vec2 & rhs)
		{
			this->m_data[0] += rhs.m_data[0];
			this->m_data[1] += rhs.m_data[1];
		}
		template <typename T>
		inline void Vec2<T>::operator-=(const Vec2 & rhs)
		{
			this->m_data[0] -= rhs.m_data[0];
			this->m_data[1] -= rhs.m_data[1];
		}
		template <typename T>
		inline void Vec2<T>::operator-=(T a)
		{
			this->m_data[0] -= a;
			this->m_data[1] -= a;
		}
		template <typename T>
		inline void Vec2<T>::operator+=(T a)
		{
			this->m_data[0] += a;
			this->m_data[1] += a;
		}
		template <typename T>
		inline void Vec2<T>::operator/=(T a)
		{
			this->m_data[0] /= a;
			this->m_data[1] /= a;
		}
		template <typename T>
		inline void Vec2<T>::operator*=(T a)
		{
			this->m_data[0] *= a;
			this->m_data[1] *= a;
		}
		template <typename T>
		inline T Vec2<T>::operator^ (const Vec2<T> & rhs) const
		{
			return m_data[0] * rhs.m_data[1] - m_data[1] * rhs.m_data[0];
		}
		template <typename T>
		inline T Vec2<T>::operator*(const Vec2<T> & rhs) const
		{
			return (m_data[0] * rhs.m_data[0] + m_data[1] * rhs.m_data[1]);
		}
		template <typename T>
		inline Vec2<T> Vec2<T>::operator+(const Vec2<T> & rhs) const
		{
			return Vec2<T>(m_data[0] + rhs.m_data[0], m_data[1] + rhs.m_data[1]);
		}
		template <typename T>
		inline  Vec2<T> Vec2<T>::operator-(const Vec2<T> & rhs) const
		{
			return Vec2<T>(m_data[0] - rhs.m_data[0], m_data[1] - rhs.m_data[1]);
		}
		template <typename T>
		inline  Vec2<T> Vec2<T>::operator-() const
		{
			return Vec2<T>(-m_data[0], -m_data[1]);
		}

		template <typename T>
		inline Vec2<T> Vec2<T>::operator*(T rhs) const
		{
			return Vec2<T>(rhs * this->m_data[0], rhs * this->m_data[1]);
		}
		template <typename T>
		inline Vec2<T> Vec2<T>::operator/ (T rhs) const
		{
			return Vec2<T>(m_data[0] / rhs, m_data[1] / rhs);
		}
		template <typename T>
		inline Vec2<T>::Vec2(T a)
		{
			m_data[0] = m_data[1] = a;
		}
		template <typename T>
		inline Vec2<T>::Vec2(T x, T y)
		{
			m_data[0] = x;
			m_data[1] = y;
		}
		template <typename T>
		inline Vec2<T>::Vec2(const Vec2 & rhs)
		{
			m_data[0] = rhs.m_data[0];
			m_data[1] = rhs.m_data[1];
		}
		template <typename T>
		inline Vec2<T>::~Vec2(void) {};

		template <typename T>
		inline Vec2<T>::Vec2() {}

		/*
		  InsideTriangle decides if a point P is Inside of the triangle
		  defined by A, B, C.
		*/
		template<typename T>
		inline const bool InsideTriangle(const Vec2<T> & a, const Vec2<T> & b, const Vec2<T> & c, const Vec2<T> & p)
		{
			T ax, ay, bx, by, cx, cy, apx, apy, bpx, bpy, cpx, cpy;
			T cCROSSap, bCROSScp, aCROSSbp;
			ax = c.getX() - b.getX();  ay = c.getY() - b.getY();
			bx = a.getX() - c.getX();  by = a.getY() - c.getY();
			cx = b.getX() - a.getX();  cy = b.getY() - a.getY();
			apx = p.getX() - a.getX();  apy = p.getY() - a.getY();
			bpx = p.getX() - b.getX();  bpy = p.getY() - b.getY();
			cpx = p.getX() - c.getX();  cpy = p.getY() - c.getY();
			aCROSSbp = ax*bpy - ay*bpx;
			cCROSSap = cx*apy - cy*apx;
			bCROSScp = bx*cpy - by*cpx;
			return ((aCROSSbp >= 0.0) && (bCROSScp >= 0.0) && (cCROSSap >= 0.0));
		}
	}
}


namespace nd
{
	namespace juliohull 
	{
		#define VHACD_GOOGOL_SIZE		4

		class Googol;
		Googol Determinant2x2(const Googol matrix[2][2]);
		Googol Determinant3x3(const Googol matrix[3][3]);
		double Determinant2x2(const double matrix[2][2], double* const error);
		double Determinant3x3(const double matrix[3][3], double* const error);

		inline int dExp2(int x)
		{
			int exp;
			for (exp = -1; x; x >>= 1)
			{
				exp++;
			}
			return exp;
		}

		inline int dBitReversal(int v, int base)
		{
			int x = 0;
			int power = dExp2(base) - 1;
			do
			{
				x += (v & 1) << power;
				v >>= 1;
				power--;
			} while (v);
			return x;
		}

		template<class T>
		class List
		{
			public:
			class ndNode
			{
				ndNode(ndNode* const prev, ndNode* const next)
					:m_info()
					, m_next(next)
					, m_prev(prev)
				{
					if (m_prev)
					{
						m_prev->m_next = this;
					}
					if (m_next)
					{
						m_next->m_prev = this;
					}
				}

				ndNode(const T &info, ndNode* const prev, ndNode* const next)
					:m_info(info)
					,m_next(next)
					,m_prev(prev)
				{
					if (m_prev)
					{
						m_prev->m_next = this;
					}
					if (m_next)
					{
						m_next->m_prev = this;
					}
				}

				~ndNode()
				{
				}

				void Unlink()
				{
					if (m_prev)
					{
						m_prev->m_next = m_next;
					}
					if (m_next)
					{
						m_next->m_prev = m_prev;
					}
					m_prev = nullptr;
					m_next = nullptr;
				}

				void AddLast(ndNode* const node)
				{
					m_next = node;
					node->m_prev = this;
				}

				void AddFirst(ndNode* const node)
				{
					m_prev = node;
					node->m_next = this;
				}

				public:
				T& GetInfo()
				{
					return m_info;
				}

				ndNode *GetNext() const
				{
					return m_next;
				}

				ndNode *GetPrev() const
				{
					return m_prev;
				}

				private:
				T m_info;
				ndNode *m_next;
				ndNode *m_prev;
				friend class List<T>;
			};

			public:
			List()
				:m_first(nullptr)
				, m_last(nullptr)
				, m_count(0)
			{
			}

			~List()
			{
				RemoveAll();
			}

			void RemoveAll()
			{
				for (ndNode *node = m_first; node; node = m_first)
				{
					m_count--;
					m_first = node->GetNext();
					node->Unlink();
					delete node;
				}
				m_last = nullptr;
				m_first = nullptr;
			}

			ndNode* Append()
			{
				m_count++;
				if (m_first == nullptr)
				{
					m_first = new ndNode(nullptr, nullptr);
					m_last = m_first;
				}
				else
				{
					m_last = new ndNode(m_last, nullptr);
				}
				return m_last;
			}

			ndNode* Append(const T &element)
			{
				m_count++;
				if (m_first == nullptr)
				{
					m_first = new ndNode(element, nullptr, nullptr);
					m_last = m_first;
				}
				else
				{
					m_last = new ndNode(element, m_last, nullptr);
				}
				return m_last;
			}

			ndNode* Addtop(const T &element)
			{
				m_count++;
				if (m_last == nullptr)
				{
					m_last = new ndNode(element, nullptr, nullptr);
					m_first = m_last;
				}
				else
				{
					m_first = new ndNode(element, nullptr, m_first);
				}
				return m_first;
			}

			int GetCount() const
			{
				return m_count;
			}

			//operator int() const;

			ndNode* GetLast() const
			{
				return m_last;
			}

			ndNode* GetFirst() const
			{
				return m_first;
			}

			void Remove(ndNode* const node)
			{
				Unlink(node);
				delete node;
			}

			void Unlink(ndNode* const node)
			{
				m_count--;
				if (node == m_first)
				{
					m_first = m_first->GetNext();
				}
				if (node == m_last)
				{
					m_last = m_last->GetPrev();
				}
				node->Unlink();
			}

			void Remove(const T &element)
			{
				ndNode *const node = Find(element);
				if (node)
				{
					Remove(node);
				}
			}

			ndNode* Find(const T &element) const
			{
				ndNode *node;
				for (node = m_first; node; node = node->GetNext())
				{
					if (element == node->m_info)
					{
						break;
					}
				}
				return node;
			}

			private:
			ndNode* m_first;
			ndNode* m_last;
			int m_count;
			friend class ndNode;
		};

		class hullVector : public juliohull::Vect3<double>
		{
			public:
			hullVector()
				:Vect3<double>(0, 0, 0)
			{
			}

			hullVector(double x)
				:Vect3<double>(x, x, x)
			{
			}

			hullVector(const hullVector& x)
				:Vect3<double>(x.getX(), x.getY(), x.getZ())
			{
			}

			hullVector(double x, double y, double z, double)
				:Vect3<double>(x, y, z)
			{
			}

			hullVector GetMin(const hullVector& p) const
			{
				return hullVector(
					getX() < p.getX() ? getX() : p.getX(),
					getY() < p.getY() ? getY() : p.getY(),
					getZ() < p.getZ() ? getZ() : p.getZ(), 0.0);
			}

			hullVector GetMax(const hullVector& p) const
			{
				return hullVector(
					getX() > p.getX() ? getX() : p.getX(),
					getY() > p.getY() ? getY() : p.getY(),
					getZ() > p.getZ() ? getZ() : p.getZ(), 0.0);
			}

			hullVector Scale(double s) const
			{
				return hullVector(getX() * s, getY() * s, getZ() * s, 0.0);
			}

			inline hullVector operator+(const hullVector & rhs) const
			{
				return hullVector(getX() + rhs.getX(), getY() + rhs.getY(), getZ() + rhs.getZ(), 0.0f);
			}

			inline hullVector operator-(const hullVector & rhs) const
			{
				return hullVector(getX() - rhs.getX(), getY() - rhs.getY(), getZ() - rhs.getZ(), 0.0f);
			}

			inline hullVector operator*(const hullVector & rhs) const
			{
				return hullVector(getX() * rhs.getX(), getY() * rhs.getY(), getZ() * rhs.getZ(), 0.0f);
			}

			inline double DotProduct(const hullVector & rhs) const
			{
				return getX() * rhs.getX() + getY() * rhs.getY() + getZ() * rhs.getZ();
			}

			inline hullVector CrossProduct(const hullVector & rhs) const
			{
				return hullVector(getY() * rhs.getZ() - getZ() * rhs.getY(), getZ() * rhs.getX() - getX() * rhs.getZ(), getX() * rhs.getY() - getY() * rhs.getX(), 0.0);
			}

			inline hullVector operator= (const Vect3 & rhs)
			{
				getX() = rhs.getX();
				getY() = rhs.getY();
				getZ() = rhs.getZ();
				return *this;
			}
		};

		class hullPlane : public hullVector
		{
			public:
			hullPlane(double x, double y, double z, double w)
				:hullVector(x, y, z, 0.0)
				, m_w(w)
			{
			}

			hullPlane(const hullVector &P0, const hullVector &P1, const hullVector &P2)
				:hullVector((P1 - P0).CrossProduct(P2 - P0))
			{
				m_w = -DotProduct(P0);
			}

			hullPlane Scale(double s) const
			{
				return hullPlane(getX() * s, getY() * s, getZ() * s, m_w * s);
			}

			inline hullPlane operator= (const hullPlane &rhs)
			{
				getX() = rhs.getX();
				getY() = rhs.getY();
				getZ() = rhs.getZ();
				m_w = rhs.m_w;
				return *this;
			}

			inline hullVector operator*(const hullVector & rhs) const
			{
				return hullVector(getX() * rhs.getX(), getY() * rhs.getY(), getZ() * rhs.getZ(), 0.0f);
			}

			double Evalue(const hullVector &point) const
			{
				return DotProduct(point) + m_w;
			}

			double m_w;
		};

		class Googol
		{
			public:
			Googol(void);
			Googol(double value);

			operator double() const;
			Googol operator+ (const Googol &A) const;
			Googol operator- (const Googol &A) const;
			Googol operator* (const Googol &A) const;
			Googol operator/ (const Googol &A) const;

			Googol operator+= (const Googol &A);
			Googol operator-= (const Googol &A);

			bool operator> (const Googol &A) const;
			bool operator>= (const Googol &A) const;
			bool operator< (const Googol &A) const;
			bool operator<= (const Googol &A) const;
			bool operator== (const Googol &A) const;
			bool operator!= (const Googol &A) const;

			Googol Abs() const;
			Googol Sqrt() const;
			Googol InvSqrt() const;
			Googol Floor() const;

			void Trace() const;
			void ToString(char* const string) const;

			private:
			void InitFloatFloat(double value);
			void NegateMantissa(uint64_t* const mantissa) const;
			void CopySignedMantissa(uint64_t* const mantissa) const;
			int NormalizeMantissa(uint64_t* const mantissa) const;
			uint64_t CheckCarrier(uint64_t a, uint64_t b) const;
			void ShiftRightMantissa(uint64_t* const mantissa, int bits) const;

			int LeadingZeros(uint64_t a) const;
			void ExtendeMultiply(uint64_t a, uint64_t b, uint64_t& high, uint64_t& low) const;
			void ScaleMantissa(uint64_t* const out, uint64_t scale) const;

			int m_sign;
			int m_exponent;
			uint64_t m_mantissa[VHACD_GOOGOL_SIZE];

			public:
			static Googol m_zero;
			static Googol m_one;
			static Googol m_two;
			static Googol m_three;
			static Googol m_half;
		};

		template <class T>
		inline T Max(T A, T B)
		{
			return (A > B) ? A : B;
		}

		template <class T>
		inline void Swap(T& A, T& B)
		{
			T tmp(A);
			A = B;
			B = tmp;
		}

		template <class T, class dCompareKey>
		void Sort(T* const array, int elements)
		{
			const int batchSize = 8;
			int stack[1024][2];

			stack[0][0] = 0;
			stack[0][1] = elements - 1;
			int stackIndex = 1;
			const dCompareKey comparator;
			while (stackIndex)
			{
				stackIndex--;
				int lo = stack[stackIndex][0];
				int hi = stack[stackIndex][1];
				if ((hi - lo) > batchSize)
				{
					int mid = (lo + hi) >> 1;
					if (comparator.Compare(array[lo], array[mid]) > 0)
					{
						Swap(array[lo], array[mid]);
					}
					if (comparator.Compare(array[mid], array[hi]) > 0)
					{
						Swap(array[mid], array[hi]);
					}
					if (comparator.Compare(array[lo], array[mid]) > 0)
					{
						Swap(array[lo], array[mid]);
					}
					int i = lo + 1;
					int j = hi - 1;
					const T pivot(array[mid]);
					do
					{
						while (comparator.Compare(array[i], pivot) < 0)
						{
							i++;
						}
						while (comparator.Compare(array[j], pivot) > 0)
						{
							j--;
						}

						if (i <= j)
						{
							Swap(array[i], array[j]);
							i++;
							j--;
						}
					} while (i <= j);

					if (i < hi)
					{
						stack[stackIndex][0] = i;
						stack[stackIndex][1] = hi;
						stackIndex++;
					}
					if (lo < j)
					{
						stack[stackIndex][0] = lo;
						stack[stackIndex][1] = j;
						stackIndex++;
					}
					assert(stackIndex < int(sizeof(stack) / (2 * sizeof(stack[0][0]))));
				}
			}

			int stride = batchSize + 1;
			if (elements < stride)
			{
				stride = elements;
			}
			for (int i = 1; i < stride; ++i)
			{
				if (comparator.Compare(array[0], array[i]) > 0)
				{
					Swap(array[0], array[i]);
				}
			}

			for (int i = 1; i < elements; ++i)
			{
				int j = i;
				const T tmp(array[i]);
				for (; comparator.Compare(array[j - 1], tmp) > 0; --j)
				{
					assert(j > 0);
					array[j] = array[j - 1];
				}
				array[j] = tmp;
			}
		}
	}
}


namespace nd
{
	namespace juliohull
	{
		#define Absolute(a)  ((a) >= 0.0 ? (a) : -(a))

		Googol Googol::m_zero(0.0);
		Googol Googol::m_one(1.0);
		Googol Googol::m_two(2.0);
		Googol Googol::m_three(3.0);
		Googol Googol::m_half(0.5);

		Googol::Googol(void)
			:m_sign(0)
			,m_exponent(0)
		{
			memset(m_mantissa, 0, sizeof(m_mantissa));
		}

		Googol::Googol(double value)
			:m_sign(0)
			, m_exponent(0)
		{
			int exp;
			double mantissa = fabs(frexp(value, &exp));

			m_exponent = int(exp);
			m_sign = (value >= 0) ? 0 : 1;

			memset(m_mantissa, 0, sizeof(m_mantissa));
			m_mantissa[0] = uint64_t(double(uint64_t(1) << 62) * mantissa);
		}

		void Googol::CopySignedMantissa(uint64_t* const mantissa) const
		{
			memcpy(mantissa, m_mantissa, sizeof(m_mantissa));
			if (m_sign)
			{
				NegateMantissa(mantissa);
			}
		}

		Googol::operator double() const
		{
			double mantissa = (double(1.0f) / double(uint64_t(1) << 62)) * double(m_mantissa[0]);
			mantissa = ldexp(mantissa, m_exponent) * (m_sign ? double(-1.0f) : double(1.0f));
			return mantissa;
		}

		Googol Googol::operator+ (const Googol &A) const
		{
			Googol tmp;
			if (m_mantissa[0] && A.m_mantissa[0])
			{
				uint64_t mantissa0[VHACD_GOOGOL_SIZE];
				uint64_t mantissa1[VHACD_GOOGOL_SIZE];
				uint64_t mantissa[VHACD_GOOGOL_SIZE];

				CopySignedMantissa(mantissa0);
				A.CopySignedMantissa(mantissa1);

				int exponetDiff = m_exponent - A.m_exponent;
				int exponent = m_exponent;
				if (exponetDiff > 0)
				{
					ShiftRightMantissa(mantissa1, exponetDiff);
				}
				else if (exponetDiff < 0)
				{
					exponent = A.m_exponent;
					ShiftRightMantissa(mantissa0, -exponetDiff);
				}

				uint64_t carrier = 0;
				for (int i = VHACD_GOOGOL_SIZE - 1; i >= 0; i--)
				{
					uint64_t m0 = mantissa0[i];
					uint64_t m1 = mantissa1[i];
					mantissa[i] = m0 + m1 + carrier;
					carrier = CheckCarrier(m0, m1) | CheckCarrier(m0 + m1, carrier);
				}

				int sign = 0;
				if (int64_t(mantissa[0]) < 0)
				{
					sign = 1;
					NegateMantissa(mantissa);
				}

				int bits = NormalizeMantissa(mantissa);
				if (bits <= (-64 * VHACD_GOOGOL_SIZE))
				{
					tmp.m_sign = 0;
					tmp.m_exponent = 0;
				}
				else
				{
					tmp.m_sign = sign;
					tmp.m_exponent = int(exponent + bits);
				}

				memcpy(tmp.m_mantissa, mantissa, sizeof(m_mantissa));
			}
			else if (A.m_mantissa[0])
			{
				tmp = A;
			}
			else
			{
				tmp = *this;
			}

			return tmp;
		}

		Googol Googol::operator- (const Googol &A) const
		{
			Googol tmp(A);
			tmp.m_sign = !tmp.m_sign;
			return *this + tmp;
		}

		void Googol::ScaleMantissa(uint64_t* const dst, uint64_t scale) const
		{
			uint64_t carrier = 0;
			for (int i = VHACD_GOOGOL_SIZE - 1; i >= 0; i--)
			{
				if (m_mantissa[i])
				{
					uint64_t low;
					uint64_t high;
					ExtendeMultiply(scale, m_mantissa[i], high, low);
					uint64_t acc = low + carrier;
					carrier = CheckCarrier(low, carrier);
					carrier += high;
					dst[i + 1] = acc;
				}
				else
				{
					dst[i + 1] = carrier;
					carrier = 0;
				}

			}
			dst[0] = carrier;
		}

		Googol Googol::operator* (const Googol &A) const
		{
			if (m_mantissa[0] && A.m_mantissa[0])
			{
				uint64_t mantissaAcc[VHACD_GOOGOL_SIZE * 2];
				memset(mantissaAcc, 0, sizeof(mantissaAcc));
				for (int i = VHACD_GOOGOL_SIZE - 1; i >= 0; i--)
				{
					uint64_t a = m_mantissa[i];
					if (a)
					{
						uint64_t mantissaScale[2 * VHACD_GOOGOL_SIZE];
						memset(mantissaScale, 0, sizeof(mantissaScale));
						A.ScaleMantissa(&mantissaScale[i], a);

						uint64_t carrier = 0;
						for (int j = 0; j < 2 * VHACD_GOOGOL_SIZE; j++)
						{
							const int k = 2 * VHACD_GOOGOL_SIZE - 1 - j;
							uint64_t m0 = mantissaAcc[k];
							uint64_t m1 = mantissaScale[k];
							mantissaAcc[k] = m0 + m1 + carrier;
							carrier = CheckCarrier(m0, m1) | CheckCarrier(m0 + m1, carrier);
						}
					}
				}

				uint64_t carrier = 0;
				//int bits = uint64_t(LeadingZeros (mantissaAcc[0]) - 2);
				int bits = LeadingZeros(mantissaAcc[0]) - 2;
				for (int i = 0; i < 2 * VHACD_GOOGOL_SIZE; i++)
				{
					const int k = 2 * VHACD_GOOGOL_SIZE - 1 - i;
					uint64_t a = mantissaAcc[k];
					mantissaAcc[k] = (a << uint64_t(bits)) | carrier;
					carrier = a >> uint64_t(64 - bits);
				}

				int exp = m_exponent + A.m_exponent - (bits - 2);

				Googol tmp;
				tmp.m_sign = m_sign ^ A.m_sign;
				tmp.m_exponent = int(exp);
				memcpy(tmp.m_mantissa, mantissaAcc, sizeof(m_mantissa));

				return tmp;
			}
			return Googol(0.0);
		}

		Googol Googol::operator/ (const Googol &A) const
		{
			Googol tmp(1.0 / A);
			tmp = tmp * (m_two - A * tmp);
			tmp = tmp * (m_two - A * tmp);
			int test = 0;
			int passes = 0;
			do
			{
				passes++;
				Googol tmp0(tmp);
				tmp = tmp * (m_two - A * tmp);
				test = memcmp(&tmp0, &tmp, sizeof(Googol));
			} while (test && (passes < (2 * VHACD_GOOGOL_SIZE)));
			return (*this) * tmp;
		}

		Googol Googol::Abs() const
		{
			Googol tmp(*this);
			tmp.m_sign = 0;
			return tmp;
		}

		Googol Googol::Floor() const
		{
			if (m_exponent < 1)
			{
				return Googol(0.0);
			}
			int bits = m_exponent + 2;
			int start = 0;
			while (bits >= 64)
			{
				bits -= 64;
				start++;
			}

			Googol tmp(*this);
			for (int i = VHACD_GOOGOL_SIZE - 1; i > start; i--)
			{
				tmp.m_mantissa[i] = 0;
			}
			// some compilers do no like this and I do not know why is that
			//uint64_t mask = (-1LL) << (64 - bits);
			uint64_t mask(~0ULL);
			mask <<= (64 - bits);
			tmp.m_mantissa[start] &= mask;
			return tmp;
		}

		Googol Googol::InvSqrt() const
		{
			const Googol& me = *this;
			Googol x(1.0f / sqrt(me));

			int test = 0;
			int passes = 0;
			do
			{
				passes++;
				Googol tmp(x);
				x = m_half * x * (m_three - me * x * x);
				test = memcmp(&x, &tmp, sizeof(Googol));
			} while (test && (passes < (2 * VHACD_GOOGOL_SIZE)));
			return x;
		}

		Googol Googol::Sqrt() const
		{
			return *this * InvSqrt();
		}

		void Googol::ToString(char* const string) const
		{
			Googol tmp(*this);
			Googol base(10.0);
			while (double(tmp) > 1.0)
			{
				tmp = tmp / base;
			}

			int index = 0;
			while (tmp.m_mantissa[0])
			{
				tmp = tmp * base;
				Googol digit(tmp.Floor());
				tmp -= digit;
				double val = digit;
				string[index] = char(val) + '0';
				index++;
			}
			string[index] = 0;
		}

		void Googol::NegateMantissa(uint64_t* const mantissa) const
		{
			uint64_t carrier = 1;
			for (int i = VHACD_GOOGOL_SIZE - 1; i >= 0; i--)
			{
				uint64_t a = ~mantissa[i] + carrier;
				if (a)
				{
					carrier = 0;
				}
				mantissa[i] = a;
			}
		}

		void Googol::ShiftRightMantissa(uint64_t* const mantissa, int bits) const
		{
			uint64_t carrier = 0;
			if (int64_t(mantissa[0]) < int64_t(0))
			{
				carrier = uint64_t(-1);
			}

			while (bits >= 64)
			{
				for (int i = VHACD_GOOGOL_SIZE - 2; i >= 0; i--)
				{
					mantissa[i + 1] = mantissa[i];
				}
				mantissa[0] = carrier;
				bits -= 64;
			}

			if (bits > 0)
			{
				carrier <<= (64 - bits);
				for (int i = 0; i < VHACD_GOOGOL_SIZE; i++)
				{
					uint64_t a = mantissa[i];
					mantissa[i] = (a >> bits) | carrier;
					carrier = a << (64 - bits);
				}
			}
		}

		int Googol::LeadingZeros(uint64_t a) const
		{
			#define dgCOUNTBIT(mask,add)		\
			{									\
				uint64_t test = a & mask;		\
				n += test ? 0 : add;			\
				a = test ? test : (a & ~mask);	\
			}

			int n = 0;
			dgCOUNTBIT(0xffffffff00000000LL, 32);
			dgCOUNTBIT(0xffff0000ffff0000LL, 16);
			dgCOUNTBIT(0xff00ff00ff00ff00LL, 8);
			dgCOUNTBIT(0xf0f0f0f0f0f0f0f0LL, 4);
			dgCOUNTBIT(0xccccccccccccccccLL, 2);
			dgCOUNTBIT(0xaaaaaaaaaaaaaaaaLL, 1);

			return n;
		}

		int Googol::NormalizeMantissa(uint64_t* const mantissa) const
		{
			int bits = 0;
			if (int64_t(mantissa[0] * 2) < 0)
			{
				bits = 1;
				ShiftRightMantissa(mantissa, 1);
			}
			else
			{
				while (!mantissa[0] && bits > (-64 * VHACD_GOOGOL_SIZE))
				{
					bits -= 64;
					for (int i = 1; i < VHACD_GOOGOL_SIZE; i++) {
						mantissa[i - 1] = mantissa[i];
					}
					mantissa[VHACD_GOOGOL_SIZE - 1] = 0;
				}

				if (bits > (-64 * VHACD_GOOGOL_SIZE))
				{
					int n = LeadingZeros(mantissa[0]) - 2;
					if (n > 0)
					{
						uint64_t carrier = 0;
						for (int i = VHACD_GOOGOL_SIZE - 1; i >= 0; i--)
						{
							uint64_t a = mantissa[i];
							mantissa[i] = (a << n) | carrier;
							carrier = a >> (64 - n);
						}
						bits -= n;
					}
					else if (n < 0)
					{
						// this is very rare but it does happens, whee the leading zeros of the mantissa is an exact multiple of 64
						uint64_t carrier = 0;
						int shift = -n;
						for (int i = 0; i < VHACD_GOOGOL_SIZE; i++)
						{
							uint64_t a = mantissa[i];
							mantissa[i] = (a >> shift) | carrier;
							carrier = a << (64 - shift);
						}
						bits -= n;
					}
				}
			}
			return bits;
		}

		uint64_t Googol::CheckCarrier(uint64_t a, uint64_t b) const
		{
			return ((uint64_t(-1) - b) < a) ? uint64_t(1) : 0;
		}

		void Googol::ExtendeMultiply(uint64_t a, uint64_t b, uint64_t& high, uint64_t& low) const
		{
			uint64_t bLow = b & 0xffffffff;
			uint64_t bHigh = b >> 32;
			uint64_t aLow = a & 0xffffffff;
			uint64_t aHigh = a >> 32;

			uint64_t l = bLow * aLow;

			uint64_t c1 = bHigh * aLow;
			uint64_t c2 = bLow * aHigh;
			uint64_t m = c1 + c2;
			uint64_t carrier = CheckCarrier(c1, c2) << 32;

			uint64_t h = bHigh * aHigh + carrier;

			uint64_t ml = m << 32;
			uint64_t ll = l + ml;
			uint64_t mh = (m >> 32) + CheckCarrier(l, ml);
			uint64_t hh = h + mh;

			low = ll;
			high = hh;
		}

		Googol Googol::operator+= (const Googol &A)
		{
			*this = *this + A;
			return *this;
		}

		Googol Googol::operator-= (const Googol &A)
		{
			*this = *this - A;
			return *this;
		}

		bool Googol::operator> (const Googol &A) const
		{
			Googol tmp(*this - A);
			return double(tmp) > 0.0;
		}

		bool Googol::operator>= (const Googol &A) const
		{
			Googol tmp(*this - A);
			return double(tmp) >= 0.0;
		}

		bool Googol::operator< (const Googol &A) const
		{
			Googol tmp(*this - A);
			return double(tmp) < 0.0;
		}

		bool Googol::operator<= (const Googol &A) const
		{
			Googol tmp(*this - A);
			return double(tmp) <= 0.0;
		}

		bool Googol::operator== (const Googol &A) const
		{
			Googol tmp(*this - A);
			return double(tmp) == 0.0;
		}

		bool Googol::operator!= (const Googol &A) const
		{
			Googol tmp(*this - A);
			return double(tmp) != 0.0;
		}

		void Googol::Trace() const
		{
			//dTrace (("%f ", double (*this)));
		}

		double Determinant2x2(const double matrix[2][2], double* const error)
		{
			double a00xa11 = matrix[0][0] * matrix[1][1];
			double a01xa10 = matrix[0][1] * matrix[1][0];
			*error = Absolute(a00xa11) + Absolute(a01xa10);
			return a00xa11 - a01xa10;
		}

		double Determinant3x3(const double matrix[3][3], double* const error)
		{
			double sign = double(-1.0f);
			double det = double(0.0f);
			double accError = double(0.0f);
			for (int i = 0; i < 3; i++)
			{
				double cofactor[2][2];
				for (int j = 0; j < 2; j++)
				{
					int k0 = 0;
					for (int k = 0; k < 3; k++)
					{
						if (k != i)
						{
							cofactor[j][k0] = matrix[j][k];
							k0++;
						}
					}
				}

				double parcialError;
				double minorDet = Determinant2x2(cofactor, &parcialError);
				accError += parcialError * Absolute(matrix[2][i]);
				det += sign * minorDet * matrix[2][i];
				sign *= double(-1.0f);
			}

			*error = accError;
			return det;
		}

		Googol Determinant2x2(const Googol matrix[2][2])
		{
			Googol a00xa11(matrix[0][0] * matrix[1][1]);
			Googol a01xa10(matrix[0][1] * matrix[1][0]);
			return a00xa11 - a01xa10;
		}

		Googol Determinant3x3(const Googol matrix[3][3])
		{
			Googol negOne(double(-1.0f));
			Googol sign(double(-1.0f));
			Googol det = double(0.0f);
			for (int i = 0; i < 3; i++)
			{
				Googol cofactor[2][2];
				for (int j = 0; j < 2; j++)
				{
					int k0 = 0;
					for (int k = 0; k < 3; k++)
					{
						if (k != i)
						{
							cofactor[j][k0] = matrix[j][k];
							k0++;
						}
					}
				}

				Googol minorDet(Determinant2x2(cofactor));
				det = det + sign * minorDet * matrix[2][i];
				sign = sign * negOne;
			}
			return det;
		}
	}
}


namespace nd
{
	namespace juliohull
	{
		class ConvexHullVertex;
		class ConvexHullAABBTreeNode;

		class ConvexHullFace
		{
			public:
			ConvexHullFace();
			double Evalue(const hullVector* const pointArray, const hullVector& point) const;
			hullPlane GetPlaneEquation(const hullVector* const pointArray, bool& isValid) const;

			public:
			int m_index[3];

			private:
			int m_mark;
			List<ConvexHullFace>::ndNode* m_twin[3];

			friend class ConvexHull;
		};

		class ConvexHull : public List<ConvexHullFace>
		{
			class ndNormalMap;

			public:
			ConvexHull(const ConvexHull& source);
			ConvexHull(const double* const vertexCloud, int strideInBytes, int count, double distTol, int maxVertexCount = 0x7fffffff);
			~ConvexHull();

			const std::vector<hullVector>& GetVertexPool() const;

			private:
			void BuildHull(const double* const vertexCloud, int strideInBytes, int count, double distTol, int maxVertexCount);

			void GetUniquePoints(std::vector<ConvexHullVertex>& points);
			int InitVertexArray(std::vector<ConvexHullVertex>& points, void* const memoryPool, int maxMemSize);

			ConvexHullAABBTreeNode* BuildTreeNew(std::vector<ConvexHullVertex>& points, char** const memoryPool, int& maxMemSize) const;
			ConvexHullAABBTreeNode* BuildTreeOld(std::vector<ConvexHullVertex>& points, char** const memoryPool, int& maxMemSize);
			ConvexHullAABBTreeNode* BuildTreeRecurse(ConvexHullAABBTreeNode* const parent, ConvexHullVertex* const points, int count, int baseIndex, char** const memoryPool, int& maxMemSize) const;

			ndNode* AddFace(int i0, int i1, int i2);

			void CalculateConvexHull3d(ConvexHullAABBTreeNode* vertexTree, std::vector<ConvexHullVertex>& points, int count, double distTol, int maxVertexCount);

			int SupportVertex(ConvexHullAABBTreeNode** const tree, const std::vector<ConvexHullVertex>& points, const hullVector& dir, const bool removeEntry = true) const;
			double TetrahedrumVolume(const hullVector& p0, const hullVector& p1, const hullVector& p2, const hullVector& p3) const;

			hullVector m_aabbP0;
			hullVector m_aabbP1;
			double m_diag;
			std::vector<hullVector> m_points;
		};
	}
}

namespace nd
{
	namespace juliohull
	{

		#define VHACD_CONVEXHULL_3D_VERTEX_CLUSTER_SIZE 8

		ConvexHullFace::ConvexHullFace()
		{
			m_mark = 0;
			m_twin[0] = nullptr;
			m_twin[1] = nullptr;
			m_twin[2] = nullptr;
		}

		hullPlane ConvexHullFace::GetPlaneEquation(const hullVector* const pointArray, bool& isvalid) const
		{
			const hullVector& p0 = pointArray[m_index[0]];
			const hullVector& p1 = pointArray[m_index[1]];
			const hullVector& p2 = pointArray[m_index[2]];
			hullPlane plane(p0, p1, p2);

			isvalid = false;
			double mag2 = plane.DotProduct(plane);
			if (mag2 > 1.0e-16f)
			{
				isvalid = true;
				plane = plane.Scale(1.0f / sqrt(mag2));
			}
			return plane;
		}

		double ConvexHullFace::Evalue(const hullVector* const pointArray, const hullVector& point) const
		{
			const hullVector& p0 = pointArray[m_index[0]];
			const hullVector& p1 = pointArray[m_index[1]];
			const hullVector& p2 = pointArray[m_index[2]];

			double matrix[3][3];
			for (int i = 0; i < 3; ++i) 
			{
				matrix[0][i] = p2[i] - p0[i];
				matrix[1][i] = p1[i] - p0[i];
				matrix[2][i] = point[i] - p0[i];
			}

			double error;
			double det = Determinant3x3(matrix, &error);

			// the code use double, however the threshold for accuracy test is the machine precision of a float.
			// by changing this to a smaller number, the code should run faster since many small test will be considered valid
			// the precision must be a power of two no smaller than the machine precision of a double, (1<<48)
			// float64(1<<30) can be a good value

			// double precision	= double (1.0f) / double (1<<30);
			double precision = double(1.0f) / double(1 << 24);
			double errbound = error * precision;
			if (fabs(det) > errbound) 
			{
				return det;
			}
	
			Googol exactMatrix[3][3];
			for (int i = 0; i < 3; ++i) 
			{
				exactMatrix[0][i] = Googol(p2[i]) - Googol(p0[i]);
				exactMatrix[1][i] = Googol(p1[i]) - Googol(p0[i]);
				exactMatrix[2][i] = Googol(point[i]) - Googol(p0[i]);
			}
			return Determinant3x3(exactMatrix);
		}

		class ConvexHullVertex : public hullVector
		{
			public:
			int m_mark;
		};

		class ConvexHullAABBTreeNode
		{
			public:
			//ConvexHullAABBTreeNode(ConvexHullAABBTreeNode* const parent)
			ConvexHullAABBTreeNode()
				:m_left(nullptr)
				,m_right(nullptr)
				,m_parent(nullptr)
			{
			}

			ConvexHullAABBTreeNode(ConvexHullAABBTreeNode* const parent)
				:m_left(nullptr)
				,m_right(nullptr)
				,m_parent(parent)
			{
			}

			hullVector m_box[2];
			ConvexHullAABBTreeNode* m_left;
			ConvexHullAABBTreeNode* m_right;
			ConvexHullAABBTreeNode* m_parent;
		};

		class ConvexHull3dPointCluster : public ConvexHullAABBTreeNode
		{
			public:
			ConvexHull3dPointCluster()
				:ConvexHullAABBTreeNode()
			{
			}

			ConvexHull3dPointCluster(ConvexHullAABBTreeNode* const parent)
				:ConvexHullAABBTreeNode(parent)
			{
			}

			int m_count;
			int m_indices[VHACD_CONVEXHULL_3D_VERTEX_CLUSTER_SIZE];
		};

		class ConvexHull::ndNormalMap
		{
			public:
			ndNormalMap()
				:m_count(sizeof(m_normal) / sizeof(m_normal[0]))
			{
				hullVector p0(double(1.0f), double(0.0f), double(0.0f), double(0.0f));
				hullVector p1(double(-1.0f), double(0.0f), double(0.0f), double(0.0f));
				hullVector p2(double(0.0f), double(1.0f), double(0.0f), double(0.0f));
				hullVector p3(double(0.0f), double(-1.0f), double(0.0f), double(0.0f));
				hullVector p4(double(0.0f), double(0.0f), double(1.0f), double(0.0f));
				hullVector p5(double(0.0f), double(0.0f), double(-1.0f), double(0.0f));

				int count = 0;
				int subdivitions = 2;
				TessellateTriangle(subdivitions, p4, p0, p2, count);
				TessellateTriangle(subdivitions, p0, p5, p2, count);
				TessellateTriangle(subdivitions, p5, p1, p2, count);
				TessellateTriangle(subdivitions, p1, p4, p2, count);
				TessellateTriangle(subdivitions, p0, p4, p3, count);
				TessellateTriangle(subdivitions, p5, p0, p3, count);
				TessellateTriangle(subdivitions, p1, p5, p3, count);
				TessellateTriangle(subdivitions, p4, p1, p3, count);
			}

			static const ndNormalMap& GetNormaMap()
			{
				static ndNormalMap normalMap;
				return normalMap;
			}

			void TessellateTriangle(int level, const hullVector& p0, const hullVector& p1, const hullVector& p2, int& count)
			{
				if (level) 
				{
					assert(fabs(p0.DotProduct(p0) - double(1.0f)) < double(1.0e-4f));
					assert(fabs(p1.DotProduct(p1) - double(1.0f)) < double(1.0e-4f));
					assert(fabs(p2.DotProduct(p2) - double(1.0f)) < double(1.0e-4f));
					hullVector p01(p0 + p1);
					hullVector p12(p1 + p2);
					hullVector p20(p2 + p0);

					p01 = p01.Scale(1.0 / sqrt(p01.DotProduct(p01)));
					p12 = p12.Scale(1.0 / sqrt(p12.DotProduct(p12)));
					p20 = p20.Scale(1.0 / sqrt(p20.DotProduct(p20)));

					assert(fabs(p01.DotProduct(p01) - double(1.0f)) < double(1.0e-4f));
					assert(fabs(p12.DotProduct(p12) - double(1.0f)) < double(1.0e-4f));
					assert(fabs(p20.DotProduct(p20) - double(1.0f)) < double(1.0e-4f));

					TessellateTriangle(level - 1, p0, p01, p20, count);
					TessellateTriangle(level - 1, p1, p12, p01, count);
					TessellateTriangle(level - 1, p2, p20, p12, count);
					TessellateTriangle(level - 1, p01, p12, p20, count);
				}
				else 
				{
					hullPlane n(p0, p1, p2);
					n = n.Scale(double(1.0f) / sqrt(n.DotProduct(n)));
					n.m_w = double(0.0f);
					int index = dBitReversal(count, sizeof(m_normal) / sizeof(m_normal[0]));
					m_normal[index] = n;
					count++;
					assert(count <= int(sizeof(m_normal) / sizeof(m_normal[0])));
				}
			}

			hullVector m_normal[128];
			int m_count;
		};

		ConvexHull::ConvexHull(const double* const vertexCloud, int strideInBytes, int count, double distTol, int maxVertexCount)
			:List<ConvexHullFace>()
			,m_aabbP0(0)
			,m_aabbP1(0)
			,m_diag()
			,m_points()
		{
			m_points.resize(0);
			if (count >= 4)
			{
				BuildHull(vertexCloud, strideInBytes, count, distTol, maxVertexCount);
			}
		}

		ConvexHull::~ConvexHull()
		{
		}

		const std::vector<hullVector>& ConvexHull::GetVertexPool() const
		{
			return m_points;
		}


		void ConvexHull::BuildHull(const double* const vertexCloud, int strideInBytes, int count, double distTol, int maxVertexCount)
		{
			int treeCount = count / (VHACD_CONVEXHULL_3D_VERTEX_CLUSTER_SIZE >> 1);
			if (treeCount < 4)
			{
				treeCount = 4;
			}
			treeCount *= 2;

			std::vector<ConvexHullVertex> points(count);
			std::vector<ConvexHull3dPointCluster> treePool(treeCount + 256);
			points.resize(count);
			treePool.resize(treeCount + 256);

			const int stride = int(strideInBytes / sizeof(double));
			for (int i = 0; i < count; ++i)
			{
				int index = i * stride;
				hullVector& vertex = points[i];
				vertex = hullVector(vertexCloud[index], vertexCloud[index + 1], vertexCloud[index + 2], double(0.0f));
				points[i].m_mark = 0;
			}
			count = InitVertexArray(points, &treePool[0], sizeof (ConvexHull3dPointCluster) * int (treePool.size()));

			if (m_points.size() >= 4)
			{
				CalculateConvexHull3d(&treePool[0], points, count, distTol, maxVertexCount);
			}
		}

		void ConvexHull::GetUniquePoints(std::vector<ConvexHullVertex>& points)
		{
			class CompareVertex
			{
				public:
				int Compare(const ConvexHullVertex& elementA, const ConvexHullVertex& elementB) const
				{
					for (int i = 0; i < 3; i++)
					{
						if (elementA[i] < elementB[i])
						{
							return -1;
						}
						else if (elementA[i] > elementB[i])
						{
							return 1;
						}
					}
					return 0;
				}
			};

			int count = int(points.size());
			Sort<ConvexHullVertex, CompareVertex>(&points[0], count);

			int indexCount = 0;
			CompareVertex compareVetex;
			for (int i = 1; i < count; ++i)
			{
				for (; i < count; ++i)
				{
					if (compareVetex.Compare(points[indexCount], points[i]))
					{
						indexCount++;
						points[indexCount] = points[i];
						break;
					}
				}
			}
			points.resize(indexCount + 1);
		}

		ConvexHullAABBTreeNode* ConvexHull::BuildTreeRecurse(ConvexHullAABBTreeNode* const parent, ConvexHullVertex* const points, int count, int baseIndex, char** memoryPool, int& maxMemSize) const
		{
			ConvexHullAABBTreeNode* tree = nullptr;

			assert(count);
			hullVector minP(double(1.0e15f));
			hullVector maxP(-double(1.0e15f));
			if (count <= VHACD_CONVEXHULL_3D_VERTEX_CLUSTER_SIZE)
			{
				ConvexHull3dPointCluster* const clump = new (*memoryPool) ConvexHull3dPointCluster();
				*memoryPool += sizeof(ConvexHull3dPointCluster);
				maxMemSize -= sizeof(ConvexHull3dPointCluster);
				assert(maxMemSize >= 0);

				assert(clump);
				clump->m_count = count;
				for (int i = 0; i < count; ++i)
				{
					clump->m_indices[i] = i + baseIndex;

					const hullVector& p = points[i];
					minP = minP.GetMin(p);
					maxP = maxP.GetMax(p);
				}

				clump->m_left = nullptr;
				clump->m_right = nullptr;
				tree = clump;
			}
			else
			{
				hullVector median(0);
				hullVector varian(0);
				for (int i = 0; i < count; ++i)
				{
					const hullVector& p = points[i];
					minP = minP.GetMin(p);
					maxP = maxP.GetMax(p);
					median += p;
					varian += p * p;
				}

				varian = varian.Scale(double(count)) - median * median;
				int index = 0;
				double maxVarian = double(-1.0e10f);
				for (int i = 0; i < 3; ++i)
				{
					if (varian[i] > maxVarian)
					{
						index = i;
						maxVarian = varian[i];
					}
				}
				hullVector center(median.Scale(double(1.0f) / double(count)));

				double test = center[index];

				int i0 = 0;
				int i1 = count - 1;
				do
				{
					for (; i0 <= i1; i0++)
					{
						double val = points[i0][index];
						if (val > test)
						{
							break;
						}
					}

					for (; i1 >= i0; i1--)
					{
						double val = points[i1][index];
						if (val < test)
						{
							break;
						}
					}

					if (i0 < i1)
					{
						Swap(points[i0], points[i1]);
						i0++;
						i1--;
					}
				} while (i0 <= i1);

				if (i0 == 0)
				{
					i0 = count / 2;
				}
				if (i0 >= (count - 1))
				{
					i0 = count / 2;
				}

				tree = new (*memoryPool) ConvexHullAABBTreeNode();
				*memoryPool += sizeof(ConvexHullAABBTreeNode);
				maxMemSize -= sizeof(ConvexHullAABBTreeNode);
				assert(maxMemSize >= 0);

				assert(i0);
				assert(count - i0);

				tree->m_left = BuildTreeRecurse(tree, points, i0, baseIndex, memoryPool, maxMemSize);
				tree->m_right = BuildTreeRecurse(tree, &points[i0], count - i0, i0 + baseIndex, memoryPool, maxMemSize);
			}

			assert(tree);
			tree->m_parent = parent;
			tree->m_box[0] = minP - hullVector(double(1.0e-3f));
			tree->m_box[1] = maxP + hullVector(double(1.0e-3f));
			return tree;
		}

		ConvexHullAABBTreeNode* ConvexHull::BuildTreeOld(std::vector<ConvexHullVertex>& points, char** const memoryPool, int& maxMemSize)
		{
			GetUniquePoints(points);
			int count = int(points.size());
			if (count < 4)
			{
				return nullptr;
			}
			return BuildTreeRecurse(nullptr, &points[0], count, 0, memoryPool, maxMemSize);
		}

		ConvexHullAABBTreeNode* ConvexHull::BuildTreeNew(std::vector<ConvexHullVertex>& points, char** const memoryPool, int& maxMemSize) const
		{
			class dCluster
			{
				public:
				hullVector m_sum;
				hullVector m_sum2;
				int m_start;
				int m_count;
			};

			dCluster firstCluster;
			firstCluster.m_start = 0;
			firstCluster.m_count = int (points.size());
			firstCluster.m_sum = hullVector(0);
			firstCluster.m_sum2 = hullVector(0);

			for (int i = 0; i < firstCluster.m_count; ++i)
			{
				const hullVector& p = points[i];
				firstCluster.m_sum += p;
				firstCluster.m_sum2 += p * p;
			}

			int baseCount = 0;
			const int clusterSize = 16;

			if (firstCluster.m_count > clusterSize)
			{
				dCluster spliteStack[128];
				spliteStack[0] = firstCluster;
				int stack = 1;

				while (stack)
				{
					stack--;
					dCluster cluster (spliteStack[stack]);

					const hullVector origin(cluster.m_sum.Scale(1.0f / cluster.m_count));
					const hullVector variance2(cluster.m_sum2.Scale(1.0f / cluster.m_count) - origin * origin);
					double maxVariance2 = Max(Max(variance2.getX(), variance2.getY()), variance2.getZ());

					if ((cluster.m_count <= clusterSize) || (stack > (sizeof(spliteStack) / sizeof(spliteStack[0]) - 4)) || (maxVariance2 < 1.e-4f))
					{
						// no sure if this is beneficial, 
						// the array is so small that seem too much overhead
						//int maxIndex = 0;
						//double min_x = 1.0e20f;
						//for (int i = 0; i < cluster.m_count; ++i)
						//{
						//	if (points[cluster.m_start + i].getX() < min_x)
						//	{
						//		maxIndex = i;
						//		min_x = points[cluster.m_start + i].getX();
						//	}
						//}
						//Swap(points[cluster.m_start], points[cluster.m_start + maxIndex]);
						//
						//for (int i = 2; i < cluster.m_count; ++i)
						//{
						//	int j = i;
						//	ConvexHullVertex tmp(points[cluster.m_start + i]);
						//	for (; points[cluster.m_start + j - 1].getX() > tmp.getX(); --j)
						//	{
						//		assert(j > 0);
						//		points[cluster.m_start + j] = points[cluster.m_start + j - 1];
						//	}
						//	points[cluster.m_start + j] = tmp;
						//}

						int count = cluster.m_count;
						for (int i = cluster.m_count - 1; i > 0; --i)
						{
							for (int j = i - 1; j >= 0; --j)
							{
								hullVector error(points[cluster.m_start + j] - points[cluster.m_start + i]);
								double mag2 = error.DotProduct(error);
								if (mag2 < 1.0e-6)
								{
									points[cluster.m_start + j] = points[cluster.m_start + i];
									count--;
									break;
								}
							}
						}

						assert(baseCount <= cluster.m_start);
						for (int i = 0; i < count; ++i)
						{
							points[baseCount] = points[cluster.m_start + i];
							baseCount++;
						}
					}
					else
					{
						int firstSortAxis = 0;
						if ((variance2.getY() >= variance2.getX()) && (variance2.getY() >= variance2.getZ()))
						{
							firstSortAxis = 1;
						}
						else if ((variance2.getZ() >= variance2.getX()) && (variance2.getZ() >= variance2.getY()))
						{
							firstSortAxis = 2;
						}
						double axisVal = origin[firstSortAxis];

						int i0 = 0;
						int i1 = cluster.m_count - 1;

						const int start = cluster.m_start;
						while (i0 < i1)
						{
							while ((points[start + i0][firstSortAxis] <= axisVal) && (i0 < i1))
							{
								++i0;
							};

							while ((points[start + i1][firstSortAxis] > axisVal) && (i0 < i1))
							{
								--i1;
							}

							assert(i0 <= i1);
							if (i0 < i1)
							{
								Swap(points[start + i0], points[start + i1]);
								++i0;
								--i1;
							}
						}

						while ((points[start + i0][firstSortAxis] <= axisVal) && (i0 < cluster.m_count))
						{
							++i0;
						};

						#ifdef _DEBUG
						for (int i = 0; i < i0; ++i)
						{
							assert(points[start + i][firstSortAxis] <= axisVal);
						}

						for (int i = i0; i < cluster.m_count; ++i)
						{
							assert(points[start + i][firstSortAxis] > axisVal);
						}
						#endif

						hullVector xc(0);
						hullVector x2c(0);
						for (int i = 0; i < i0; ++i)
						{
							const hullVector& x = points[start + i];
							xc += x;
							x2c += x * x;
						}

						dCluster cluster_i1(cluster);
						cluster_i1.m_start = start + i0;
						cluster_i1.m_count = cluster.m_count - i0;
						cluster_i1.m_sum -= xc;
						cluster_i1.m_sum2 -= x2c;
						spliteStack[stack] = cluster_i1;
						assert(cluster_i1.m_count > 0);
						stack++;

						dCluster cluster_i0(cluster);
						cluster_i0.m_start = start;
						cluster_i0.m_count = i0;
						cluster_i0.m_sum = xc;
						cluster_i0.m_sum2 = x2c;
						assert(cluster_i0.m_count > 0);
						spliteStack[stack] = cluster_i0;
						stack++;
					}
				}
			}

			points.resize(baseCount);
			if (baseCount < 4)
			{
				return nullptr;
			}

			hullVector sum(0);
			hullVector sum2(0);
			hullVector minP(double(1.0e15f));
			hullVector maxP(-double(1.0e15f));
			class dTreeBox
			{
				public:
				hullVector m_min;
				hullVector m_max;
				hullVector m_sum;
				hullVector m_sum2;
				ConvexHullAABBTreeNode* m_parent;
				ConvexHullAABBTreeNode** m_child;
				int m_start;
				int m_count;
			};

			for (int i = 0; i < baseCount; ++i)
			{
				const hullVector& p = points[i];
				sum += p;
				sum2 += p * p;
				minP = minP.GetMin(p);
				maxP = maxP.GetMax(p);
			}
	
			dTreeBox treeBoxStack[128];
			treeBoxStack[0].m_start = 0;
			treeBoxStack[0].m_count = baseCount;
			treeBoxStack[0].m_sum = sum;
			treeBoxStack[0].m_sum2 = sum2;
			treeBoxStack[0].m_min = minP;
			treeBoxStack[0].m_max = maxP;
			treeBoxStack[0].m_child = nullptr;
			treeBoxStack[0].m_parent = nullptr;

			int stack = 1;
			ConvexHullAABBTreeNode* root = nullptr;
			while (stack)
			{
				stack--;
				dTreeBox box (treeBoxStack[stack]);
				if (box.m_count <= VHACD_CONVEXHULL_3D_VERTEX_CLUSTER_SIZE)
				{
					ConvexHull3dPointCluster* const clump = new (*memoryPool) ConvexHull3dPointCluster(box.m_parent);
					*memoryPool += sizeof(ConvexHull3dPointCluster);
					maxMemSize -= sizeof(ConvexHull3dPointCluster);
					assert(maxMemSize >= 0);
		
					assert(clump);
					clump->m_count = box.m_count;
					for (int i = 0; i < box.m_count; ++i)
					{
						clump->m_indices[i] = i + box.m_start;
					}
					clump->m_box[0] = box.m_min;
					clump->m_box[1] = box.m_max;

					if (box.m_child)
					{
						*box.m_child = clump;
					}

					if (!root)
					{
						root = clump;
					}
				}
				else
				{
					const hullVector origin(box.m_sum.Scale(1.0f / box.m_count));
					const hullVector variance2(box.m_sum2.Scale(1.0f / box.m_count) - origin * origin);

					int firstSortAxis = 0;
					if ((variance2.getY() >= variance2.getX()) && (variance2.getY() >= variance2.getZ()))
					{
						firstSortAxis = 1;
					}
					else if ((variance2.getZ() >= variance2.getX()) && (variance2.getZ() >= variance2.getY()))
					{
						firstSortAxis = 2;
					}
					double axisVal = origin[firstSortAxis];

					int i0 = 0;
					int i1 = box.m_count - 1;

					const int start = box.m_start;
					while (i0 < i1)
					{
						while ((points[start + i0][firstSortAxis] <= axisVal) && (i0 < i1))
						{
							++i0;
						};

						while ((points[start + i1][firstSortAxis] > axisVal) && (i0 < i1))
						{
							--i1;
						}

						assert(i0 <= i1);
						if (i0 < i1)
						{
							Swap(points[start + i0], points[start + i1]);
							++i0;
							--i1;
						}
					}

					while ((points[start + i0][firstSortAxis] <= axisVal) && (i0 < box.m_count))
					{
						++i0;
					};

					#ifdef _DEBUG
					for (int i = 0; i < i0; ++i)
					{
						assert(points[start + i][firstSortAxis] <= axisVal);
					}

					for (int i = i0; i < box.m_count; ++i)
					{
						assert(points[start + i][firstSortAxis] > axisVal);
					}
					#endif

					ConvexHullAABBTreeNode* const node = new (*memoryPool) ConvexHullAABBTreeNode(box.m_parent);
					*memoryPool += sizeof(ConvexHullAABBTreeNode);
					maxMemSize -= sizeof(ConvexHullAABBTreeNode);
					assert(maxMemSize >= 0);

					node->m_box[0] = box.m_min;
					node->m_box[1] = box.m_max;
					if (box.m_child)
					{
						*box.m_child = node;
					}

					if (!root)
					{
						root = node;
					}

					{
						hullVector xc(0);
						hullVector x2c(0);
						hullVector p0(double(1.0e15f));
						hullVector p1(-double(1.0e15f));
						for (int i = i0; i < box.m_count; ++i)
						{
							const hullVector& p = points[start + i];
							xc += p;
							x2c += p * p;
							p0 = p0.GetMin(p);
							p1 = p1.GetMax(p);
						}

						dTreeBox cluster_i1(box);
						cluster_i1.m_start = start + i0;
						cluster_i1.m_count = box.m_count - i0;
						cluster_i1.m_sum = xc;
						cluster_i1.m_sum2 = x2c;
						cluster_i1.m_min = p0;
						cluster_i1.m_max = p1;
						cluster_i1.m_parent = node;
						cluster_i1.m_child = &node->m_right;
						treeBoxStack[stack] = cluster_i1;
						assert(cluster_i1.m_count > 0);
						stack++;
					}

					{
						hullVector xc(0);
						hullVector x2c(0);
						hullVector p0(double(1.0e15f));
						hullVector p1(-double(1.0e15f));
						for (int i = 0; i < i0; ++i)
						{
							const hullVector& p = points[start + i];
							xc += p;
							x2c += p * p;
							p0 = p0.GetMin(p);
							p1 = p1.GetMax(p);
						}

						dTreeBox cluster_i0(box);
						cluster_i0.m_start = start;
						cluster_i0.m_count = i0;
						cluster_i0.m_min = p0;
						cluster_i0.m_max = p1;
						cluster_i0.m_sum = xc;
						cluster_i0.m_sum2 = x2c;
						cluster_i0.m_parent = node;
						cluster_i0.m_child = &node->m_left;
						assert(cluster_i0.m_count > 0);
						treeBoxStack[stack] = cluster_i0;
						stack++;
					}
				}
			}
	
			return root;
		}

		int ConvexHull::SupportVertex(ConvexHullAABBTreeNode** const treePointer, const std::vector<ConvexHullVertex>& points, const hullVector& dirPlane, const bool removeEntry) const
		{
		#define DG_STACK_DEPTH_3D 64
			double aabbProjection[DG_STACK_DEPTH_3D];
			const ConvexHullAABBTreeNode *stackPool[DG_STACK_DEPTH_3D];

			hullVector dir(dirPlane);

			int index = -1;
			int stack = 1;
			stackPool[0] = *treePointer;
			aabbProjection[0] = double(1.0e20f);
			double maxProj = double(-1.0e20f);
			int ix = (dir[0] > double(0.0f)) ? 1 : 0;
			int iy = (dir[1] > double(0.0f)) ? 1 : 0;
			int iz = (dir[2] > double(0.0f)) ? 1 : 0;
			while (stack)
			{
				stack--;
				double boxSupportValue = aabbProjection[stack];
				if (boxSupportValue > maxProj)
				{
					const ConvexHullAABBTreeNode* const me = stackPool[stack];

					if (me->m_left && me->m_right)
					{
						const hullVector leftSupportPoint(me->m_left->m_box[ix].getX(), me->m_left->m_box[iy].getY(), me->m_left->m_box[iz].getZ(), 0.0f);
						double leftSupportDist = leftSupportPoint.DotProduct(dir);

						const hullVector rightSupportPoint(me->m_right->m_box[ix].getX(), me->m_right->m_box[iy].getY(), me->m_right->m_box[iz].getZ(), 0.0f);
						double rightSupportDist = rightSupportPoint.DotProduct(dir);

						if (rightSupportDist >= leftSupportDist)
						{
							aabbProjection[stack] = leftSupportDist;
							stackPool[stack] = me->m_left;
							stack++;
							assert(stack < DG_STACK_DEPTH_3D);
							aabbProjection[stack] = rightSupportDist;
							stackPool[stack] = me->m_right;
							stack++;
							assert(stack < DG_STACK_DEPTH_3D);
						}
						else
						{
							aabbProjection[stack] = rightSupportDist;
							stackPool[stack] = me->m_right;
							stack++;
							assert(stack < DG_STACK_DEPTH_3D);
							aabbProjection[stack] = leftSupportDist;
							stackPool[stack] = me->m_left;
							stack++;
							assert(stack < DG_STACK_DEPTH_3D);
						}
					}
					else
					{
						ConvexHull3dPointCluster* const cluster = (ConvexHull3dPointCluster*)me;
						for (int i = 0; i < cluster->m_count; ++i)
						{
							const ConvexHullVertex& p = points[cluster->m_indices[i]];
							assert(p.getX() >= cluster->m_box[0].getX());
							assert(p.getX() <= cluster->m_box[1].getX());
							assert(p.getY() >= cluster->m_box[0].getY());
							assert(p.getY() <= cluster->m_box[1].getY());
							assert(p.getZ() >= cluster->m_box[0].getZ());
							assert(p.getZ() <= cluster->m_box[1].getZ());
							if (!p.m_mark)
							{
								//assert(p.m_w == double(0.0f));
								double dist = p.DotProduct(dir);
								if (dist > maxProj)
								{
									maxProj = dist;
									index = cluster->m_indices[i];
								}
							}
							else if (removeEntry)
							{
								cluster->m_indices[i] = cluster->m_indices[cluster->m_count - 1];
								cluster->m_count = cluster->m_count - 1;
								i--;
							}
						}

						if (cluster->m_count == 0)
						{
							ConvexHullAABBTreeNode* const parent = cluster->m_parent;
							if (parent)
							{
								ConvexHullAABBTreeNode* const sibling = (parent->m_left != cluster) ? parent->m_left : parent->m_right;
								assert(sibling != cluster);
								ConvexHullAABBTreeNode* const grandParent = parent->m_parent;
								if (grandParent)
								{
									sibling->m_parent = grandParent;
									if (grandParent->m_right == parent)
									{
										grandParent->m_right = sibling;
									}
									else
									{
										grandParent->m_left = sibling;
									}
								}
								else
								{
									sibling->m_parent = nullptr;
									*treePointer = sibling;
								}
							}
						}
					}
				}
			}

			assert(index != -1);
			return index;
		}

		double ConvexHull::TetrahedrumVolume(const hullVector& p0, const hullVector& p1, const hullVector& p2, const hullVector& p3) const
		{
			const hullVector p1p0(p1 - p0);
			const hullVector p2p0(p2 - p0);
			const hullVector p3p0(p3 - p0);
			return p3p0.DotProduct(p1p0.CrossProduct(p2p0));
		}

		int ConvexHull::InitVertexArray(std::vector<ConvexHullVertex>& points, void* const memoryPool, int maxMemSize)
		{
		#if 1
			ConvexHullAABBTreeNode* tree = BuildTreeOld(points, (char**)&memoryPool, maxMemSize);
		#else
			ConvexHullAABBTreeNode* tree = BuildTreeNew(points, (char**)&memoryPool, maxMemSize);
		#endif
			int count = int (points.size());
			if (count < 4)
			{
				m_points.resize(0);
				return 0;
			}
		
			m_points.resize(count);
			m_aabbP0 = tree->m_box[0];
			m_aabbP1 = tree->m_box[1];
	
			hullVector boxSize(tree->m_box[1] - tree->m_box[0]);
			m_diag = double(sqrt(boxSize.DotProduct(boxSize)));
			const ndNormalMap& normalMap = ndNormalMap::GetNormaMap();
	
			int index0 = SupportVertex(&tree, points, normalMap.m_normal[0]);
			m_points[0] = points[index0];
			points[index0].m_mark = 1;
	
			bool validTetrahedrum = false;
			hullVector e1(0.0);
			for (int i = 1; i < normalMap.m_count; ++i)
			{
				int index = SupportVertex(&tree, points, normalMap.m_normal[i]);
				assert(index >= 0);
	
				e1 = points[index] - m_points[0];
				double error2 = e1.DotProduct(e1);
				if (error2 > (double(1.0e-4f) * m_diag * m_diag))
				{
					m_points[1] = points[index];
					points[index].m_mark = 1;
					validTetrahedrum = true;
					break;
				}
			}
			if (!validTetrahedrum)
			{
				m_points.resize(0);
				assert(0);
				return count;
			}
	
			validTetrahedrum = false;
			hullVector e2(0.0);
			hullVector normal(0.0);
			for (int i = 2; i < normalMap.m_count; ++i)
			{
				int index = SupportVertex(&tree, points, normalMap.m_normal[i]);
				assert(index >= 0);
				e2 = points[index] - m_points[0];
				normal = e1.CrossProduct(e2);
				double error2 = sqrt(normal.DotProduct(normal));
				if (error2 > (double(1.0e-4f) * m_diag * m_diag))
				{
					m_points[2] = points[index];
					points[index].m_mark = 1;
					validTetrahedrum = true;
					break;
				}
			}
	
			if (!validTetrahedrum)
			{
				m_points.resize(0);
				assert(0);
				return count;
			}
	
			// find the largest possible tetrahedron
			validTetrahedrum = false;
			hullVector e3(0.0);
	
			index0 = SupportVertex(&tree, points, normal);
			e3 = points[index0] - m_points[0];
			double err2 = normal.DotProduct(e3);
			if (fabs(err2) > (double(1.0e-6f) * m_diag * m_diag))
			{
				// we found a valid tetrahedral, about and start build the hull by adding the rest of the points
				m_points[3] = points[index0];
				points[index0].m_mark = 1;
				validTetrahedrum = true;
			}
			if (!validTetrahedrum)
			{
				hullVector n(normal.Scale(double(-1.0f)));
				int index = SupportVertex(&tree, points, n);
				e3 = points[index] - m_points[0];
				double error2 = normal.DotProduct(e3);
				if (fabs(error2) > (double(1.0e-6f) * m_diag * m_diag))
				{
					// we found a valid tetrahedral, about and start build the hull by adding the rest of the points
					m_points[3] = points[index];
					points[index].m_mark = 1;
					validTetrahedrum = true;
				}
			}
			if (!validTetrahedrum)
			{
				for (int i = 3; i < normalMap.m_count; ++i)
				{
					int index = SupportVertex(&tree, points, normalMap.m_normal[i]);
					assert(index >= 0);
	
					//make sure the volume of the fist tetrahedral is no negative
					e3 = points[index] - m_points[0];
					double error2 = normal.DotProduct(e3);
					if (fabs(error2) > (double(1.0e-6f) * m_diag * m_diag))
					{
						// we found a valid tetrahedral, about and start build the hull by adding the rest of the points
						m_points[3] = points[index];
						points[index].m_mark = 1;
						validTetrahedrum = true;
						break;
					}
				}
			}
			if (!validTetrahedrum)
			{
				// the points do not form a convex hull
				m_points.resize(0);
				return count;
			}
	
			m_points.resize(4);
			double volume = TetrahedrumVolume(m_points[0], m_points[1], m_points[2], m_points[3]);
			if (volume > double(0.0f))
			{
				Swap(m_points[2], m_points[3]);
			}
			assert(TetrahedrumVolume(m_points[0], m_points[1], m_points[2], m_points[3]) < double(0.0f));
			return count;
		}

		ConvexHull::ndNode* ConvexHull::AddFace(int i0, int i1, int i2)
		{
			ndNode* const node = Append();
			ConvexHullFace& face = node->GetInfo();

			face.m_index[0] = i0;
			face.m_index[1] = i1;
			face.m_index[2] = i2;
			return node;
		}

		void ConvexHull::CalculateConvexHull3d(ConvexHullAABBTreeNode* vertexTree, std::vector<ConvexHullVertex>& points, int count, double distTol, int maxVertexCount)
		{
			distTol = fabs(distTol) * m_diag;
			ndNode* const f0Node = AddFace(0, 1, 2);
			ndNode* const f1Node = AddFace(0, 2, 3);
			ndNode* const f2Node = AddFace(2, 1, 3);
			ndNode* const f3Node = AddFace(1, 0, 3);

			ConvexHullFace* const f0 = &f0Node->GetInfo();
			ConvexHullFace* const f1 = &f1Node->GetInfo();
			ConvexHullFace* const f2 = &f2Node->GetInfo();
			ConvexHullFace* const f3 = &f3Node->GetInfo();

			f0->m_twin[0] = f3Node;
			f0->m_twin[1] = f2Node;
			f0->m_twin[2] = f1Node;

			f1->m_twin[0] = f0Node;
			f1->m_twin[1] = f2Node;
			f1->m_twin[2] = f3Node;

			f2->m_twin[0] = f0Node;
			f2->m_twin[1] = f3Node;
			f2->m_twin[2] = f1Node;

			f3->m_twin[0] = f0Node;
			f3->m_twin[1] = f1Node;
			f3->m_twin[2] = f2Node;
	
			List<ndNode*> boundaryFaces;
			boundaryFaces.Append(f0Node);
			boundaryFaces.Append(f1Node);
			boundaryFaces.Append(f2Node);
			boundaryFaces.Append(f3Node);

			m_points.resize(count);

			count -= 4;
			maxVertexCount -= 4;
			int currentIndex = 4;

			std::vector<ndNode*> stackPool;
			std::vector<ndNode*> coneListPool;
			std::vector<ndNode*> deleteListPool;

			stackPool.resize(1024 + count);
			coneListPool.resize(1024 + count);
			deleteListPool.resize(1024 + count);

			ndNode** const stack = &stackPool[0];
			ndNode** const coneList = &stackPool[0];
			ndNode** const deleteList = &deleteListPool[0];

			while (boundaryFaces.GetCount() && count && (maxVertexCount > 0))
			{
				// my definition of the optimal convex hull of a given vertex count,
				// is the convex hull formed by a subset of the input vertex that minimizes the volume difference
				// between the perfect hull formed from all input vertex and the hull of the sub set of vertex.
				// When using a priority heap this algorithms will generate the an optimal of a fix vertex count.
				// Since all Newton's tools do not have a limit on the point count of a convex hull, I can use either a stack or a queue.
				// a stack maximize construction speed, a Queue tend to maximize the volume of the generated Hull approaching a perfect Hull.
				// For now we use a queue.
				// For general hulls it does not make a difference if we use a stack, queue, or a priority heap.
				// perfect optimal hull only apply for when build hull of a limited vertex count.
				//
				// Also when building Hulls of a limited vertex count, this function runs in constant time.
				// yes that is correct, it does not makes a difference if you build a N point hull from 100 vertex
				// or from 100000 vertex input array.

				// using a queue (some what slower by better hull when reduced vertex count is desired)
				bool isvalid;
				ndNode* const faceNode = boundaryFaces.GetLast()->GetInfo();
				ConvexHullFace* const face = &faceNode->GetInfo();
				hullPlane planeEquation(face->GetPlaneEquation(&m_points[0], isvalid));

				int index = 0;
				double dist = 0;
				hullVector p;
				if (isvalid)
				{
					index = SupportVertex(&vertexTree, points, planeEquation);
					p = points[index];
					dist = planeEquation.Evalue(p);
				}

				if (isvalid && (dist >= distTol) && (face->Evalue(&m_points[0], p) > double(0.0f)))
				{
					stack[0] = faceNode;

					int stackIndex = 1;
					int deletedCount = 0;

					while (stackIndex)
					{
						stackIndex--;
						ndNode* const node1 = stack[stackIndex];
						ConvexHullFace* const face1 = &node1->GetInfo();
			
						if (!face1->m_mark && (face1->Evalue(&m_points[0], p) > double(0.0f)))
						{
							#ifdef _DEBUG
							for (int i = 0; i < deletedCount; ++i)
							{
								assert(deleteList[i] != node1);
							}
							#endif
			
							deleteList[deletedCount] = node1;
							deletedCount++;
							assert(deletedCount < int(deleteListPool.size()));
							face1->m_mark = 1;
							for (int i = 0; i < 3; ++i)
							{
								ndNode* const twinNode = face1->m_twin[i];
								assert(twinNode);
								ConvexHullFace* const twinFace = &twinNode->GetInfo();
								if (!twinFace->m_mark)
								{
									stack[stackIndex] = twinNode;
									stackIndex++;
									assert(stackIndex < int(stackPool.size()));
								}
							}
						}
					}
			
					m_points[currentIndex] = points[index];
					points[index].m_mark = 1;
			
					int newCount = 0;
					for (int i = 0; i < deletedCount; ++i)
					{
						ndNode* const node1 = deleteList[i];
						ConvexHullFace* const face1 = &node1->GetInfo();
						assert(face1->m_mark == 1);
						for (int j0 = 0; j0 < 3; j0++)
						{
							ndNode* const twinNode = face1->m_twin[j0];
							ConvexHullFace* const twinFace = &twinNode->GetInfo();
							if (!twinFace->m_mark)
							{
								int j1 = (j0 == 2) ? 0 : j0 + 1;
								ndNode* const newNode = AddFace(currentIndex, face1->m_index[j0], face1->m_index[j1]);
								boundaryFaces.Addtop(newNode);
			
								ConvexHullFace* const newFace = &newNode->GetInfo();
								newFace->m_twin[1] = twinNode;
								for (int k = 0; k < 3; k++)
								{
									if (twinFace->m_twin[k] == node1)
									{
										twinFace->m_twin[k] = newNode;
									}
								}
								coneList[newCount] = newNode;
								newCount++;
								assert(newCount < int(coneListPool.size()));
							}
						}
					}
			
					for (int i = 0; i < newCount - 1; ++i)
					{
						ndNode* const nodeA = coneList[i];
						ConvexHullFace* const faceA = &nodeA->GetInfo();
						assert(faceA->m_mark == 0);
						for (int j = i + 1; j < newCount; j++) 
						{
							ndNode* const nodeB = coneList[j];
							ConvexHullFace* const faceB = &nodeB->GetInfo();
							assert(faceB->m_mark == 0);
							if (faceA->m_index[2] == faceB->m_index[1])
							{
								faceA->m_twin[2] = nodeB;
								faceB->m_twin[0] = nodeA;
								break;
							}
						}
			
						for (int j = i + 1; j < newCount; j++)
						{
							ndNode* const nodeB = coneList[j];
							ConvexHullFace* const faceB = &nodeB->GetInfo();
							assert(faceB->m_mark == 0);
							if (faceA->m_index[1] == faceB->m_index[2])
							{
								faceA->m_twin[0] = nodeB;
								faceB->m_twin[2] = nodeA;
								break;
							}
						}
					}
			
					for (int i = 0; i < deletedCount; ++i)
					{
						ndNode* const node = deleteList[i];
						boundaryFaces.Remove(node);
						Remove(node);
					}

					maxVertexCount--;
					currentIndex++;
					count--;
				}
				else
				{
					boundaryFaces.Remove(faceNode);
				}
			}
			m_points.resize(currentIndex);
		}
	}
}

} // end of juliohull namespace

//***********************************************************************************************
// End of ConvexHull generation code by Julio Jerez <jerezjulio0@gmail.com>
//***********************************************************************************************

namespace juliohull
{
class JulioHullImpl : public JulioHull
{
public:
	JulioHullImpl(void)
	{
	}

	virtual ~JulioHullImpl(void)
	{
	}

	virtual uint32_t computeConvexHull(uint32_t vertexCount,const double *vertices,uint32_t maxHullVertices,double distanceTolerance) final
	{
		nd::juliohull::ConvexHull ch(vertices,sizeof(double)*3,vertexCount,distanceTolerance,maxHullVertices);

		mTriangleCount = 0;
        mIndices.clear();
		mVertices.clear();

        auto &vlist = ch.GetVertexPool();
        if ( !vlist.empty() )
        {
            size_t vcount = vlist.size();
            mVertices.resize(vcount*3);
            memcpy(&mVertices[0],&vlist[0],sizeof(double)*3*vcount);
        }
        
		for (nd::juliohull::ConvexHull::ndNode* node = ch.GetFirst(); node; node = node->GetNext())
		{
			nd::juliohull::ConvexHullFace* const face = &node->GetInfo();
            mIndices.push_back(face->m_index[0]);
            mIndices.push_back(face->m_index[1]);
            mIndices.push_back(face->m_index[2]);
            mTriangleCount++;
		}

        return mTriangleCount;
	}

	virtual const double *getVertices(uint32_t &vcount) final
	{
        const double *ret = nullptr;

        vcount = (uint32_t)mVertices.size()/3;

        if ( vcount )
        {
            ret = &mVertices[0];
        }

        return ret;
	}

	virtual const uint32_t *getIndices(uint32_t &tcount) final
	{
        const uint32_t *ret = nullptr;
        tcount = mTriangleCount;
        if ( mTriangleCount )
        {
            ret = &mIndices[0];
        }

        return ret;
	}

	virtual void release(void) final
	{
		delete this;
	}

    std::vector< double >   mVertices;
    uint32_t                mTriangleCount{0};
    std::vector<uint32_t>   mIndices;
};

JulioHull *JulioHull::create(void)
{
	auto ret = new JulioHullImpl;
	return static_cast< JulioHull *>(ret);
}


} // end of juliohull namespace

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif


