/**
 * @file slam_monocular.h
 * @brief Provides nodes and factors for monocular vision applications.
 * @author Michael Kaess
 * @version $Id: slam_monocular.h 9316 2013-11-18 20:26:11Z kaess $
 *
 * Copyright (C) 2009-2013 Massachusetts Institute of Technology.
 * Michael Kaess, Hordur Johannsson, David Rosen,
 * Nicholas Carlevaris-Bianco and John. J. Leonard
 *
 * This file is part of iSAM.
 *
 * iSAM is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the
 * Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * iSAM is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 * License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with iSAM.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#pragma once

#include <string>
#include <sstream>
#include <math.h>
#include <Eigen/Dense>

#include "isam/Node.h"
#include "isam/Factor.h"
#include "isam/Pose3d.h"
#include "isam/Point3dh.h"
#include "isam/slam3d.h"

/*! \brief SLAM_Monocular 
 * Contains nodes and factors for optimizing single camera trajectories with point features. */

namespace isam 
{
	
	/*! \brief Represents the observation of a point by a camera. */
	class MonocularMeasurement 
	{
		friend std::ostream& operator<<(std::ostream& out, const MonocularMeasurement& t) 
		{
			t.write(out);
			return out;
		}
		
	public:
		
		double u;
		double v;
		bool valid; // Denotes whether the measurement is in front of the camera plane
		
		MonocularMeasurement(double u, double v) : u(u), v(v), valid(true) {}
		MonocularMeasurement(double u, double v, bool valid) : u(u), v(v), valid(valid) {}
		
		Eigen::Vector2d vector() const 
		{
			Eigen::Vector2d tmp(u, v);
			return tmp;
		}
		
		void write(std::ostream &out) const 
		{
			out << "(" << u << ", " << v << ")";
		}
	};
	
	/*! \brief Represents the intrinsic parameters of a monocular camera. Uses a
	 * simple pinhole model with no distortion.*/
	class MonocularIntrinsics 
	{
		double _fx, _fy;
		Eigen::Vector2d _pp;
		
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		static const int dim = 2;
		static const char* name() 
		{
			return "MonocularIntrinsics";
		}
		
		typedef std::shared_ptr<MonocularIntrinsics> Ptr;
		
		MonocularIntrinsics() 
			: _fx(1), _fy(1), _pp(Eigen::Vector2d(0.5,0.5)) 
		{}
		
		MonocularIntrinsics(double fx, double fy, const Eigen::Vector2d& pp) 
			: _fx(fx), _fy(fy), _pp(pp) 
		{}
		
		double fx() const { return _fx; }
		double fy() const { return _fy; }
		
		Eigen::Vector2d principalPoint() const { return _pp; }
		
		Eigen::Matrix<double, 3, 4> K() const 
		{
			Eigen::Matrix<double, 3, 4> K;
			K.row(0) << _fx, 0, _pp(0), 0;
			K.row(1) << 0, _fy, _pp(1), 0;
			K.row(2) << 0, 0, 1, 0;
			return K;
		}
		
		MonocularIntrinsics exmap( const Eigen::VectorXd& delta )
		{
			Eigen::VectorXd current = vector();
			current += delta;
			Eigen::Vector2d pp;
			pp << current(2), current(3);
			return MonocularIntrinsics( current(0), current(1), pp );
		}
	
		void set( const Eigen::VectorXd& v )
		{
			_fx = v(0);
			_fy = v(1);
			_pp(0) = v(2);
			_pp(1) = v(3);
		}
		
		Eigen::VectorXb is_angle() const {
			Eigen::VectorXb isang( dim );
			isang << false, false, false, false;
			return isang;
		}
	
		Eigen::VectorXd vector() const
		{
			Eigen::VectorXd vec(4);
			vec << _fx, _fy, _pp(0), _pp(1);
			return vec;
		}
		
		void write( std::ostream& out ) const
		{
			out << "(fx: " << _fx << " fy: " << _fy << "pp: [" << _pp(0) << ", " << _pp(1) << "])";
		}
	};
	
	std::ostream& operator<<( std::ostream& out, const MonocularIntrinsics& intrinsics )
	{
		intrinsics.write( out );
		return out;
	}
	
	class Monocular_Factor_Base : public FactorT<MonocularMeasurement>
	{
	public:
	
		typedef std::shared_ptr<Monocular_Factor_Base> Ptr;
		
		Monocular_Factor_Base( const char* type, const Noise& noise, const MonocularMeasurement& measure )
			: FactorT<MonocularMeasurement>( type, 2, noise, measure ) {}
			
		virtual ~Monocular_Factor_Base() {}
			
		virtual Eigen::VectorXd image_error( const Point3d& p, Selector s = ESTIMATE ) const
		{
			Eigen::Matrix<double, 3, 4> P = projectionMatrix( s );
			MonocularMeasurement predicted = project( P, p );
			if( predicted.valid )
			{
				return predicted.vector() - _measure.vector();
			}
			return Eigen::VectorXd::Zero( 2 ); // Nullifies points behind camera
		}
			
		static MonocularMeasurement project( const Eigen::Matrix<double,3,4>& P, Point3d p )
		{
			Eigen::Vector4d point( p.x(), p.y(), p.z(), 1.0 );
			Eigen::Vector3d x;
			x = P * point;
			bool valid = x(2) > 0;
			if( valid )
			{
				return MonocularMeasurement( x(0) / x(2), x(1) / x(2), true );
			}
			return MonocularMeasurement( 0, 0, false );
		}
		
		virtual Eigen::Matrix<double, 3, 4> projectionMatrix( Selector s = ESTIMATE ) const = 0;
	};
	
	/*! \brief Monocular observation of a 3D point. Projective or Euclidean 
	 * geometry depending on constructor used. */
	class Monocular_Factor : public Monocular_Factor_Base 
	{
		Pose3d_Node* _cam;				// Pose of the camera reference frame
		Point3d_Node* _point;			// Position of the feature
		MonocularIntrinsics _intrinsics;	// Intrinsics of the camera
		
	public:
		
		typedef std::shared_ptr<Monocular_Factor> Ptr;
		
		Monocular_Factor( Pose3d_Node* cam, Point3d_Node* point, const MonocularIntrinsics& intrinsics,
						  const MonocularMeasurement& measure, const isam::Noise& noise )
			: Monocular_Factor_Base( "Monocular_Factor", noise, measure ),
			_cam( cam ), _point( point ), _intrinsics( intrinsics ) 
			{
			_nodes.resize(2);
			_nodes[0] = cam;
			_nodes[1] = point;
		}
		
		void initialize() 
		{
			require(_cam->initialized(), "Monocular_Factor requires pose to be initialized.");
			require( _point->initialized(), "Monocular_Factor requires point to be initialized." );
		}
		
		virtual Eigen::Matrix<double, 3, 4> projectionMatrix( Selector s = ESTIMATE ) const 
		{
			const Pose3d& cam = _cam->value(s);
			Eigen::Matrix<double, 3, 4> P = _intrinsics.K() * cam.oTw() ;
			return P;
		}
		
		Eigen::VectorXd basic_error( Selector s = ESTIMATE ) const 
		{
			Point3d point = _point->value( s );
			return image_error( point, s );
		}
		
	};
	
}
