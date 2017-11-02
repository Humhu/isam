/**
 * @file slam_monocular.h
 * @brief Provides nodes and factors for monocular vision applications.
 * @author Michael Kaess
 * @modified Humphrey Hu
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
public:
	
	double u;
	double v;
	bool valid; // Denotes whether the measurement is in front of the camera plane
	
	MonocularMeasurement(double u, double v);
	MonocularMeasurement(double u, double v, bool valid);
	
	Eigen::Vector2d vector() const;
	
	void write(std::ostream &out) const;
};

std::ostream& operator<<(std::ostream& out, const MonocularMeasurement& t);

/*! \brief Represents the intrinsic parameters of a monocular camera. Uses a
 * simple pinhole model with no distortion.*/
class MonocularIntrinsics 
{
	double _fx, _fy;
	Eigen::Vector2d _pp;
	
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	static const int dim = 4;
	static const char* name();
	
	typedef std::shared_ptr<MonocularIntrinsics> Ptr;
	
	MonocularIntrinsics();
	
	MonocularIntrinsics(double fx, double fy, const Eigen::Vector2d& pp);
	
	MonocularIntrinsics( double fx, double fy, double px, double py );
	
	double fx() const;
	double fy() const;
	
	Eigen::Vector2d principalPoint() const;
	
	Eigen::Matrix<double, 3, 4> K() const;
	
	MonocularIntrinsics exmap( const Eigen::VectorXd& delta );

	void set( const Eigen::VectorXd& v );
	
	Eigen::VectorXb is_angle() const;

	Eigen::VectorXd vector() const;
	
	void write( std::ostream& out ) const;
};

std::ostream& operator<<( std::ostream& out, const MonocularIntrinsics& intrinsics );

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
