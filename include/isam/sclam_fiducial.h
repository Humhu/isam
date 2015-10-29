/**
 * @file sclam_fiducial.h
 * @brief Provides nodes and factors for monocular fiducial applications.
 * @author Humphrey Hu
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

#include "isam/sclam_monocular.h"

namespace isam
{

/*! \brief Represents a fiducial as a uniquely ordered group of points in space.
 * The points are ordered uniquely and specified in the fiducial frame of reference. */
class FiducialIntrinsics
{
public:
	
	typedef Eigen::Matrix <double, 3, 1> PointType;
	typedef Eigen::Matrix <double, Eigen::Dynamic, 1> VectorType;
	typedef Eigen::Matrix <double, 3, Eigen::Dynamic> MatrixType;
	
	// We have fixed-size Eigen members
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	
	static const char* name() { return "FiducialIntrinsics"; }
	
	/*! \brief Construct a fiducial from a vector of points. */
	FiducialIntrinsics( const std::vector <isam::Point3d>& pts )
	{
		if( points.size() == 0 )
		{
			throw std::runtime_error( "Cannot create fiducial with zero points." );
		}
		
		points = VectorType( 3 * pts.size() );
		for( unsigned int i = 0; i < pts.size(); i++ )
		{
			points.block<3,1>( 3*i, 0 ) = PointType( pts[i].x(), pts[i].y(), pts[i].z() );
		}
	}
	
	FiducialIntrinsics( const VectorType& v )
	: points( v ) {}
	
	/*! \brief Create a new fiducial by applying a small change to this fiducial. 
	 * The change should be aggregated X Y and Z for each point in order. */
	FiducialIntrinsics exmap( const Eigen::VectorXd& delta )
	{
		if( delta.rows() != points.rows() )
		{
			throw std::runtime_error( "Cannot apply delta of size " 
			    + std::to_string( delta.rows() ) + " to fiducial of size " 
			    + std::to_string( points.rows() ) );
		}
		return FiducialIntrinsics( points + delta );
	}
	
	/*! \brief Returns the dimensionality, equal to 3 times the number of points. */
	int dim() const
	{
		return points.rows();
	}
	
	/*! \brief Set the fiducial point positions. */
	void set( const Eigen::VectorXd& v )
	{
		if( v.rows() != points.rows() )
		{
			throw std::runtime_error( "Cannot set arg of size " 
			    + std::to_string( v.rows() ) + " to fiducial of size " 
			    + std::to_string( points.rows() ) );
		}
		points = v;
	}
	
	/*! \brief Returns boolean vector indicating angle elements. */
	Eigen::VectorXb is_angle() const
	{
		return Eigen::VectorXb::Constant( points.rows()/3, 1, false );
	}
	
	MatrixType matrix() const
	{
		Eigen::Map <const MatrixType> mMap( points.data(), 3, points.rows() );
		return MatrixType( mMap );
	}
	
	/*! \brief Returns aggregated X Y and Z point coordinates. */
	Eigen::VectorXd vector() const
	{
		return points;
	}
	
	void write( std::ostream& out ) const
	{
		out << "(points: " << points.transpose() << ")";
	}
	
private:

	Eigen::VectorXd points;
	
};

std::ostream& operator<<( std::ostream& out, const FiducialIntrinsics& in )
{
	in.write( out );
	return out;
}

typedef NodeT <FiducialIntrinsics> FiducialIntrinsics_Node;

/*! \brief Provides a prior on a fiducial intrinsic. */
class FiducialIntrinsics_Prior
: public FactorT <FiducialIntrinsics>
{
public:
	
	typedef std::shared_ptr <FiducialIntrinsics_Prior> Ptr;
	typedef FactorT <FiducialIntrinsics> FactorType;
	
	FiducialIntrinsics_Prior( FiducialIntrinsics_Node* fiducial,
	                          const FiducialIntrinsics& prior,
	                          const Noise& noise )
	: FactorType( "FiducialIntrinsics_Prior", prior.dim(), noise, prior ),
	_fiducial( fiducial )
	{
		_nodes.resize( 1 );
		_nodes[ 0 ] = fiducial;
	}
	
	void initialize()
	{
		if( !_fiducial->initialized() )
		{
			_fiducial->init( _measure );
		}
	}
	
	Eigen::VectorXd basic_error( Selector s = ESTIMATE ) const
	{
		return _fiducial->vector( s ) - _measure.vector();
	}
	
private:
	
	FiducialIntrinsics_Node* _fiducial;
	
};

/*! \brief Represents the image or camera coordinate detections of a fiducial. 
 * The points are expected to be ordered uniquely such that naively comparing
 * detections is appropriate. */
class FiducialDetection
{
public:
	
	/*! \brief Construct a detection from aggregated point x y. */
	FiducialDetection( const Eigen::VectorXd& p )
	{
		if( p.rows() % 2 != 0 )
		{
			throw std::runtime_error( "FiducialDetection: Uneven vector size." );
		}
		points = p;
	}
	
	/*! \brief Return the total dimensionality equal to the number of points times 2. */
	int dim() const
	{
		return points.rows();
	}
	
	/*! \brief Return the detection in aggregate vector form. */
	Eigen::VectorXd vector() const
	{
		return points;
	}
	
	void write( std::ostream& out ) const
	{
		out << "(points: " << points.transpose() << ")";
	}
	
private:
	
	Eigen::VectorXd points;
	
};

std::ostream& operator<<( std::ostream& out, const FiducialDetection& det )
{
	det.write( out );
	return out;
}

FiducialDetection Predict( const FiducialIntrinsics& fiducial,
                           const MonocularIntrinsics& camera,
                           const Pose3d& fiducialToCamera )
{
	
	Eigen::Transform <double, 2, Eigen::Affine> cameraMatrix( camera.K().block<2,2>(0,0) );
	Eigen::Transform <double, 3, Eigen::Isometry> relPose( fiducialToCamera.oTw() );
	
	Eigen::Matrix <double, 3, Eigen::Dynamic> relPoints = relPose * fiducial.matrix();
	Eigen::Matrix <double, 2, Eigen::Dynamic> imgPoints = (cameraMatrix * relPoints).colwise().hnormalized();
	
	
	return FiducialDetection( Eigen::Map <Eigen::VectorXd> ( imgPoints.data(), 2 * imgPoints.cols() ) );
}

/*! \brief Factor that allows full camera and fiducial intrinsic and extrinsic
 * calibration and estimation.*/
class FiducialFactor 
: public FactorT <FiducialDetection>
{
public:

	struct Properties
	{
		bool optCamReference;
		bool optCamIntrinsics;
		bool optCamExtrinsics;
		bool optFidReference;
		bool optFidIntrinsics;
		bool optFidExtrinsics;
	};
	
	typedef FactorT <FiducialDetection> FactorType;
	
	FiducialFactor( Pose3d_Node* camRef, MonocularIntrinsics_Node* camInt,
	                Pose3d_Node* camExt, Pose3d_Node* fidRef,
	                FiducialIntrinsics_Node* fidInt, Pose3d_Node* fidExt,
	                const FiducialDetection& detection, const Noise& noise,
	                Properties prop )
	: FactorType( "FiducialFactor", detection.dim(), noise, detection ),
	_cam_ref( camRef ), _cam_int( camInt ), _cam_ext( camExt ),
	_fid_ref( fidRef ), _fid_int( fidInt ), _fid_ext( fidExt ),
	properties( prop )
	{
		if( properties.optCamReference ) { _nodes.push_back( _cam_ref ); }
		if( properties.optCamIntrinsics ) { _nodes.push_back( _cam_int ); }
		if( properties.optCamExtrinsics ) { _nodes.push_back( _cam_ext ); }
		if( properties.optFidReference ) { _nodes.push_back( _fid_ref ); }
		if( properties.optFidIntrinsics ) { _nodes.push_back( _fid_int ); }
		if( properties.optFidExtrinsics ) { _nodes.push_back( _fid_ext ); }
		require( _nodes.size() > 0, "FiducialFactor created with no optimization variables." );
	}
	
	void initialize()
	{
		require( _cam_ref->initialized() && _cam_int->initialized() && 
		         _cam_ext->initialized() && _fid_ref->initialized() &&
		         _fid_int->initialized() && _fid_ext->initialized(),
		         "FiducialFactor requires all nodes to be initialized." );
	}
	
	Eigen::VectorXd basic_error( Selector s = ESTIMATE ) const
	{
		Pose3d relPose( _cam_ext->value(s).oTw() * _cam_ref->value(s).oTw() *
		                _fid_ref->value(s).wTo() * _fid_ext->value(s).wTo() );
		FiducialDetection predicted = Predict( _fid_int->value(s), _cam_int->value(s), relPose ).vector();
		return predicted.vector() - _measure.vector();
	}
	
private:
	
	Pose3d_Node* _cam_ref;
	MonocularIntrinsics_Node* _cam_int;
	Pose3d_Node* _cam_ext;
	Pose3d_Node* _fid_ref;
	FiducialIntrinsics_Node* _fid_int;
	Pose3d_Node* _fid_ext;
	Properties properties;
	
};
	
}
