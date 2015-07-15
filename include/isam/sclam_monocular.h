/**
 * @file slam_monocular.h
 * @brief Provides nodes and factors for monocular vision applications.
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

#include <Eigen/Dense>
#include "Node.h"
#include "Pose3d.h"
#include "Point3d.h"
#include "slam_monocular.h"

/*! \brief SCLAM_Monocular
 * Contains nodes and factors for optimizing single camera intrinsics and relative poses with
 * point features. */

namespace isam
{
	/*! \brief Represents optimizable monocular camera intrinsics. */
	typedef NodeT<MonocularIntrinsics> MonocularIntrinsics_Node;
	
	/*! \brief Represents a prior on monocular camera intrinsics. */
	class Intrinsics_Factor : public FactorT<MonocularIntrinsics> 
	{
		MonocularIntrinsics_Node* _intrinsics;
		
	public:
		
		typedef std::shared_ptr<Intrinsics_Factor> Ptr;
		
		Intrinsics_Factor( MonocularIntrinsics_Node* intrinsics, const MonocularIntrinsics& prior, 
						   const Noise& noise )
		: FactorT<MonocularIntrinsics>("Intrinsics_Factor", 4, noise, prior), _intrinsics(intrinsics) 
		{
			_nodes.resize(1);
			_nodes[0] = intrinsics;
		}
		
		void initialize() 
		{
			if( !_intrinsics->initialized()) 
			{
				MonocularIntrinsics predict = _measure;
				_intrinsics->init( predict );
			}
		}
		
		Eigen::VectorXd basic_error( Selector s = ESTIMATE ) const 
		{
			Eigen::VectorXd err = _nodes[0]->vector(s) - _measure.vector();
			return err;
		}
	};
	
	/*! \brief General monocular camera calibration factor. Can handle hand-eye extrinsics calibrations
	 * as well as feature-structure calibration. */
	class Monocular_Calibration_Factor : public Monocular_Factor_Base
	{
		Pose3d_Node* _cam_ref;			// Pose of the camera reference frame
		Pose3d_Node* _point_ref;		// Pose of the point reference frame
		Pose3d_Node* _cam_ext;			// Relative pose of the camera
		Point3d_Node* _point_ext;		// Position of the point feature
		MonocularIntrinsics_Node* _cam_int;	// Intrinsics of the camera
		
	public:
		
		struct Properties
		{
			bool optimizePose;			// Camera reference pose
			bool optimizeCamExtrinsics;	// Camera extrinsics
			bool optimizeCamIntrinsics;	// Camera intrinsics
			bool optimizeLocation;		// Feature reference frame pose
			bool optimizeStructure;		// Feature relative position
			
			Properties() : optimizePose( true ), optimizeCamExtrinsics( true ), optimizeCamIntrinsics( true ), 
				optimizeLocation( true ), optimizeStructure( true ) {}
		};
		
		typedef std::shared_ptr<Monocular_Calibration_Factor> Ptr;
		
		Monocular_Calibration_Factor( Pose3d_Node* cam_ref, Pose3d_Node* point_ref,
									 Pose3d_Node* cam_ext, Point3d_Node* point_ext, 
									 MonocularIntrinsics_Node* cam_int,
									 const MonocularMeasurement& measure, const isam::Noise& noise,
									 Properties prop = Properties() )
		: Monocular_Factor_Base("Monocular_Calibration_Factor", noise, measure),
		_cam_ref( cam_ref ), _point_ref( point_ref ), _cam_ext( cam_ext ), _point_ext( point_ext ),
		_cam_int( cam_int )
		{
			if( prop.optimizePose ) 			{ _nodes.push_back( cam_ref ); }
			if( prop.optimizeCamExtrinsics ) 	{ _nodes.push_back( cam_ext ); }
			if( prop.optimizeCamIntrinsics ) 	{ _nodes.push_back( cam_int ); }
			if( prop.optimizeLocation ) 		{ _nodes.push_back( point_ref ); }
			if( prop.optimizeStructure ) 		{ _nodes.push_back( point_ext ); }
			if( _nodes.size() == 0 )
			{
				throw std::runtime_error( "Monocular_Calibration_Factor created with no optimization variables." );
			}
		}
		
		void initialize() {
			require( _cam_ref->initialized() && _point_ref->initialized() && _cam_ext->initialized() &&
					_point_ext->initialized() && _cam_int->initialized(),
					 "Monocular_Calibration_Factor requires all nodes to be initialized" );
		}
		
		virtual Eigen::Matrix<double, 3, 4> projectionMatrix(Selector s = ESTIMATE) const 
		{
			const Pose3d& cref = _cam_ref->value(s);
			const Pose3d& pref = _point_ref->value(s);
			const Pose3d& cext = _cam_ext->value(s);
			
			Eigen::Matrix<double, 3, 4> P = _cam_int->value(s).K() 
				* cext.oTw() * cref.oTw() * pref.wTo();
			return P;
		}
		
		Eigen::VectorXd basic_error(Selector s = ESTIMATE) const 
		{
			Point3d point = _point_ext->value( s );
			return image_error( point, s );
		}
	};
}
