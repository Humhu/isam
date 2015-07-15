/**
 * @file tagSlam.h
 * @brief Data structures and iSAM factor for Apriltags
 * @author: Michael Kaess
 * @modified: Humphrey Hu
 */

#pragma once

#include <Eigen/Dense>

#include "isam.h"
#include "slam_monocular.h"
#include "sclam_monocular.h"

namespace isam {
	
	// TODO Add code, ID, family?
	class TagIntrinsics {
	public:
		
		static const int dim = 1;
		static const char* name() {
			return "TagIntrinsics";
		}
		
		typedef std::shared_ptr<TagIntrinsics> Ptr;
		
		double tagSize; // Dimension of one side
		
		TagIntrinsics()
			: tagSize( 1.0 )
		{}
		
		TagIntrinsics( double ts )
			: tagSize( ts )
		{}
		
		TagIntrinsics exmap( const Eigen::VectorXd& delta )
		{
			Eigen::VectorXd current = vector();
			current += delta;
			return TagIntrinsics( current(0) );
		}
		
		void set( const Eigen::VectorXd& v )
		{
			tagSize = v(0);
		}
		
		Eigen::VectorXb is_angle() const
		{
			Eigen::VectorXb isang( dim );
			isang << false;
			return isang;
		}
		
		Eigen::VectorXd vector() const
		{
			Eigen::VectorXd vec(1);
			vec << tagSize;
			return vec;
		}
		
		void write( std::ostream& out ) const
		{
			out << "(size: " << tagSize << ")";
		}
		
	};
	
	std::ostream& operator<<( std::ostream& out, const TagIntrinsics& in )
	{
		in.write( out );
		return out;
	}
	
	typedef NodeT<TagIntrinsics> TagIntrinsics_Node;

	class Tag_Intrinsics_Factor : public FactorT<TagIntrinsics> {
		TagIntrinsics_Node* _tag;
		
	public:
		
		typedef std::shared_ptr<Tag_Intrinsics_Factor> Ptr;
		
		Tag_Intrinsics_Factor( TagIntrinsics_Node* tag, const TagIntrinsics& prior, const Noise& noise )
		: FactorT<TagIntrinsics>( "Tag_Intrinsics_Factor", 1, noise, prior ), _tag( tag ) {
			_nodes.resize(1);
			_nodes[0] = tag;
		}
		
		void initialize() {
			if( !_tag->initialized()) {
				TagIntrinsics predict = _measure;
				_tag->init( predict );
			}
		}
		
		Eigen::VectorXd basic_error( Selector s = ESTIMATE ) const {
			Eigen::VectorXd err = _nodes[0]->vector(s) - _measure.vector();
			return err;
		}
	};
	
	
	// Apriltag detection: image coordinates of four corners in fixed order
	class TagCorners {
	public:
		isam::Point2d bl, br, tr, tl;

		TagCorners()
			: bl ( 0, 0 ), br( 0, 0 ), tr( 0, 0 ), tl( 0, 0 )
		{}
		
		TagCorners( isam::Point2d bl, isam::Point2d br, isam::Point2d tr, 
						 isam::Point2d tl ) 
			: bl(bl), br(br), tr(tr), tl(tl) 
		{}

		friend std::ostream& operator<<( std::ostream& out, const TagCorners& t ) {
			out << "(" << t.bl << " ; " << t.br << " ; " << t.tr << " ; " << t.tl
				<< ")";
			return out;
		}

		Eigen::VectorXd vector() const {
			Eigen::VectorXd ret(8);
			ret << bl.vector(), br.vector(), tr.vector(), tl.vector();
			return ret;
		}
	};
	
	class Tag_Factor_Base : public FactorT<TagCorners> {
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		typedef std::shared_ptr<Tag_Factor_Base> Ptr;
		
		Eigen::Matrix4d _rot; // Moves tag to x-forward

		Tag_Factor_Base( const char* type, int size, 
						 const Noise& noise, const TagCorners& measure )
			: FactorT<TagCorners>( type, size, noise, measure )
		{
			_rot.setZero();
			_rot(0, 1) = -1;
			_rot(1, 2) = -1;
			_rot(2, 0) = 1;
			_rot(3, 3) = 1;
		}
		
		virtual ~Tag_Factor_Base() {}
		
		virtual Eigen::VectorXd basic_error(Selector s = ESTIMATE) const {
			TagCorners predicted = predict( s );
			Eigen::VectorXd err = predicted.vector() - _measure.vector();
			return err;
		}

		virtual TagCorners predict(Selector s = ESTIMATE) const {
			Eigen::Matrix<double, 3, 4> P = projectionMatrix( s );
				
			double h = tagSize( s )/2.0;
			TagCorners predicted(project(P, -h, -h), project(P, h, -h), project(P, h, h),
				project(P, -h, h));
			
			return predicted;
		}
		
		// NOTE x-forward convention here for camera
		virtual Point2d project( const Eigen::Matrix<double, 3, 4>& P, double u, double v) const {
			Eigen::Vector4d corner(0.0, u, v, 1.0);
			Eigen::Vector3d x;
			x = P * corner;
			return Point2d(x(0) / x(2), x(1) / x(2));
		}
		
		// Override for derived factors
		virtual Eigen::Matrix<double, 3, 4> projectionMatrix(Selector s = ESTIMATE) const = 0;
		virtual double tagSize(Selector s = ESTIMATE) const = 0;
	};
	
	/*! \brief Factor to estimate robot pose, tag intrinsics/extrinsics, and camera intrinsics/extrinsics. */
	class Tag_Calibration_Factor: public Tag_Factor_Base {
		Pose3d_Node* _cam_ref;				// Pose of the camera reference frame
		Pose3d_Node* _tag_pose;					// Pose of the tag
		Pose3d_Node* _cam_ext;				// Relative pose of the camera
		MonocularIntrinsics_Node* _cam_int;	// Intrinsics of the camera
		TagIntrinsics_Node* _tag_int;					// Intrinsics of the tag
		
	public:

		struct Properties
		{
			bool optimizePose;			// Camera reference pose
			bool optimizeCamExtrinsics;	// Camera extrinsics
			bool optimizeCamIntrinsics;	// Camera intrinsics
			bool optimizeTagLocation;	// Tag pose
			bool optimizeTagParameters;	// Tag size
			
			Properties() : optimizePose( true ), optimizeCamExtrinsics( true ), optimizeCamIntrinsics( true ), 
				optimizeTagLocation( true ), optimizeTagParameters( true ) {}
		};
		
		typedef std::shared_ptr<Tag_Calibration_Factor> Ptr;

		Tag_Calibration_Factor( Pose3d_Node* cam_ref, Pose3d_Node* tag, 
								Pose3d_Node* cam_ext, MonocularIntrinsics_Node* cam_int,
								TagIntrinsics_Node* tag_int,
								const TagCorners& measure, const Noise& noise,
								Properties prop = Properties() ) 
		: Tag_Factor_Base("Tag_Calibration_Factor", 8, noise, measure), 
			_cam_ref( cam_ref ), _tag_pose( tag ), _cam_ext( cam_ext ), _cam_int( cam_int ), 
			_tag_int( tag_int )
		{
			if( prop.optimizePose ) 			{ _nodes.push_back( cam_ref ); }
			if( prop.optimizeCamExtrinsics ) 	{ _nodes.push_back( cam_ext ); }
			if( prop.optimizeCamIntrinsics ) 	{ _nodes.push_back( cam_int ); }
			if( prop.optimizeTagLocation )		{ _nodes.push_back( tag ); }
			if( prop.optimizeTagParameters )	{ _nodes.push_back( tag_int ); }
			require( _nodes.size() > 0,
					 "Tag_Calibration_Factor created with no optimization variables." );
		}

		void initialize() {
			require( _cam_ref->initialized() && _tag_pose->initialized() && _cam_ext->initialized() &&
					 _cam_int->initialized() && _tag_int->initialized(), 
					 "Tag_Calibration_Factor requires all nodes to be initialized." );
		}

		virtual Eigen::Matrix<double,3,4> projectionMatrix(Selector s = ESTIMATE) const {
			const Pose3d& bot = _cam_ref->value(s);
			const Pose3d& tag = _tag_pose->value(s);
			const Pose3d& ext = _cam_ext->value(s);
			Eigen::Matrix<double, 3, 4> P = _cam_int->value(s).K() * _rot
				* ext.oTw() * bot.oTw() * tag.wTo() ;
			return P;
		}
			
		virtual double tagSize(Selector s = ESTIMATE) const {
			return _tag_int->value(s).tagSize;
		}
	}; 

}
