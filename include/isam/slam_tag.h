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

namespace isam {
	
	// TODO Add code, ID, family?
	class TagIntrinsics {
	public:
		
		typedef std::shared_ptr<TagIntrinsics> Ptr;
		
		double tagSize; // Dimension of one side
		double halfSize;
		
		TagIntrinsics()
			: tagSize( 1.0 ), halfSize( 0.5 )
		{}
		
		TagIntrinsics( double ts )
			: tagSize( ts ), halfSize( tagSize/2.0 )
		{}
		
	};
	
	// Apriltag detection: image coordinates of four corners in fixed order
	class TagCorners {
	public:
		isam::Point2d bl, br, tr, tl;

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

	class Tag_Factor : public FactorT<TagCorners> {
		Pose3d_Node* _pose;
		Pose3d_Node* _tag;
		TagIntrinsics* _tagIntrinsics;
		MonocularIntrinsics* _camera;
	};
	
	/*! \brief Factor to estimate robot pose, tag pose, and camera extrinsics. */
	class Tag_Extrinsics_Factor: public FactorT<TagCorners> {
		Pose3d_Node* _pose;
		Pose3d_Node* _tag;
		TagIntrinsics* _tagIntrinsics;
		Pose3d_Node* _extrinsics;
		MonocularIntrinsics* _camera;
		Eigen::Matrix4d _rot; // Moves tag to x-forward

	public:

		typedef std::shared_ptr<Tag_Extrinsics_Factor> Ptr;

		/**
		* Constructor.
		* @param pose The pose of the robot observing the tag in the world frame.
		* @param tag The pose of the tag in the world frame.
		* @param tagIntrinsics The intrinsics of the tag being observed.
		* @param cameraExtrinsics The pose of the camera observing the tag in the robot frame.
		* @param camera The param struct of the observing camera.
		* @param measure The measurements of the corners of the tag
		* @param noise The 8x8 noise matrix.
		*/
		Tag_Extrinsics_Factor( Pose3d_Node* pose, 
							   Pose3d_Node* tag, TagIntrinsics* tagIntrinsics,
							   Pose3d_Node* cameraExtrinsics, MonocularIntrinsics* camera, 
							   const TagCorners& measure, const Noise& noise ) 
			: FactorT<TagCorners>("Tag_Extrinsics_Factor", 8, noise, measure), 
				_pose( pose ), _tag( tag ), _tagIntrinsics( tagIntrinsics ),
				_extrinsics( cameraExtrinsics ), _camera( camera )
		{
			_nodes.resize(3);
			_nodes[0] = _pose;
			_nodes[1] = _tag;
 			_nodes[2] = _extrinsics;

			_rot.setZero();
			_rot(0, 1) = -1;
			_rot(1, 2) = -1;
			_rot(2, 0) = 1;
			_rot(3, 3) = 1;
		}

		void initialize() {
			require(_tag->initialized(),
				"Tag_Extrinsics_Factor requires tag to be initialized");
			require(_pose->initialized(),
				"Tag_Extrinsics_Factor requires pose to be initialized");
			require( _extrinsics->initialized(),
				"Tag_Extrinsics_Factor requires camera extrinsics to be initialized");
			require( _tagIntrinsics != nullptr,
				"Tag_Extrinsics_Factor requires tag intrinsics to be initialized");
			require( _camera != nullptr,
				"Tag_Extrinsics_Factor requires camera intrinsics to be initialized");
		}

		inline Point2d proj( const Eigen::Matrix<double, 3, 4>& P, double u, double v) const {
			Eigen::Vector4d corner(0.0, u, v, 1.);
			Eigen::Vector3d x;
			x = P * corner;
			return Point2d(x(0) / x(2), x(1) / x(2));
		}
		
		Eigen::VectorXd basic_error(Selector s = ESTIMATE) const {
			const Pose3d& bot = _pose->value(s);
			const Pose3d& tag = _tag->value(s);
			const Pose3d& ext = _extrinsics->value(s);
			Eigen::Matrix<double, 3, 4> P = _camera->K() * _rot
				* ext.oTw() * bot.oTw() * tag.wTo() ;

			double h = _tagIntrinsics->halfSize;
			TagCorners predicted(proj(P, -h, -h), proj(P, h, -h), proj(P, h, h),
				proj(P, -h, h));
			Eigen::VectorXd err = predicted.vector() - _measure.vector();
			return err;
		}

		TagCorners predict(Selector s = ESTIMATE) const {
			const Pose3d& bot = _pose->value(s);
			const Pose3d& tag = _tag->value(s);
			const Pose3d& ext = _extrinsics->value(s);
			Eigen::Matrix<double, 3, 4> P = _camera->K() * _rot
				* ext.oTw() * bot.oTw() * tag.wTo() ;
				
			double h = _tagIntrinsics->halfSize;
			TagCorners predicted(proj(P, -h, -h), proj(P, h, -h), proj(P, h, h),
				proj(P, -h, h));
			
			return predicted;
		}
	};
	
	/*! \brief Factor to estimate robot pose, tag pose, and camera intrinsics/extrinsics. */
	class Tag_Intrinsics_Extrinsics_Factor: public FactorT<TagCorners> {
		Pose3d_Node* _pose;
		Pose3d_Node* _tag;
		TagIntrinsics* _tagIntrinsics;
		Pose3d_Node* _extrinsics;
		Intrinsics_Node* _cameraIntrinsics;
		Eigen::Matrix4d _rot; // Moves tag to x-forward

	public:

		typedef std::shared_ptr<Tag_Intrinsics_Extrinsics_Factor> Ptr;

		/**
		* Constructor.
		* @param pose The pose of the robot observing the tag in the world frame.
		* @param tag The pose of the tag in the world frame.
		* @param tagIntrinsics The intrinsics of the tag being observed.
		* @param cameraExtrinsics The pose of the camera observing the tag in the robot frame.
		* @param camera The param struct of the observing camera.
		* @param measure The measurements of the corners of the tag
		* @param noise The 8x8 noise matrix.
		*/
		Tag_Intrinsics_Extrinsics_Factor( Pose3d_Node* pose, 
							   Pose3d_Node* tag, TagIntrinsics* tagIntrinsics,
							   Pose3d_Node* cameraExtrinsics, Intrinsics_Node* cameraIntrinsics, 
							   const TagCorners& measure, const Noise& noise ) 
			: FactorT<TagCorners>("Tag_Intrinsics_Extrinsics_Factor", 8, noise, measure), 
				_pose( pose ), _tag( tag ), _tagIntrinsics( tagIntrinsics ),
				_extrinsics( cameraExtrinsics ), _cameraIntrinsics( cameraIntrinsics )
		{
			_nodes.resize(4);
			_nodes[0] = _pose;
			_nodes[1] = _tag;
 			_nodes[2] = _extrinsics;
			_nodes[3] = _cameraIntrinsics;
			
			_rot.setZero();
			_rot(0, 1) = -1;
			_rot(1, 2) = -1;
			_rot(2, 0) = 1;
			_rot(3, 3) = 1;
		}

		void initialize() {
			require(_tag->initialized(),
				"Tag_Intrinsics_Extrinsics_Factor requires tag to be initialized");
			require(_pose->initialized(),
				"Tag_Intrinsics_Extrinsics_Factor requires pose to be initialized");
			require( _extrinsics->initialized(),
				"Tag_Intrinsics_Extrinsics_Factor requires camera extrinsics to be initialized");
			require( _tagIntrinsics != nullptr,
				"Tag_Intrinsics_Extrinsics_Factor requires tag intrinsics to be initialized");
			require( _cameraIntrinsics ->initialized(),
				"Tag_Intrinsics_Extrinsics_Factor requires camera intrinsics to be initialized");
		}

		inline Point2d proj( const Eigen::Matrix<double, 3, 4>& P, double u, double v) const {
			Eigen::Vector4d corner(0.0, u, v, 1.);
			Eigen::Vector3d x;
			x = P * corner;
			return Point2d(x(0) / x(2), x(1) / x(2));
		}
		
		Eigen::VectorXd basic_error(Selector s = ESTIMATE) const {
			const Pose3d& bot = _pose->value(s);
			const Pose3d& tag = _tag->value(s);
			const Pose3d& ext = _extrinsics->value(s);
			Eigen::Matrix<double, 3, 4> P = _cameraIntrinsics->value(s).K() * _rot
				* ext.oTw() * bot.oTw() * tag.wTo() ;

			double h = _tagIntrinsics->halfSize;
			TagCorners predicted(proj(P, -h, -h), proj(P, h, -h), proj(P, h, h),
				proj(P, -h, h));
			Eigen::VectorXd err = predicted.vector() - _measure.vector();
			return err;
		}

		TagCorners predict(Selector s = ESTIMATE) const {
			const Pose3d& bot = _pose->value(s);
			const Pose3d& tag = _tag->value(s);
			const Pose3d& ext = _extrinsics->value(s);
			Eigen::Matrix<double, 3, 4> P = _cameraIntrinsics->value(s).K() * _rot
				* ext.oTw() * bot.oTw() * tag.wTo() ;
				
			double h = _tagIntrinsics->halfSize;
			TagCorners predicted(proj(P, -h, -h), proj(P, h, -h), proj(P, h, h),
				proj(P, -h, h));
			
			return predicted;
		}
	};

}
