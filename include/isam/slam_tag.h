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
	
	typedef NodeT<TagIntrinsics> Tag_Node;

	class Tag_Intrinsics_Factor : public FactorT<TagIntrinsics> {
		Tag_Node* _tag;
		
	public:
		
		typedef std::shared_ptr<Tag_Intrinsics_Factor> Ptr;
		
		Tag_Intrinsics_Factor( Tag_Node* tag, const TagIntrinsics& prior, const Noise& noise )
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
			TagCorners predicted(proj(P, -h, -h), proj(P, h, -h), proj(P, h, h),
				proj(P, -h, h));
			
			return predicted;
		}
		
		virtual Point2d proj( const Eigen::Matrix<double, 3, 4>& P, double u, double v) const {
			Eigen::Vector4d corner(0.0, u, v, 1.);
			Eigen::Vector3d x;
			x = P * corner;
			return Point2d(x(0) / x(2), x(1) / x(2));
		}
		
		// Override for derived factors
		virtual Eigen::Matrix<double, 3, 4> projectionMatrix(Selector s = ESTIMATE) const = 0;
		virtual double tagSize(Selector s = ESTIMATE) const = 0;
	};
	
	/*! \brief Factor to estimate camera pose, tag pose. */
	class Tag_Factor : public Tag_Factor_Base {
		Pose3d_Node* _pose;
		Pose3d_Node* _tag;
		TagIntrinsics* _tagIntrinsics;
		MonocularIntrinsics* _camera;

	public:

		typedef std::shared_ptr<Tag_Factor> Ptr;
		
		/**
		* Constructor.
		* @param pose The pose of the camera observing the tag in the world frame.
		* @param tag The pose of the tag in the world frame.
		* @param tagIntrinsics The intrinsics of the tag being observed.
		* @param camera The param struct of the observing camera.
		* @param measure The measurements of the corners of the tag
		* @param noise The 8x8 noise matrix.
		*/
		Tag_Factor( Pose3d_Node* pose, Pose3d_Node* tag, TagIntrinsics* tagIntrinsics,
					MonocularIntrinsics* camera, const TagCorners& measure, const Noise& noise ) 
			: Tag_Factor_Base( "Tag_Factor", 8, noise, measure ), 
			_pose( pose ), _tag( tag ), _tagIntrinsics( tagIntrinsics ), _camera( camera )
		{
			_nodes.resize(2);
			_nodes[0] = _pose;
			_nodes[1] = _tag;
		}

		void initialize() {
			require(_tag->initialized(),
				"Tag_Factor requires tag to be initialized");
			require(_pose->initialized(),
				"Tag_Factor requires pose to be initialized");
			require( _tagIntrinsics != nullptr,
				"Tag_Factor requires tag intrinsics to be initialized");
		}
		
		virtual Eigen::Matrix<double, 3, 4> projectionMatrix(Selector s = ESTIMATE) const {
			const Pose3d& cam = _pose->value(s);
			const Pose3d& tag = _tag->value(s);
			Eigen::Matrix<double, 3, 4> P = _camera->K() * _rot
				* cam.oTw() * tag.wTo() ;
			return P;
		}
		
		virtual double tagSize(Selector s = ESTIMATE) const {
			return _tagIntrinsics->tagSize;
		}

	};
	
	/*! \brief Factor to estimate robot pose, tag pose, and camera extrinsics. */
	class Tag_Extrinsics_Factor: public Tag_Factor_Base {
		Pose3d_Node* _pose;
		Pose3d_Node* _tag;
		TagIntrinsics* _tagIntrinsics;
		Pose3d_Node* _extrinsics;
		MonocularIntrinsics* _camera;

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
			: Tag_Factor_Base("Tag_Extrinsics_Factor", 8, noise, measure), 
				_pose( pose ), _tag( tag ), _tagIntrinsics( tagIntrinsics ),
				_extrinsics( cameraExtrinsics ), _camera( camera )
		{
			_nodes.resize(3);
			_nodes[0] = _pose;
			_nodes[1] = _tag;
 			_nodes[2] = _extrinsics;
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

		virtual Eigen::Matrix<double, 3, 4> projectionMatrix(Selector s = ESTIMATE) const {
			const Pose3d& bot = _pose->value(s);
			const Pose3d& tag = _tag->value(s);
			const Pose3d& ext = _extrinsics->value(s);
			Eigen::Matrix<double, 3, 4> P = _camera->K() * _rot
				* ext.oTw() * bot.oTw() * tag.wTo() ;
			return P;
		}
				
		virtual double tagSize(Selector s = ESTIMATE) const {
			return _tagIntrinsics->tagSize;
		}
	};
	
	/*! \brief Factor to estimate robot pose, tag pose, and camera intrinsics/extrinsics. */
	class Tag_Intrinsics_Extrinsics_Factor: public Tag_Factor_Base {
		Pose3d_Node* _pose;
		Pose3d_Node* _tag;
		TagIntrinsics* _tagIntrinsics;
		Pose3d_Node* _extrinsics;
		Intrinsics_Node* _cameraIntrinsics;

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
			: Tag_Factor_Base("Tag_Intrinsics_Extrinsics_Factor", 8, noise, measure), 
				_pose( pose ), _tag( tag ), _tagIntrinsics( tagIntrinsics ),
				_extrinsics( cameraExtrinsics ), _cameraIntrinsics( cameraIntrinsics )
		{
			_nodes.resize(4);
			_nodes[0] = _pose;
			_nodes[1] = _tag;
 			_nodes[2] = _extrinsics;
			_nodes[3] = _cameraIntrinsics;
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

		virtual Eigen::Matrix<double,3,4> projectionMatrix(Selector s = ESTIMATE) const {
			const Pose3d& bot = _pose->value(s);
			const Pose3d& tag = _tag->value(s);
			const Pose3d& ext = _extrinsics->value(s);
			Eigen::Matrix<double, 3, 4> P = _cameraIntrinsics->value(s).K() * _rot
				* ext.oTw() * bot.oTw() * tag.wTo() ;
			return P;
		}
			
		virtual double tagSize(Selector s = ESTIMATE) const {
			return _tagIntrinsics->tagSize;
		}
	};
	
	/*! \brief Factor to estimate robot pose, tag intrinsics/extrinsics, and camera extrinsics. */
	class Tag_Size_Factor: public Tag_Factor_Base {
		Pose3d_Node* _pose;
		Pose3d_Node* _tag;
		Tag_Node* _tagIntrinsics;
		Pose3d_Node* _extrinsics;
		MonocularIntrinsics* _cameraIntrinsics;

	public:

		typedef std::shared_ptr<Tag_Size_Factor> Ptr;

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
		Tag_Size_Factor( Pose3d_Node* pose, 
							   Pose3d_Node* tag, Tag_Node* tagIntrinsics,
							   Pose3d_Node* cameraExtrinsics, MonocularIntrinsics* cameraIntrinsics, 
							   const TagCorners& measure, const Noise& noise ) 
			: Tag_Factor_Base("Tag_Size_Factor", 8, noise, measure), 
				_pose( pose ), _tag( tag ), _tagIntrinsics( tagIntrinsics ),
				_extrinsics( cameraExtrinsics ), _cameraIntrinsics( cameraIntrinsics )
		{
			_nodes.resize(4);
			_nodes[0] = _pose;
			_nodes[1] = _tag;
 			_nodes[2] = _extrinsics;
			_nodes[3] = _tagIntrinsics;
		}

		void initialize() {
			require(_tag->initialized(),
				"Tag_Size_Factor requires tag to be initialized");
			require(_pose->initialized(),
				"Tag_Size_Factor requires pose to be initialized");
			require( _extrinsics->initialized(),
				"Tag_Size_Factor requires camera extrinsics to be initialized");
			require( _tagIntrinsics->initialized(),
				"Tag_Size_Factor requires tag intrinsics to be initialized");
			require( _cameraIntrinsics != nullptr,
				"Tag_Size_Factor requires camera intrinsics to be initialized");
		}

		virtual Eigen::Matrix<double,3,4> projectionMatrix(Selector s = ESTIMATE) const {
			const Pose3d& bot = _pose->value(s);
			const Pose3d& tag = _tag->value(s);
			const Pose3d& ext = _extrinsics->value(s);
			Eigen::Matrix<double, 3, 4> P = _cameraIntrinsics->K() * _rot
				* ext.oTw() * bot.oTw() * tag.wTo() ;
			return P;
		}
			
		virtual double tagSize(Selector s = ESTIMATE) const {
			return _tagIntrinsics->value(s).tagSize;
		}
	}; 
	
	/*! \brief Factor to estimate robot pose, tag intrinsics/extrinsics, and camera intrinsics/extrinsics. */
	class Tag_Calibration_Factor: public Tag_Factor_Base {
		Pose3d_Node* _pose;
		Pose3d_Node* _tag;
		Tag_Node* _tagIntrinsics;
		Pose3d_Node* _extrinsics;
		Intrinsics_Node* _cameraIntrinsics;

	public:

		typedef std::shared_ptr<Tag_Calibration_Factor> Ptr;

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
		Tag_Calibration_Factor( Pose3d_Node* pose, 
							   Pose3d_Node* tag, Tag_Node* tagIntrinsics,
							   Pose3d_Node* cameraExtrinsics, Intrinsics_Node* cameraIntrinsics, 
							   const TagCorners& measure, const Noise& noise ) 
			: Tag_Factor_Base("Tag_Calibration_Factor", 8, noise, measure), 
				_pose( pose ), _tag( tag ), _tagIntrinsics( tagIntrinsics ),
				_extrinsics( cameraExtrinsics ), _cameraIntrinsics( cameraIntrinsics )
		{
			_nodes.resize(5);
			_nodes[0] = _pose;
			_nodes[1] = _tag;
 			_nodes[2] = _extrinsics;
			_nodes[3] = _cameraIntrinsics;
			_nodes[4] = _tagIntrinsics;
		}

		void initialize() {
			require(_tag->initialized(),
				"Tag_Calibration_Factor requires tag to be initialized");
			require(_pose->initialized(),
				"Tag_Calibration_Factor requires pose to be initialized");
			require( _extrinsics->initialized(),
				"Tag_Calibration_Factor requires camera extrinsics to be initialized");
			require( _tagIntrinsics->initialized(),
				"Tag_Calibration_Factor requires tag intrinsics to be initialized");
			require( _cameraIntrinsics ->initialized(),
				"Tag_Calibration_Factor requires camera intrinsics to be initialized");
		}

		virtual Eigen::Matrix<double,3,4> projectionMatrix(Selector s = ESTIMATE) const {
			const Pose3d& bot = _pose->value(s);
			const Pose3d& tag = _tag->value(s);
			const Pose3d& ext = _extrinsics->value(s);
			Eigen::Matrix<double, 3, 4> P = _cameraIntrinsics->value(s).K() * _rot
				* ext.oTw() * bot.oTw() * tag.wTo() ;
			return P;
		}
			
		virtual double tagSize(Selector s = ESTIMATE) const {
			return _tagIntrinsics->value(s).tagSize;
		}
	}; 

}
