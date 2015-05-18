// iSAM types for monocular camera extrinsics calibration

#pragma once

#include <Eigen/Dense>
#include "Node.h"
#include "Pose3d.h"
#include "Point3d.h"
#include "slam_monocular.h"

namespace isam
{
	
	typedef NodeT<Pose3d> Extrinsics3d_Node;
	
	class Monocular_Extrinsics_Factor : public FactorT<MonocularMeasurement>
	{
		Pose3d_Node* _pose;
		Point3d_Node* _point;
		Point3dh_Node* _point_h;
		Extrinsics3d_Node* _extrinsics;
		MonocularIntrinsics* _camera;
		
	public:
		
		typedef std::shared_ptr<Monocular_Extrinsics_Factor> Ptr;
		
		// constructor for projective geometry
		Monocular_Extrinsics_Factor( Pose3d_Node* pose, Point3dh_Node* point, 
									 Extrinsics3d_Node* extrinsics, MonocularIntrinsics* camera,
									 const MonocularMeasurement& measure, const isam::Noise& noise )
		: FactorT<MonocularMeasurement>("Monocular_Extrinsics_Factor", 3, noise, measure),
		_pose(pose), _point(NULL), _point_h(point), _extrinsics(extrinsics), _camera(camera) {
			// MonocularIntrinsics could also be a node later (either with 0 variables,
			// or with calibration as variables)
			_nodes.resize(3);
			_nodes[0] = pose;
			_nodes[1] = point;
			_nodes[2] = extrinsics;
		}
		
		// constructor for Euclidean geometry - WARNING: only use for points at short range
		Monocular_Extrinsics_Factor( Pose3d_Node* pose, Point3d_Node* point, 
									 Extrinsics3d_Node* extrinsics, MonocularIntrinsics* camera,
									 const MonocularMeasurement& measure, const isam::Noise& noise)
		: FactorT<MonocularMeasurement>("Monocular_Extrinsics_Factor", 3, noise, measure),
		_pose(pose), _point(point), _point_h(NULL), _extrinsics(extrinsics), _camera(camera) {
			_nodes.resize(2);
			_nodes[0] = pose;
			_nodes[1] = point;
			_nodes[2] = extrinsics;
		}
		
		void initialize() {
			require( _pose->initialized(),
					 "Monocular_Extrinsics_Factor requires pose to be initialized" );
			require( _extrinsics->initialized(),
					 "Monocular_Extrinsics_Factor requires extrinsics to be initialized" );
			
			bool initialized = (_point_h!=NULL) ? _point_h->initialized() : _point->initialized();
			if (!initialized) {
				Pose3d pose = _pose->value();
				Pose3d cameraPose = pose.oplus( _extrinsics->value() );
				Point3dh predict = _camera->backproject( cameraPose, _measure );
				// normalize homogeneous vector
				predict = Point3dh(predict.vector()).normalize();
				if (_point_h!=NULL) {
					_point_h->init(predict);
				} else {
					_point->init(predict.to_point3d());
				}
			}
		}
		
		Eigen::VectorXd basic_error(Selector s = ESTIMATE) const {
			Point3dh point = (_point_h!=NULL) ? _point_h->value(s) : _point->value(s);
			Pose3d pose = _pose->value(s);
			Pose3d cameraPose = pose.oplus( _extrinsics->value(s) );
			MonocularMeasurement predicted = _camera->project(cameraPose, point);
			if (predicted.valid == true) {
				return (predicted.vector() - _measure.vector());
			} else {
				// effectively disables points behind the camera
				return Eigen::Vector2d::Zero();
			}
		}
	};
	
}
