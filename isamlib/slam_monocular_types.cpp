#include "isam/slam_monocular.h"

namespace isam
{
	
MonocularMeasurement::MonocularMeasurement( double u, double v )
: u( u ), v( v ), valid( true ) {}

MonocularMeasurement::MonocularMeasurement( double u, double v, bool w )
: u( u ), v( v ), valid( w ) {}

Eigen::Vector2d MonocularMeasurement::vector() const
{
	Eigen::Vector2d tmp(u, v);
	return tmp;
}

void MonocularMeasurement::write(std::ostream &out) const 
{
	out << "(" << u << ", " << v << ")";
}

std::ostream& operator<<(std::ostream& out, const MonocularMeasurement& t) 
{
	t.write(out);
	return out;
}

const char* MonocularIntrinsics::name() { return "MonocularIntrinsics"; }

MonocularIntrinsics::MonocularIntrinsics() 
: _fx(1), _fy(1), _pp(Eigen::Vector2d(0,0)) 
{}

MonocularIntrinsics::MonocularIntrinsics(double fx, double fy, const Eigen::Vector2d& pp) 
: _fx(fx), _fy(fy), _pp(pp) 
{}

MonocularIntrinsics::MonocularIntrinsics( double fx, double fy, double px, double py )
: _fx(fx), _fy(fy), _pp( px, py )
{}

double MonocularIntrinsics::fx() const { return _fx; }
double MonocularIntrinsics::fy() const { return _fy; }

Eigen::Vector2d MonocularIntrinsics::principalPoint() const { return _pp; }

Eigen::Matrix<double, 3, 4> MonocularIntrinsics::K() const 
{
	Eigen::Matrix<double, 3, 4> K;
	K.row(0) << _fx, 0, _pp(0), 0;
	K.row(1) << 0, _fy, _pp(1), 0;
	K.row(2) << 0, 0, 1, 0;
	return K;
}

MonocularIntrinsics MonocularIntrinsics::exmap( const Eigen::VectorXd& delta )
{
	Eigen::VectorXd current = vector();
	current += delta;
	Eigen::Vector2d pp;
	pp << current(2), current(3);
	return MonocularIntrinsics( current(0), current(1), pp );
}

void MonocularIntrinsics::set( const Eigen::VectorXd& v )
{
	_fx = v(0);
	_fy = v(1);
	_pp(0) = v(2);
	_pp(1) = v(3);
}

Eigen::VectorXb MonocularIntrinsics::is_angle() const {
	Eigen::VectorXb isang( dim );
	isang << false, false, false, false;
	return isang;
}

Eigen::VectorXd MonocularIntrinsics::vector() const
{
	Eigen::VectorXd vec(4);
	vec << _fx, _fy, _pp(0), _pp(1);
	return vec;
}

void MonocularIntrinsics::write( std::ostream& out ) const
{
	out << "(fx: " << _fx << " fy: " << _fy << "pp: [" << _pp(0) << ", " << _pp(1) << "])";
}

std::ostream& operator<<( std::ostream& out, const MonocularIntrinsics& intrinsics )
{
	intrinsics.write( out );
	return out;
}

}
