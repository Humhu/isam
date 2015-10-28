#pragma once

#include "isam/Slam.h"

#include <memory>
#include <stdexcept>

namespace isam
{

	/*! \brief A rudimentary interface to the original Slam object using shared
	 * pointers for safer memory management. */
	class SlamInterface
	{
	public:
		
		typedef std::shared_ptr<SlamInterface> Ptr;
		
		SlamInterface( Slam::Ptr _slam )
			: slam( _slam ) {
			
			if( !slam )
			{
				throw std::runtime_error( "Invalid SLAM instance." );
			}
		}
		
		Slam::Ptr get_slam() { return slam; }
		
		void add_node( const Node::Ptr& node )
		{
			nodes.push_back( node );
			slam->add_node( node.get() );
		}
		
		void add_factor( const Factor::Ptr& factor )
		{
			factors.push_back( factor );
			slam->add_factor( factor.get() );
		}
		
		void remove_node( const Node::Ptr& node )
		{
			slam->remove_node( node.get() );
			nodes.remove( node );
		}
		
		void remove_factor( const Factor::Ptr& factor )
		{
			slam->remove_factor( factor.get() );
			factors.remove( factor );
		}
		
	private:
		
		Slam::Ptr slam;
		
		std::list< Node::Ptr > nodes;
		std::list< Factor::Ptr > factors;
		
	};
	
}
