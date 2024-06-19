#ifndef _GNNTrafficManager_HPP_
#define _GNNTrafficManager_HPP_

#include <iostream>
#include <vector>
#include <list>

#include "config_utils.hpp"
#include "stats.hpp"
#include "trafficmanager.hpp"
#include "booksim.hpp"
#include "booksim_config.hpp"
#include "flit.hpp"
class DomainCrossRequest;
class Network;
class booksim2::Interconnect;
//#include "Domain/Interface/DomainCrossRequest.hpp"

class GNNTrafficManager : public TrafficManager {
protected:
  virtual void _RetireFlit( Flit *f, int dest );
  virtual void _GeneratePacket(void* packet,
                               uint64_t addr, int bytes, booksim2::Interconnect::Type type,
                               int header_size, uint32_t subnet, int cl, int time, int src, int dst);
  virtual int  _IssuePacket( int source, int cl );
  virtual void _Step();
  
  // record size of _partial_packets for each subnet
  vector<vector<vector<list<Flit *> > > > _input_queue;

  uint32_t flit_size;
  
public:
  GNNTrafficManager( const Configuration &config, const vector<Network *> & net, booksim2::Interconnect* icnt );
  virtual ~GNNTrafficManager( );
  
  // correspond to TrafficManger::Run/SingleSim
  void Init();
  
  // TODO: if it is not good...
  friend class booksim2::Interconnect;
  
  //    virtual void WriteStats( ostream & os = cout ) const;
  //    virtual void DisplayStats( ostream & os = cout ) const;
  //    virtual void DisplayOverallStats( ostream & os = cout ) const;
  
};



#endif
