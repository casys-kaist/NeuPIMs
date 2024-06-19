#ifndef GNNSIM_CONFIG_HPP_
#define GNNSIM_CONFIG_HPP_

#include "config_utils.hpp"
#include "booksim_config.hpp"

class GNNSimConfig : public BookSimConfig {
public:
  GNNSimConfig();
  ~GNNSimConfig() = default;
};

#endif
