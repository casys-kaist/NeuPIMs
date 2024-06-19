#pragma once

#include "Common.h"

/**
 * Logger takes vector of StatClass and filename
 * Expects fname without extension. (without .tsv)
 * StatClass needs following methods
 *   static std::string get_columns(): log names of the column separated with tab
 *   std::string repr(): log stats separated with tab
 *   to write stat in a single line.
 */
namespace Logger {
template <typename StatClass>
void log(std::vector<StatClass> stats, std::string fname) {
    fname += ".tsv";
    std::ofstream ofile(fname);
    if (!ofile.is_open()) {
        assert(0);
    }
    ofile << StatClass::get_columns();
    for (auto stat : stats) {
        ofile << stat.repr();
    }
    ofile.close();
}
};  // namespace Logger