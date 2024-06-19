#include "Common.h"

namespace RequestGenerator {
extern uint32_t answer_index;
extern uint32_t row_index;
extern std::vector<std::string> columns;
extern std::vector<std::vector<uint32_t>> table;

void init(std::string path, uint32_t _answer_index);
bool has_data();
std::pair<uint32_t, uint32_t> get_qa_length();
int get_total_req_cnt();
void parse(std::string path);
}  // namespace RequestGenerator
