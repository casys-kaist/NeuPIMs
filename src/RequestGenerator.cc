#include "RequestGenerator.h"

namespace RequestGenerator {
uint32_t answer_index;
uint32_t row_index;
std::vector<std::string> columns;
std::vector<std::vector<uint32_t>> table;

void init(std::string path, uint32_t _answer_index) {
    row_index = 0;

    // todo
    // initialize answer_index depending on the file type
    answer_index = _answer_index;

    parse(path);
    spdlog::info("parsed {} lines from file {}", table.size(), path);
}
int get_total_req_cnt() { return table.size(); }

bool has_data() { return row_index < table.size(); }

std::pair<uint32_t, uint32_t> get_qa_length() {
    ast(has_data());
    auto row = table[row_index++];
    return std::make_pair(row[0], row[answer_index]);
}

void parse(std::string path) {
    std::ifstream input_file(path);
    if (!input_file.is_open()) {
        std::cout << path << std::endl;
        assert(0);
    }

    std::string line;
    if (std::getline(input_file, line)) {
        std::istringstream iss(line);
        std::string column_name;
        while (std::getline(iss, column_name, ',')) {
            columns.push_back(column_name);
        }
    }

    while (std::getline(input_file, line)) {
        std::vector<uint32_t> buffer;
        std::istringstream iss(line);
        std::string cell;
        while (std::getline(iss, cell, ',')) {
            buffer.push_back(std::stoul(cell));
        }
        table.push_back(buffer);
    }
}
}  // namespace RequestGenerator