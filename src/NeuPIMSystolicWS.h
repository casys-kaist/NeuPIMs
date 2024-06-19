#include "NeuPIMSCore.h"

class NeuPIMSystolicWS : public NeuPIMSCore {
   public:
    NeuPIMSystolicWS(uint32_t id, SimulationConfig config);
    virtual void cycle() override;
    virtual void print_stats() override;
    virtual void log() override;

   protected:
    virtual cycle_type get_inst_compute_cycles(Instruction& inst) override;
    uint32_t _stat_systolic_inst_issue_count = 0;
    uint32_t _stat_systolic_preload_issue_count = 0;
    cycle_type get_vector_compute_cycles(Instruction& inst);
    cycle_type calculate_add_tree_iterations(uint32_t vector_size);
    cycle_type calculate_vector_op_iterations(uint32_t vector_size);
    void issue_ex_inst(Instruction inst);
    void pim_issue_ex_inst(Instruction inst);
    Instruction get_first_ready_ex_inst();

    std::vector<NPUStat> _stat;

    // NPU SA, VU cycle
    void systolic_cycle();
    void vector_unit_cycle();

    // Queue for SA block, PIM block
    void ld_queue_cycle();
    void st_queue_cycle();
    void ex_queue_cycle();

    // Queue for SA block, PIM block
    void pim_ld_queue_cycle();
    void pim_st_queue_cycle();
    void pim_ex_queue_cycle();

    // Update stats
    void update_stats();
};