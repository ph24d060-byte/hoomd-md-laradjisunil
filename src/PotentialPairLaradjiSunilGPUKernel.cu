// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "EvaluatorPairLaradjiSunil.h"
#include "hoomd/md/PotentialPairGPU.cuh"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {

template __attribute__((visibility("default"))) hipError_t
gpu_compute_pair_forces<EvaluatorPairLaradjiSunil>(
    const pair_args_t& pair_args,
    const EvaluatorPairLaradjiSunil::param_type* d_params);

    } // namespace kernel
    } // namespace md
    } // namespace hoomd
