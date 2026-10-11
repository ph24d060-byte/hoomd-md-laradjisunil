// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "EvaluatorPairLaradjiSunil.h"

#include "hoomd/md/PotentialPair.h"
#include <pybind11/pybind11.h>

#ifdef ENABLE_HIP
#include "hoomd/md/PotentialPairGPU.h"
#endif

namespace hoomd
    {
namespace md
    {

PYBIND11_MODULE(_laradjisunil, m)
    {
    detail::export_PotentialPair<EvaluatorPairLaradjiSunil>(
        m, "PotentialPairLaradjiSunil");

#ifdef ENABLE_HIP
    detail::export_PotentialPairGPU<EvaluatorPairLaradjiSunil>(
        m, "PotentialPairLaradjiSunilGPU");
#endif
    }

    } // namespace md
    } // namespace hoomd
