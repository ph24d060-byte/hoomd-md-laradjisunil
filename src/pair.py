# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Laradji-Sunil pair potential."""

from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.laradjisunil import _laradjisunil
from hoomd.md import pair


class LaradjiSunil(pair.Pair):
    """Soft-core Laradji-Sunil pair interaction."""

    _ext_module = _laradjisunil
    _cpp_class_name = "PotentialPairLaradjiSunil"

    # The LS potential already goes smoothly to zero at r_cut.
    # XPLOR switching is therefore not required.
    _accepted_modes = ("none", "shift")

    def __init__(
        self,
        nlist,
        default_r_cut=None,
        default_r_on=0.0,
        mode="none",
    ):
        super().__init__(nlist, default_r_cut, default_r_on, mode)

        params = TypeParameter(
            "params",
            "particle_types",
            TypeParameterDict(
                Umax=float,
                Umin=float,
                rm=float,
                len_keys=2,
            ),
        )
        self._add_typeparam(params)
