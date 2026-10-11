// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PAIR_EVALUATOR_LARADJISUNIL_H__
#define __PAIR_EVALUATOR_LARADJISUNIL_H__

#include "hoomd/HOOMDMath.h"

#ifndef __HIPCC__
#include <string>
#endif

// DEVICE is __device__ when compiling GPU kernels.
// HOSTDEVICE is __host__ __device__ when supported by the compiler.
#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

namespace hoomd
    {
namespace md
    {

class EvaluatorPairLaradjiSunil
    {
    public:
    //! Parameters for each particle-type pair.
    struct param_type
        {
        Scalar Umax;
        Scalar Umin;
        Scalar rm;

        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr,
                                        unsigned int& available_bytes) const
            {
            }

#ifdef ENABLE_HIP
        void set_memory_hint() const
            {
            }
#endif

#ifndef __HIPCC__
        param_type()
            : Umax(0), Umin(0), rm(1)
            {
            }

        param_type(pybind11::dict v, bool managed = false)
            {
            Umax = v["Umax"].cast<Scalar>();
            Umin = v["Umin"].cast<Scalar>();
            rm = v["rm"].cast<Scalar>();
            }

        pybind11::dict asDict()
            {
            pybind11::dict v;
            v["Umax"] = Umax;
            v["Umin"] = Umin;
            v["rm"] = rm;
            return v;
            }
#endif
        }
#if HOOMD_LONGREAL_SIZE == 32
        __attribute__((aligned(8)));
#else
        __attribute__((aligned(16)));
#endif

    //! Construct the evaluator.
    /*!
     * \param _rsq Squared distance between particles.
     * \param _rcutsq Squared pair cutoff.
     * \param _params Parameters for this particle-type pair.
     */
    DEVICE EvaluatorPairLaradjiSunil(Scalar _rsq,
                                     Scalar _rcutsq,
                                     const param_type& _params)
        : rsq(_rsq),
          rcutsq(_rcutsq),
          Umax(_params.Umax),
          Umin(_params.Umin),
          rm(_params.rm)
        {
        }

    //! The potential does not depend on particle charge.
    DEVICE static bool needsCharge()
        {
        return false;
        }

    //! Accept the optional charge arguments.
    DEVICE void setCharge(Scalar qi, Scalar qj) { }

    //! Evaluate the pair force and potential energy.
    /*!
     * force_divr = F(r)/r, where F(r) = -dU/dr.
     *
     * The potential is already zero with zero slope at r = rc.
     * Therefore no additional energy shift is necessary.
     */
    DEVICE bool evalForceAndEnergy(Scalar& force_divr,
                                   Scalar& pair_eng,
                                   bool energy_shift)
        {
        // The HOOMD pair implementation handles the cutoff.
        if (!(rsq < rcutsq))
            {
            return false;
            }

        // Handle exact overlap without producing NaNs or infinities.
        // A physically meaningful direction does not exist at r = 0,
        // so the force is set to zero in this exceptional case.
        if (rsq <= Scalar(0))
            {
            force_divr = Scalar(0);
            pair_eng = Umax;
            return true;
            }

        Scalar rinv = fast::rsqrt(rsq);
        Scalar r = rsq * rinv;

        // Recover the cutoff from HOOMD's squared cutoff.
        Scalar rcut = rcutsq * fast::rsqrt(rcutsq);

        // Avoid division by zero for invalid parameters.
        // Valid use requires rm > 0 and rcut > rm.
        if (!(rm > Scalar(0)) || !(rcut > rm))
            {
            force_divr = Scalar(0);
            pair_eng = Scalar(0);
            return false;
            }

        Scalar force;

        if (r <= rm)
            {
            // Repulsive core:
            //
            // U(r) = (Umax-Umin)*[(rm-r)/rm]^2 + Umin
            //
            // F(r) = -dU/dr
            //
            // U(0)  = Umax
            // U(rm) = Umin

            Scalar dr = rm - r;
            Scalar scaled_dr = dr / rm;

            pair_eng = (Umax - Umin) * scaled_dr * scaled_dr + Umin;

            force = Scalar(2) * (Umax - Umin) * dr / (rm * rm);
            }
        else
            {
            // Smooth attractive tail:
            //
            // x = (rc-r)/(rc-rm)
            //
            // U(r) = Umin*x^2*(3-2*x)
            //
            // F(r) = 6*Umin*x*(1-x)/(rc-rm)
            //
            // U(rc) = 0 and F(rc) = 0.

            Scalar range = rcut - rm;
            Scalar x = (rcut - r) / range;

            pair_eng = Umin * x * x * (Scalar(3) - Scalar(2) * x);

            force = Scalar(6) * Umin * x * (Scalar(1) - x) / range;
            }

        // HOOMD expects the radial force divided by the pair distance.
        force_divr = force * rinv;

        // energy_shift is intentionally unused: the potential is
        // already zero at the cutoff.
        (void)energy_shift;

        return true;
        }

    //! No long-range pressure correction.
    DEVICE Scalar evalPressureLRCIntegral()
        {
        return Scalar(0);
        }

    //! No long-range energy correction.
    DEVICE Scalar evalEnergyLRCIntegral()
        {
        return Scalar(0);
        }

#ifndef __HIPCC__
    //! Name reported by the evaluator.
    static std::string getName()
        {
        return std::string("laradjisunil");
        }

    //! Shape definitions are not supported.
    std::string getShapeSpec() const
        {
        throw std::runtime_error(
            "Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    Scalar rsq;
    Scalar rcutsq;

    Scalar Umax;
    Scalar Umin;
    Scalar rm;
    };

    } // namespace md
    } // namespace hoomd

#endif // __PAIR_EVALUATOR_LARADJISUNIL_H__
