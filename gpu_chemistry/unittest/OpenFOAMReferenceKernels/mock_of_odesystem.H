#pragma once

#include "test_utilities.H"

#include "ODESystem.H"
#include "create_foam_inputs.H"
#include "openfoam_reference_reactions.H"

namespace Foam {

enum jacobianType { fast = 0, exact = 1 };

//- Reference ODE system mirroring chemistryModels::Standard::derivatives and
//  ::jacobian of OpenFOAM-dev (f31f38e312), with the reactions' Jacobian
//  work arrays held by the reactions as OpenFOAM-dev (62be3e2828).
//
//  The state is Y_0..Y_{n-1}, T (nEqns = nSpecie + 1) with the pressure
//  held constant as a member (p_), as in the gpuFoam GPU ODE system.
class MockOFSystem : public ODESystem {

    const PtrList<FoamGpu::testThermoType1>            specieThermos_;
    const autoPtr<reactionList>              reactionsPtr_;
    const reactionList&                      reactions_;
    label                                    nSpecie_;
    scalar                                   p_;
    mutable scalarField                      Y_;
    mutable reaction::state                  s_;
    mutable FixedList<scalarField, 5>        YTWork_;
    mutable FixedList<scalarSquareMatrix, 2> YTYTWork_;

    const bool  reduction_ = false;
    List<label> cTos_{};

    //- Type of the Jacobian to be calculated
    const jacobianType jacobianType_ = jacobianType::fast;

public:

    const reactionList& getReactions() const
    {
        return reactions_;
    }


    MockOFSystem(TestData::Mechanism m)
        : specieThermos_(TestData::makeCpuThermos_h(m))
        , reactionsPtr_(OFReferenceKernels::makeReactions(m, specieThermos_))
        , reactions_(reactionsPtr_())
        , nSpecie_(TestData::speciesCount(m))
        , p_(TestData::pInf(m))
        , Y_(nSpecie_)
        , s_(nSpecie_)
        , YTWork_(scalarField(nSpecie_ + 1))
        , YTYTWork_(scalarSquareMatrix(nSpecie_ + 1, Zero)) {}

    virtual ~MockOFSystem() {}

    //- Set the (constant) pressure
    void setPressure(const scalar p) { p_ = p; }

    label nEqns() const override { return nSpecie_ + 1; }

    void derivatives(const scalar       time,
                     const scalarField& YT,
                     const label        li,
                     scalarField&       dYTdt) const override {

        forAll(Y_, i) { Y_[i] = max(YT[i], 0); }

        const scalar T
        (
            maxMin(YT[nSpecie_], reactions_.Tlow(), reactions_.Thigh())
        );

        // Evaluate the mixture density
        scalar rhoM = 0;
        for (label i = 0; i < Y_.size(); i++) {
            rhoM += Y_[i] / specieThermos_[i].rho(p_, T);
        }
        rhoM = 1 / rhoM;

        // Evaluate the concentrations
        for (label i = 0; i < Y_.size(); i++) {
            s_.c[i] = rhoM / specieThermos_[i].W() * Y_[i];
        }

        s_.reset(p_, T, li);

        // Evaluate contributions from reactions
        dYTdt = Zero;
        forAll(reactions_, ri) {
            reactions_[ri].dNdtByV(s_, dYTdt, reduction_, cTos_);
        }

        // Reactions return dNdtByV, so we need to convert the result
        // to dYdt
        for (label i = 0; i < nSpecie_; i++) {
            const scalar WiByrhoM =
                specieThermos_[sToc(i)].W() / rhoM;
            scalar& dYidt = dYTdt[i];
            dYidt *= WiByrhoM;
        }

        // Evaluate the effect on the thermodynamic system ...

        // Evaluate the mixture Cp
        scalar CpM = 0;
        for (label i = 0; i < Y_.size(); i++) {
            CpM += Y_[i] * specieThermos_[i].Cp(p_, T);
        }

        // dT/dt
        scalar& dTdt = dYTdt[nSpecie_];
        for (label i = 0; i < nSpecie_; i++) {
            dTdt -= dYTdt[i] * specieThermos_[sToc(i)].ha(p_, T);
        }
        dTdt /= CpM;
    }

    void jacobian(const scalar        t,
                  const scalarField&  YT,
                  const label         li,
                  scalarField&        dYTdt,
                  scalarSquareMatrix& J) const override {

        forAll(Y_, i) { Y_[i] = max(YT[i], 0); }

        const scalar T
        (
            maxMin(YT[nSpecie_], reactions_.Tlow(), reactions_.Thigh())
        );

        // Evaluate the specific volumes and mixture density
        scalarField& v = YTWork_[0];
        for (label i = 0; i < Y_.size(); i++) {
            v[i] = 1 / specieThermos_[i].rho(p_, T);
        }
        scalar rhoM = 0;
        for (label i = 0; i < Y_.size(); i++) {
            rhoM += Y_[i] * v[i];
        }
        rhoM = 1 / rhoM;

        // Evaluate the concentrations
        for (label i = 0; i < Y_.size(); i++) {
            s_.c[i] = rhoM / specieThermos_[i].W() * Y_[i];
        }

        s_.reset(p_, T, li);

        // Evaluate the derivatives of concentration w.r.t. mass
        // fraction
        scalarSquareMatrix& dcdY = YTYTWork_[0];
        for (label i = 0; i < nSpecie_; i++) {
            const scalar rhoMByWi =
                rhoM / specieThermos_[sToc(i)].W();
            switch (jacobianType_) {
            case jacobianType::fast: {
                dcdY(i, i) = rhoMByWi;
            } break;
            case jacobianType::exact:
                for (label j = 0; j < nSpecie_; j++) {
                    dcdY(i, j) =
                        rhoMByWi *
                        ((i == j) - rhoM * v[sToc(j)] * Y_[sToc(i)]);
                }
                break;
            }
        }

        // Evaluate the mixture thermal expansion coefficient
        scalar alphavM = 0;
        for (label i = 0; i < Y_.size(); i++) {
            alphavM +=
                Y_[i] * rhoM * v[i] * specieThermos_[i].alphav(p_, T);
        }

        // Evaluate contributions from reactions
        dYTdt = Zero;
        scalarSquareMatrix& ddNdtByVdcT = YTYTWork_[1];
        for (label i = 0; i < nSpecie_ + 1; i++) {
            for (label j = 0; j < nSpecie_ + 1; j++) {
                ddNdtByVdcT[i][j] = 0;
            }
        }
        forAll(reactions_, ri) {
            OFReferenceKernels::ddNdtByVdcT(
                reactions_[ri], s_, dYTdt, ddNdtByVdcT, nSpecie_);
        }

        // Reactions return dNdtByV, so we need to convert the result
        // to dYdt
        for (label i = 0; i < nSpecie_; i++) {
            const scalar WiByrhoM =
                specieThermos_[sToc(i)].W() / rhoM;
            scalar& dYidt = dYTdt[i];
            dYidt *= WiByrhoM;

            for (label j = 0; j < nSpecie_; j++) {
                scalar ddNidtByVdYj = 0;
                switch (jacobianType_) {
                case jacobianType::fast: {
                    const scalar ddNidtByVdcj = ddNdtByVdcT(i, j);
                    ddNidtByVdYj = ddNidtByVdcj * dcdY(j, j);
                } break;
                case jacobianType::exact:
                    for (label k = 0; k < nSpecie_; k++) {
                        const scalar ddNidtByVdck = ddNdtByVdcT(i, k);
                        ddNidtByVdYj += ddNidtByVdck * dcdY(k, j);
                    }
                    break;
                }

                scalar& ddYidtdYj = J(i, j);
                ddYidtdYj         = WiByrhoM * ddNidtByVdYj +
                            rhoM * v[sToc(j)] * dYidt;
            }

            scalar ddNidtByVdT = ddNdtByVdcT(i, nSpecie_);
            for (label j = 0; j < nSpecie_; j++) {
                const scalar ddNidtByVdcj = ddNdtByVdcT(i, j);
                ddNidtByVdT -= ddNidtByVdcj * s_.c[sToc(j)] * alphavM;
            }

            scalar& ddYidtdT = J(i, nSpecie_);
            ddYidtdT = WiByrhoM * ddNidtByVdT + alphavM * dYidt;
        }

        // Evaluate the effect on the thermodynamic system ...

        // Evaluate the mixture Cp and its derivative
        scalarField& Cp  = YTWork_[3];
        scalar       CpM = 0, dCpMdT = 0;
        for (label i = 0; i < Y_.size(); i++) {
            Cp[i] = specieThermos_[i].Cp(p_, T);
            CpM += Y_[i] * Cp[i];
            dCpMdT += Y_[i] * specieThermos_[i].dCpdT(p_, T);
        }

        // dT/dt
        scalarField& ha   = YTWork_[4];
        scalar&      dTdt = dYTdt[nSpecie_];
        for (label i = 0; i < nSpecie_; i++) {
            ha[sToc(i)] = specieThermos_[sToc(i)].ha(p_, T);
            dTdt -= dYTdt[i] * ha[sToc(i)];
        }
        dTdt /= CpM;

        // d(dTdt)/dY
        for (label i = 0; i < nSpecie_; i++) {
            scalar& ddTdtdYi = J(nSpecie_, i);
            ddTdtdYi         = 0;
            for (label j = 0; j < nSpecie_; j++) {
                const scalar ddYjdtdYi = J(j, i);
                ddTdtdYi -= ddYjdtdYi * ha[sToc(j)];
            }
            ddTdtdYi -= Cp[sToc(i)] * dTdt;
            ddTdtdYi /= CpM;
        }

        // d(dTdt)/dT
        scalar& ddTdtdT = J(nSpecie_, nSpecie_);
        ddTdtdT         = 0;
        for (label i = 0; i < nSpecie_; i++) {
            const scalar dYidt    = dYTdt[i];
            const scalar ddYidtdT = J(i, nSpecie_);
            ddTdtdT -= dYidt * Cp[sToc(i)] + ddYidtdT * ha[sToc(i)];
        }
        ddTdtdT -= dTdt * dCpMdT;
        ddTdtdT /= CpM;
    }

    inline label sToc(const label si) const { return si; }

    inline label cTos(const label ci) const { return ci; }
};

} // namespace Foam