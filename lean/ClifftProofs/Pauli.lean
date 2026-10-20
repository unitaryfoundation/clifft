import Mathlib.Basic.Complex.Basic
import Mathlib.LinearAlgebra.Matrix.ConjTranspose
import Mathlib.LinearAlgebra.Matrix.Notation
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.NormNum

namespace ClifftProofs

abbrev Operator := Matrix (Fin 2) (Fin 2) Complex

def xMatrix : Operator := !![0, 1; 1, 0]

def yMatrix : Operator := !![0, -Complex.I; Complex.I, 0]

def zMatrix : Operator := !![1, 0; 0, -1]

/-- The phase is an exponent of i, not the sign of a Hermitian Pauli. -/
structure Pauli where
  x : Bool
  z : Bool
  phase : Fin 4
  deriving DecidableEq

namespace Pauli

def phaseFactor (phase : Fin 4) : Complex := Complex.I ^ phase.val

def body (x z : Bool) : Operator :=
  (if x then xMatrix else 1) * (if z then zMatrix else 1)

def denote (p : Pauli) : Operator :=
  HSMul.hSMul (phaseFactor p.phase) (body p.x p.z)

/-- Moving the left Z past the right X contributes a minus sign. -/
def mul (p q : Pauli) : Pauli where
  x := Bool.xor p.x q.x
  z := Bool.xor p.z q.z
  phase := p.phase + q.phase + if p.z && q.x then 2 else 0

/-- Multiplying XZ by i gives Y; adding two to the phase negates it. -/
def signed (x z negative : Bool) : Pauli where
  x := x
  z := z
  phase := (if x && z then 1 else 0) + if negative then 2 else 0

/-- Only Hermitian inputs have a real sign after removing the body's Y phase. -/
def sign (p : Pauli) : Bool :=
  p.phase - (if p.x && p.z then 1 else 0) == 2

def isHermitian (p : Pauli) : Bool :=
  let delta := p.phase - (if p.x && p.z then 1 else 0)
  delta == 0 || delta == 2

theorem sign_signed (x z negative : Bool) : sign (signed x z negative) = negative := by
  cases x <;> cases z <;> cases negative <;> decide

theorem phaseFactor_add (a b : Fin 4) :
    phaseFactor (a + b) = phaseFactor a * phaseFactor b := by
  fin_cases a <;> fin_cases b <;>
    norm_num [phaseFactor, Fin.add_def, pow_succ, Complex.I_sq]

theorem body_mul (x z x' z' : Bool) :
    body x z * body x' z' =
      HSMul.hSMul (phaseFactor (if z && x' then 2 else 0))
        (body (Bool.xor x x') (Bool.xor z z')) := by
  cases x <;> cases z <;> cases x' <;> cases z' <;>
    ext i j <;> fin_cases i <;> fin_cases j <;>
    norm_num [body, xMatrix, zMatrix, phaseFactor, Matrix.mul_apply,
      Fin.sum_univ_two, Matrix.one_apply]

/-- This is exact operator equality, including the global phase. -/
theorem denote_mul (p q : Pauli) :
    denote (mul p q) = denote p * denote q := by
  simp only [denote, mul, phaseFactor_add, Matrix.smul_mul, Matrix.mul_smul,
    body_mul, smul_smul, mul_assoc, mul_left_comm]

theorem denote_signed_I : denote (signed false false false) = 1 := by
  norm_num [denote, signed, phaseFactor, body]

theorem denote_signed_X : denote (signed true false false) = xMatrix := by
  norm_num [denote, signed, phaseFactor, body]

theorem denote_signed_Z : denote (signed false true false) = zMatrix := by
  norm_num [denote, signed, phaseFactor, body]

theorem denote_signed_Y : denote (signed true true false) = yMatrix := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [denote, signed, phaseFactor, body, xMatrix, yMatrix, zMatrix,
      Matrix.mul_apply, Fin.sum_univ_two]

theorem denote_signed_negative (x z : Bool) :
    denote (signed x z true) = -denote (signed x z false) := by
  cases x <;> cases z <;>
    norm_num [denote, signed, phaseFactor, pow_succ, Complex.I_sq, neg_smul]

theorem isHermitian_iff_signed (p : Pauli) :
    isHermitian p = true <-> exists negative, p = signed p.x p.z negative := by
  cases p with
  | mk x z phase =>
    cases x <;> cases z <;> fin_cases phase <;> decide

/-- The phase test agrees with the usual adjoint-based definition. -/
theorem isHermitian_iff_conjTranspose (p : Pauli) :
    isHermitian p = true <-> Matrix.conjTranspose (denote p) = denote p := by
  rw [<- Matrix.ext_iff]
  cases p with
  | mk x z phase =>
    cases x <;> cases z <;> fin_cases phase <;>
      norm_num [isHermitian, denote, phaseFactor, body, xMatrix, zMatrix,
        Fin.forall_fin_two, Matrix.conjTranspose_apply, Matrix.smul_apply,
        Matrix.mul_apply, Fin.sum_univ_two, Matrix.one_apply, pow_succ, Complex.ext_iff]

theorem hermitian_iff_signed (p : Pauli) :
    Matrix.conjTranspose (denote p) = denote p <->
      exists negative, p = signed p.x p.z negative := by
  rw [<- isHermitian_iff_conjTranspose, isHermitian_iff_signed]

theorem denote_injective : Function.Injective denote := by
  intro p q h
  rw [<- Matrix.ext_iff] at h
  cases p with
  | mk x z phase =>
    cases q with
    | mk x' z' phase' =>
      cases x <;> cases z <;> cases x' <;> cases z' <;>
        fin_cases phase <;> fin_cases phase' <;>
        norm_num [denote, phaseFactor, body, xMatrix, zMatrix,
          Fin.forall_fin_two, Matrix.mul_apply,
          Fin.sum_univ_two, Matrix.one_apply, pow_succ, Complex.ext_iff] at h <;> rfl

end Pauli

end ClifftProofs
