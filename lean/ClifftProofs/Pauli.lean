import Mathlib.Basic.Complex.Basic
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

end Pauli

end ClifftProofs
