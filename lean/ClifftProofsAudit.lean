import ClifftProofs
import Lean.Util.CollectAxioms

open Lean in
run_cmd do
  let env <- getEnv
  let allowed := #[`propext, `Classical.choice, `Quot.sound]
  let mut checked : Nat := 0
  for (name, _) in env.constants.toList do
    if let some moduleIdx := env.getModuleIdxFor? name then
      let moduleName := env.header.moduleNames[moduleIdx.toNat]!
      if (`ClifftProofs).isPrefixOf moduleName then
        checked := checked + 1
        let axioms <- collectAxioms name
        for axiomName in axioms do
          unless allowed.contains axiomName do
            throwError "{name} depends on disallowed axiom {axiomName}"
  if checked == 0 then
    throwError "No ClifftProofs declarations were audited"
  logInfo m!"Audited {checked} declarations; only standard Lean axioms are allowed."
