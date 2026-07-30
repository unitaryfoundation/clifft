const LEAKAGE_LOSS_GUIDE_URL = "https://unitaryfoundation.github.io/clifft/guide/leakage-and-loss/";

// Friendly explanation shown in place of the raw compiler error when a
// pasted circuit uses LOSS or LEVEL_TRANSITION. Those annotations require a
// noncomp.Model and aren't reachable from the browser playground.
export function NoncompNotice() {
  return (
    <>
      <strong>
        Leakage & loss annotations (LOSS, LEVEL_TRANSITION) aren't supported in the playground.
      </strong>
      <br />
      See the{" "}
      <a href={LEAKAGE_LOSS_GUIDE_URL} target="_blank" rel="noopener noreferrer">
        leakage & loss guide
      </a>{" "}
      for the Python API.
    </>
  );
}
