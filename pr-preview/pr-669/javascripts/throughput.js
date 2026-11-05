/* Enhance the published chart without fetching or recomputing benchmark data. */
(() => {
  let cleanup = () => {};
  function initialize() {
    const root = document.querySelector("[data-throughput]");
    if (root && root.dataset.ready) return;
    cleanup();
    if (!root) return;
    const lifetime = new AbortController();
    const interactive = root.querySelector(".throughput-interactive");
    const tooltip = root.querySelector(".throughput-tooltip");
    const controls = [...root.querySelectorAll("input[type=checkbox]")];
    let activePoint = null;
    let hideTimer;

    function dismiss() {
      clearTimeout(hideTimer);
      if (activePoint) activePoint.removeAttribute("aria-describedby");
      activePoint = null;
      tooltip.hidden = true;
    }

    function positionTooltip() {
      if (!activePoint) return;
      const bounds = activePoint.getBoundingClientRect();
      tooltip.hidden = bounds.bottom < 0 || bounds.top > innerHeight;
      if (tooltip.hidden) return;
      const margin = 8;
      const left = Math.max(margin, Math.min(bounds.left, innerWidth - tooltip.offsetWidth - margin));
      const below = bounds.bottom + margin;
      const top = below + tooltip.offsetHeight <= innerHeight - margin
        ? below
        : Math.max(margin, bounds.top - tooltip.offsetHeight - margin);
      tooltip.style.left = `${left}px`;
      tooltip.style.top = `${top}px`;
    }

    function show(point) {
      dismiss();
      activePoint = point;
      const detail = root.querySelector(`#${point.dataset.detail}`);
      tooltip.replaceChildren(...[...detail.children].map((node) => node.cloneNode(true)));
      // The complete table retains navigable source links; a tooltip is descriptive.
      tooltip.querySelectorAll("a").forEach((link) => link.replaceWith(link.textContent));
      point.setAttribute("aria-describedby", tooltip.id);
      positionTooltip();
    }

    root.querySelectorAll(".throughput-point").forEach((point) => {
      point.addEventListener("pointerenter", () => show(point));
      point.addEventListener("focus", () => show(point));
      point.addEventListener("click", () => show(point));
      point.addEventListener("pointerleave", (event) => {
        if (!tooltip.contains(event.relatedTarget) && document.activeElement !== point) {
          hideTimer = setTimeout(dismiss, 150);
        }
      });
      point.addEventListener("blur", dismiss);
    });
    tooltip.addEventListener("pointerenter", () => clearTimeout(hideTimer));
    tooltip.addEventListener("pointerleave", () => {
      if (document.activeElement !== activePoint) dismiss();
    });
    root.addEventListener("pointerdown", (event) => {
      if (!event.target.closest(".throughput-point, .throughput-tooltip")) dismiss();
    });
    controls.forEach((input) => input.addEventListener("change", () => {
      dismiss();
      root.querySelectorAll(".throughput-entry").forEach((entry) => {
        entry.hidden = !controls.some((control) => control.checked && control.value === entry.dataset.simulator);
      });
      const count = controls.filter((control) => control.checked).length;
      root.querySelector(".throughput-status").textContent = count
        ? `${count} of ${controls.length} simulators shown. Panel scales are unchanged.`
        : "No simulators selected. Select a simulator to show its measurements.";
    }));
    root.querySelector(".throughput-table-link").addEventListener("click", () => {
      root.querySelector("#throughput-table").open = true;
    });
    // Discard viewport listeners when Material replaces the page during instant navigation.
    window.addEventListener("scroll", positionTooltip, { signal: lifetime.signal });
    window.addEventListener("resize", positionTooltip, { signal: lifetime.signal });
    window.addEventListener("keydown", (event) => {
      if (event.key === "Escape") dismiss();
    }, { signal: lifetime.signal });
    cleanup = () => { dismiss(); lifetime.abort(); };
    root.dataset.ready = "true";
    interactive.hidden = false;
    root.querySelector(".throughput-fallback").hidden = true;
  }

  if (typeof document$ !== "undefined") document$.subscribe(initialize);
  else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initialize);
  else initialize();
})();
