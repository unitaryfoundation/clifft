(function () {
  var defined = typeof renderMathInElement !== "undefined";

  function render() {
    renderMathInElement(document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\(", right: "\\)", display: false },
        { left: "\\[", right: "\\]", display: true },
      ],
    });
  }

  if (defined) {
    // Scripts already loaded (e.g. instant navigation)
    document$.subscribe(render);
  } else {
    // First load: inject CDN scripts in order, then hook up
    var katexScript = document.createElement("script");
    katexScript.src =
      "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js";
    katexScript.onload = function () {
      var autoScript = document.createElement("script");
      autoScript.src =
        "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js";
      autoScript.onload = function () {
        render();
        document$.subscribe(render);
      };
      document.head.appendChild(autoScript);
    };
    document.head.appendChild(katexScript);
  }
})();
