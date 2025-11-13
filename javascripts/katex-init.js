
// Robust KaTeX auto-render for MkDocs Material + mkdocs-jupyter
(function() {
  function renderAll(root) {
    if (!window.renderMathInElement) return;
    try {
      renderMathInElement(root || document.body, {
        delimiters: [
          {left: "$$", right: "$$", display: true},
          {left: "$",  right: "$",  display: false},
          {left: "\\(", right: "\\)", display: false},
          {left: "\\[", right: "\\]", display: true}
        ],
        throwOnError: false
      });
    } catch(e) {
      console.error("KaTeX render error:", e);
    }
  }

  function attachObserver() {
    const target = document.querySelector("main") || document.body;
    const mo = new MutationObserver(() => renderAll(target));
    mo.observe(target, {childList: true, subtree: true});
    renderAll(target);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", attachObserver);
  } else {
    attachObserver();
  }

  // Re-render on Material's instant navigation
  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(() => attachObserver());
  }
})();
