window.MathJax = {
  tex: {
    inlineMath: [["$", "$"], ["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    tags: "ams"
  },
  options: {
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"]
  }
};

if (typeof document$ !== "undefined" && document$.subscribe) {
  document$.subscribe(() => {
    if (window.MathJax?.typesetPromise) {
      window.MathJax.typesetClear();
      window.MathJax.typesetPromise();
    }
  });
}
