MathJax.Hub.Config({
  TeX: {
    Macros: {
      sket: ["\\left|\\left. #1 \\right\\rangle\\!\\right\\rangle",1],
      sbra: ["\\left\\langle\\!\\left\\langle #1 \\right.\\right|",1],
      sbraket: ["\\left\\langle\\!\\left\\langle #1 | #2 \\right\\rangle\\!\\right\\rangle",2],
      ket: ["\\left| #1 \\right\\rangle",1],
      bra: ["\\left\\langle #1 \\right|",1],
      braket: ["\\left\\langle #1 | #2 \\right\\rangle",2],
      vec: ["\\text{vec}\\left(#1\\right)",1],
      tr: ["\\text{Tr}\\left(#1\\right)",1]
    }
  }
});
MathJax.Ajax.loadComplete("[MathJax]/config/local/local.js");
