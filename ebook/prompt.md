Builds a single self-contained HTML ebook from the five project stages
(notebook_1 ... notebook_5). Each .py script becomes one section made of its
docstring explanation and its code. Outputs, figures and tables come from the
executed notebooks in notebooks/ and from saved artefacts in evaluation/,
models/, benchmark/ and data/. Error outputs are dropped.