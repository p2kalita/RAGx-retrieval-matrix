I have a Jupyter notebook named `01_NYAYaRAG_Data_Preparation_and_Indexing.ipynb`.

I want you to convert this notebook into **10–13 separate Python (`.py`) scripts**, where each script represents one logical section of the notebook, and the code within each script is organized into **functions**.

### Core requirement

**Preserve the original notebook's code, logic, behavior, values, and execution flow.**

The goal is to **organize the existing code into functions and separate scripts**, NOT to redesign, optimize, or improve the implementation.

---

### 1. Read the entire notebook first

Before making any changes:

- Read the complete `01_NYAYaRAG_Data_Preparation_and_Indexing.ipynb`.
- Understand the complete execution flow.
- Identify the existing logical sections based on:
  - Markdown headings
  - Markdown descriptions
  - Comments
  - Existing functions
  - Code flow
  - Data flow
  - Major processing stages

Then divide the notebook into **10–13 logical sections**.

Do not arbitrarily split the notebook just to reach a particular number. The sections should follow the notebook's actual structure.

---

### 2. Create 10–13 separate `.py` files

Each logical section should become a separate Python script.

For example:

```text
01_<section_name>.py
02_<section_name>.py
03_<section_name>.py
...
10_<section_name>.py
```

Up to 13 files if the notebook naturally requires it.

Use meaningful filenames based on the actual functionality of each section.

---

### 3. Convert code into functions

Within each `.py` file, organize the existing code into appropriate functions.

For example, if the notebook contains:

```python
df = pd.read_csv(file_path)

df = clean_data(df)

df = process_documents(df)

print(len(df))
```

it can be organized as:

```python
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def process_data(df):
    df = clean_data(df)
    df = process_documents(df)
    return df
```

However, **do not invent new processing logic**.

The function should simply encapsulate the existing notebook code.

### Important distinction

You ARE allowed to:

- Create functions to encapsulate existing code.
- Add function parameters corresponding to variables already used by that code.
- Add `return` statements where required to pass existing values between functions/scripts.
- Move existing statements into functions.
- Organize existing statements into logical functions.

You are NOT allowed to:

- Add new business logic.
- Optimize algorithms.
- Change model behavior.
- Change prompts.
- Change processing logic.
- Add error handling that wasn't present.
- Add validation that wasn't present.
- Add new libraries.
- Add new functionality.
- Rewrite existing algorithms.
- Change variable values.
- Change paths.
- Change configuration.
- Rename existing variables unnecessarily.

---

### 4. Preserve existing functions

If the notebook already contains functions:

- Keep them.
- Do not rewrite or replace them unnecessarily.
- Do not change their implementation.
- Keep their names and behavior unchanged.
- Place them in the appropriate `.py` file based on their original location/functionality.

---

### 5. Preserve Markdown descriptions as docstrings

All meaningful Markdown descriptions from the notebook must be preserved.

Convert them into Python docstrings.

For example, if the notebook contains:

```markdown
## Load and preprocess the documents

This section loads the documents and performs the initial preprocessing.
```

the corresponding Python script should contain:

```python
"""
Load and preprocess the documents

This section loads the documents and performs the initial preprocessing.
"""
```

If the description belongs specifically to a function, place it inside that function as a docstring:

```python
def load_documents(file_path):
    """
    Load and preprocess the documents

    This section loads the documents and performs the initial preprocessing.
    """
    ...
```

**Do not paraphrase, summarize, or rewrite the descriptions.**

Preserve the original wording as much as possible.

---

### 6. Do not invent documentation

If the notebook does not contain a description for something:

- Do not create a new explanation.
- Do not add artificial documentation.
- Do not add comments merely to make the code look cleaner.

Only preserve documentation that already exists in the notebook, except for the minimum function structure required to organize the existing code.

---

### 7. Preserve variables and logic

Do NOT unnecessarily rename:

- Variables
- Functions
- Classes
- Constants
- Paths
- Model names
- Collection names
- File names
- Configuration values

Preserve the original code and values.

If a variable was created in one notebook cell and used later, handle the dependency through function arguments/return values **without changing the underlying logic**.

For example:

```python
documents = load_documents(...)
chunks = create_chunks(documents)
embeddings = create_embeddings(chunks)
```

is acceptable if those operations already existed in the notebook.

---

### 8. Preserve imports

Keep the notebook's existing imports.

Do not add libraries unless absolutely required by the function conversion itself.

Do not remove imports merely because they appear unused.

Do not replace one library with another.

---

### 9. Preserve execution order

The order of operations must remain the same as the notebook.

The resulting scripts should collectively follow the same pipeline:

```text
Notebook
   ↓
Section 1
   ↓
Section 2
   ↓
Section 3
   ↓
...
   ↓
Section 10–13
```

Do not reorder processing steps for "better organization."

---

### 10. Handle shared state carefully

The notebook may rely on variables created in earlier cells.

When converting to functions:

- Pass existing variables as function arguments where necessary.
- Return existing outputs where necessary.
- Do not create artificial global state unless the notebook already uses it.
- Do not redesign the data flow.

The objective is to make the existing notebook code function-based while maintaining its original behavior.

---

### 11. Use `if __name__ == "__main__":` where appropriate

If a script needs an executable entry point, use:

```python
if __name__ == "__main__":
    ...
```

The code inside it should only call the functions created from the existing notebook code.

Do not introduce new processing logic inside the main block.

For example:

```python
def load_data(file_path):
    ...
    return data


def process_data(data):
    ...
    return processed_data


if __name__ == "__main__":
    data = load_data(file_path)
    processed_data = process_data(data)
```

Only do this where it makes sense based on the original notebook execution flow.

---

### 12. Strict "no invention" rule

This is extremely important:

> **Do not invent any code or logic that does not already exist in the notebook.**

The only structural changes allowed are:

1. Splitting the notebook into 10–13 scripts.
2. Wrapping existing code into functions.
3. Adding function parameters where needed.
4. Adding return statements where needed.
5. Adding `if __name__ == "__main__":` where needed.
6. Converting existing Markdown descriptions into docstrings.

Everything else must remain faithful to the source notebook.

---

### 13. Final validation

After creating the scripts, verify:

- All notebook code is represented.
- No notebook code has been silently omitted.
- No new processing logic has been introduced.
- No algorithms have been changed.
- Existing functions are preserved.
- Existing variable names are preserved where practical.
- Existing configuration values are unchanged.
- Existing paths are unchanged.
- Existing model names are unchanged.
- Existing prompts are unchanged.
- Execution order is preserved.
- Markdown descriptions have been converted to docstrings.
- The scripts collectively reproduce the same pipeline as the notebook.

### Final objective

The final result should be:

**Original Notebook → 10–13 logical Python modules → Functions inside each module**

while keeping the **original source code, logic, descriptions, parameters, values, and behavior unchanged**.

The conversion should be a **function-based structural refactoring only**, not a code improvement or redesign.