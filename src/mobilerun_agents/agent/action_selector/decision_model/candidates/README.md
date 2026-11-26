# Enumerating choices

**Purpose:** Preserve the accessibility metadata needed for Jev's element labels.

**What belongs here / what doesn't:** Label and enumerate actionable UI tree nodes, including unlabeled containers and nodes with overlapping bounds. Do not call a model or dispatch taps.

**Contents:**

- `formatter.py` — preserve decision metadata during indexed formatting.

**Examples:** `DecisionFormatter` retains focus and clickable hints so `atomic_tools` can label a `tap_element` choice.

**Depends on / used by:** The atomic transform labels elements from the formatted tree. The decision selector maps those atomic tools to options.
