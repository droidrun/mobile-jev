// Jev selects source text; it never invents a field value. Explicit --text values take precedence.
export function textCandidates(goal, supplied = []) {
  if (supplied.length)
    return { values: [...new Set(supplied)], source: 'supplied', overflow: false };
  const words = [...goal.matchAll(/\S+/gu)];
  const values = new Set();
  for (let length = 1; length <= Math.min(8, words.length); length++) {
    for (let start = 0; start + length <= words.length; start++) {
      const end = words[start + length - 1];
      const value = goal
        .slice(words[start].index, end.index + end[0].length)
        .replace(/^["'“‘([{]+|["'”’)\]},.!?;:]+$/gu, '')
        .trim();
      if (value) values.add(value);
      if (values.size > 254) return { values: [], source: 'goal', overflow: true };
    }
  }
  return { values: [...values], source: 'goal', overflow: false };
}
