export function darkThemeState(observation) {
  if (observation?.phone?.packageName !== 'com.android.settings') return null;
  const toggles = observation.elements.filter(
    (e) => e.checkable && /^dark theme$/i.test((e.label || e.text || '').trim()),
  );
  // Fail closed if the screen has no explicit, unambiguous toggle evidence.
  return toggles.length === 1 ? toggles[0].checked === true : null;
}
