import { test } from 'node:test';
import assert from 'node:assert/strict';
import { darkThemeState } from './demo-verifiers.mjs';

test('theme verifier requires a real switch and the Settings app, not a DONE claim or label alone', () => {
  const state = {
    phone: { packageName: 'com.android.settings' },
    elements: [{ label: 'Dark theme', checkable: true, checked: true }],
  };
  assert.equal(darkThemeState(state), true);
  state.elements[0].checked = false;
  assert.equal(darkThemeState(state), false);
  state.elements[0].checkable = false;
  assert.equal(darkThemeState(state), null);
  state.elements[0].checkable = true;
  state.phone.packageName = 'other';
  assert.equal(darkThemeState(state), null);
  state.phone.packageName = 'com.android.settings';
  state.elements.push({ ...state.elements[0] });
  assert.equal(darkThemeState(state), null);
});
