import { createHash } from 'node:crypto';
import { curlRequest, decodeJson } from './http.mjs';
import { prepareInputVerification } from './input-verification.mjs';

export const GLOBAL_ACTIONS = { back: 1, home: 2, recent: 3 };
export const KEYS = { back: 4, tab: 61, enter: 66, delete: 67, forward_delete: 112 };

export class StaleObservationError extends Error {}

function targetMeaning(observation, id) {
  const target = observation.elements.find((element) => element.id === id);
  if (!target) return null;
  const meaning = Object.fromEntries(Object.entries(target).filter(([key]) => key !== 'bounds'));
  return JSON.stringify({
    ...meaning,
    content: observation.elements
      .filter((e) => e.id.startsWith(id + '.'))
      .map(({ id, text, label, resourceId, editable, enabled, checked, selected }) => ({
        id,
        text,
        label,
        resourceId,
        editable,
        enabled,
        checked,
        selected,
      })),
  });
}

export function assertFresh(current, expected, action, maxAgeMs = 30_000) {
  if (
    !expected ||
    current.deviceId !== expected.deviceId ||
    !Number.isFinite(expected.observedAt) ||
    Date.now() - expected.observedAt > maxAgeMs ||
    expected.observedAt > Date.now() ||
    current.phone.packageName !== expected.phone?.packageName ||
    JSON.stringify(current.screen) !== JSON.stringify(expected.screen)
  ) {
    throw new StaleObservationError(
      'Screen changed or observation expired; observe and decide again.',
    );
  }
  let fresh;
  if (action.type === 'tap-element') {
    fresh =
      targetMeaning(expected, action.elementId) !== null &&
      targetMeaning(current, action.elementId) === targetMeaning(expected, action.elementId);
  } else if (['type', 'clear', 'key'].includes(action.type) && expected.phone.inputElementId) {
    const id = expected.phone.inputElementId;
    fresh =
      current.phone.isEditable &&
      current.phone.inputElementId === id &&
      targetMeaning(current, id) === targetMeaning(expected, id);
  } else if (action.type === 'global' && action.name === 'home') {
    fresh = true; // HOME is independent of in-app content such as clocks and animations.
  } else if (action.type === 'global') {
    const navigationMeaning = (state) =>
      JSON.stringify({
        phone: state.phone,
        controls: state.elements
          .filter((e) => e.clickable || e.editable)
          .map((e) => targetMeaning(state, e.id)),
        headings: state.elements
          .filter((e) => e.resourceId?.endsWith(':id/title') || e.label)
          .map((e) => [e.id, e.text, e.label]),
      });
    fresh = navigationMeaning(current) === navigationMeaning(expected);
  } else if (action.type === 'swipe' && action.regionId) {
    const before = expected.elements.find((e) => e.id === action.regionId);
    const after = current.elements.find((e) => e.id === action.regionId);
    fresh = before && after?.enabled && after.scrollable && before.resourceId === after.resourceId;
  } else {
    fresh = current.fingerprint === expected.fingerprint;
  }
  if (!fresh)
    throw new StaleObservationError(
      'Screen changed or observation expired; observe and decide again.',
    );
}

function integer(value, name, min = 0) {
  if (!Number.isSafeInteger(value) || value < min)
    throw new Error(`${name} must be an integer >= ${min}.`);
}

export function summarizeState(raw, deviceId) {
  const screen = raw?.device_context?.screen_bounds;
  if (
    !screen ||
    !Number.isSafeInteger(screen.width) ||
    !Number.isSafeInteger(screen.height) ||
    screen.width < 1 ||
    screen.height < 1 ||
    !raw.phone_state ||
    !raw.a11y_tree
  ) {
    throw new Error('UI state is missing its tree, phone state, or valid screen bounds.');
  }
  const elements = [];
  const visit = (node, path) => {
    if (!node || typeof node !== 'object') return;
    const b = node.boundsInScreen;
    const bounds =
      b &&
      Object.values({ left: b.left, top: b.top, right: b.right, bottom: b.bottom }).every(
        Number.isFinite,
      )
        ? {
            left: Math.max(0, b.left),
            top: Math.max(0, b.top),
            right: Math.min(screen.width, b.right),
            bottom: Math.min(screen.height, b.bottom),
          }
        : null;
    if (
      node.isVisibleToUser !== false &&
      bounds &&
      bounds.right > bounds.left &&
      bounds.bottom > bounds.top
    ) {
      const password = node.isPassword === true;
      const text = password ? '[password]' : node.text || '';
      const label = password ? '' : node.contentDescription || '';
      const editable =
        node.isEditable === true ||
        [
          'android.widget.EditText',
          'android.widget.AutoCompleteTextView',
          'android.widget.MultiAutoCompleteTextView',
        ].includes(node.className);
      if (text || label || node.isClickable || editable || node.isScrollable) {
        elements.push({
          id: path,
          text,
          label,
          resourceId: node.resourceId || '',
          hint: node.hint || '',
          bounds,
          clickable: node.isClickable === true,
          editable,
          scrollable: node.isScrollable === true,
          enabled: node.isEnabled !== false,
          focused: node.isFocused === true,
          password,
          checkable: node.isCheckable === true,
          checked: node.isChecked === true,
          selected: node.isSelected === true,
        });
      }
    }
    if (Array.isArray(node.children))
      node.children.forEach((child, i) => visit(child, `${path}.${i}`));
  };
  visit(raw.a11y_tree, 'ui');
  const inputs = elements.filter((e) => e.editable && e.enabled);
  const focusedInputs = inputs.filter((e) => e.focused);
  const input =
    focusedInputs.length === 1
      ? focusedInputs[0]
      : raw.phone_state.keyboardVisible && inputs.length === 1
        ? inputs[0]
        : undefined;
  const phone = {
    packageName: raw.phone_state.packageName || '',
    currentApp: raw.phone_state.currentApp || '',
    isEditable: raw.phone_state.isEditable === true || Boolean(input),
    inputElementId: input?.id,
    focusEvidence: raw.phone_state.isEditable
      ? 'reported'
      : input?.focused
        ? 'focused-node'
        : input
          ? 'single-input-with-keyboard'
          : 'none',
    keyboardVisible: raw.phone_state.keyboardVisible === true,
    focusedElement: {
      resourceId: raw.phone_state.focusedElement?.resourceId || '',
      className: raw.phone_state.focusedElement?.className || '',
    },
  };
  const content = { deviceId, phone, screen, elements };
  return {
    ...content,
    fingerprint: createHash('sha256').update(JSON.stringify(content)).digest('hex'),
    observedAt: Date.now(),
  };
}

export class MobilerunDevice {
  constructor({
    apiKey = process.env.MOBILERUN_API_KEY || process.env.MOBILERUN_CLOUD_API_KEY,
    deviceId = process.env.MOBILERUN_DEVICE_ID,
    baseUrl = process.env.MOBILERUN_BASE_URL || 'https://api.mobilerun.ai/v1',
    textCompletionMode = process.env.MOBILERUN_TEXT_COMPLETION_MODE || 'accepted',
    request = curlRequest,
  } = {}) {
    this.apiKey = apiKey;
    this.deviceId = deviceId;
    const url = new URL(baseUrl);
    if (url.protocol !== 'https:' || url.username || url.password || url.search || url.hash)
      throw new Error('MOBILERUN_BASE_URL must be an HTTPS API base URL.');
    this.baseUrl = url.href.replace(/\/$/, '');
    this.request = request;
    if (!['accepted', 'committed'].includes(textCompletionMode))
      throw new Error('Text completion mode must be accepted or committed.');
    this.textCompletionMode = textCompletionMode;
    this.readyAt = -Infinity;
    this.installedApps = [];
  }

  async api(path, method = 'GET', body, binary = false) {
    const bytes = await this.request({
      url: `${this.baseUrl}${path}`,
      apiKey: this.apiKey,
      method,
      body,
    });
    return binary ? bytes : decodeJson(bytes);
  }

  path(suffix = '') {
    if (!this.deviceId)
      throw new Error('Set MOBILERUN_DEVICE_ID or pass --device. Use devices to list IDs.');
    return `/devices/${encodeURIComponent(this.deviceId)}${suffix}`;
  }

  async listDevices() {
    const devices = [];
    for (let page = 1; page <= 1000; page++) {
      const result = await this.api(`/devices?page=${page}&pageSize=100`);
      if (!Array.isArray(result?.items)) throw new Error('Invalid device list response.');
      devices.push(...result.items.map(({ id, name, state }) => ({ id, name, state })));
      if (result.items.length < 100) return devices;
    }
    throw new Error('Device pagination exceeded 1000 pages.');
  }

  async listApps() {
    const result = await this.api(this.path('/apps?includeSystemApps=true'));
    if (!Array.isArray(result)) throw new Error('Invalid installed-app response.');
    this.installedApps = result
      .filter((app) => typeof app.packageName === 'string' && typeof app.label === 'string')
      .map(({ packageName, label }) => ({ packageName, label }));
    return this.installedApps;
  }

  async assertReady() {
    const device = await this.api(this.path());
    if (device?.state !== 'ready')
      throw new Error(`Device is ${device?.state || 'unknown'}; it must be ready.`);
    this.readyAt = performance.now();
    return { id: device.id, name: device.name, state: device.state };
  }

  async observe() {
    // Keep noninteractive text: it is evidence for goal completion and control labels.
    return summarizeState(await this.api(this.path('/ui-state?filter=false')), this.deviceId);
  }

  async screenshot() {
    const bytes = await this.api(this.path('/screenshot?hideOverlay=true'), 'GET', undefined, true);
    if (!bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])))
      throw new Error('Screenshot response is not a PNG.');
    return bytes;
  }

  async act(action, { expected, maxAgeMs = 30_000 } = {}) {
    if (!action || typeof action !== 'object') throw new Error('An action object is required.');
    if (performance.now() - this.readyAt > 30_000) await this.assertReady();
    if (action.type === 'open-app') {
      if (!this.installedApps.some((app) => app.packageName === action.packageName))
        throw new Error('The app was not observed in the installed-app list.');
      if (expected && expected.deviceId !== this.deviceId)
        throw new Error('Observation belongs to a different device.');
      await this.api(this.path(`/apps/${encodeURIComponent(action.packageName)}`), 'PUT', {});
      return;
    }
    const current = await this.observe();
    if (expected) assertFresh(current, expected, action, maxAgeMs);
    const point = (x, y) => {
      integer(x, 'x');
      integer(y, 'y');
      if (x >= current.screen.width || y >= current.screen.height)
        throw new Error('Coordinates are outside the screen.');
    };
    let suffix,
      method = 'POST',
      body,
      inputVerification;
    switch (action.type) {
      case 'tap':
        point(action.x, action.y);
        suffix = '/tap';
        body = { x: action.x, y: action.y };
        break;
      case 'tap-element': {
        if (!expected) throw new Error('Element taps require their original observation.');
        const node = current.elements.find((entry) => entry.id === action.elementId);
        if (!node?.enabled || !(node.clickable || node.editable))
          throw new Error('Element is not actionable.');
        const { left, top, right, bottom } = node.bounds;
        suffix = '/tap';
        body = { x: Math.floor((left + right) / 2), y: Math.floor((top + bottom) / 2) };
        break;
      }
      case 'swipe':
        point(action.startX, action.startY);
        point(action.endX, action.endY);
        integer(action.duration ?? 300, 'duration', 10);
        suffix = '/swipe';
        body = {
          startX: action.startX,
          startY: action.startY,
          endX: action.endX,
          endY: action.endY,
          duration: action.duration ?? 300,
        };
        if (action.regionId && expected) {
          const before = expected.elements.find((e) => e.id === action.regionId).bounds;
          const after = current.elements.find((e) => e.id === action.regionId).bounds;
          // Resolve the same relative gesture in current geometry (e.g. a collapsing toolbar).
          const project = (value, oldStart, oldEnd, newStart, newEnd) => {
            const fraction = (value - oldStart) / (oldEnd - oldStart);
            if (!Number.isFinite(fraction) || fraction < 0 || fraction >= 1)
              throw new Error('Swipe leaves its observed region.');
            return Math.floor(newStart + fraction * (newEnd - newStart));
          };
          body.startX = project(action.startX, before.left, before.right, after.left, after.right);
          body.endX = project(action.endX, before.left, before.right, after.left, after.right);
          body.startY = project(action.startY, before.top, before.bottom, after.top, after.bottom);
          body.endY = project(action.endY, before.top, before.bottom, after.top, after.bottom);
          point(body.startX, body.startY);
          point(body.endX, body.endY);
        }
        break;
      case 'type':
        if (!current.phone.isEditable) throw new Error('Focus an editable field before typing.');
        if (
          typeof action.text !== 'string' ||
          (action.clear !== undefined && typeof action.clear !== 'boolean')
        )
          throw new Error('Text must be a string and clear must be boolean.');
        suffix = '/keyboard';
        inputVerification =
          this.textCompletionMode === 'accepted' ? prepareInputVerification(current, action) : null;
        body = {
          text: action.text,
          clear: action.clear ?? false,
          completionMode: inputVerification ? 'accepted' : 'committed',
        };
        break;
      case 'clear':
        if (!current.phone.isEditable) throw new Error('Focus an editable field before clearing.');
        suffix = '/keyboard';
        method = 'DELETE';
        break;
      case 'key':
        if (!Object.hasOwn(KEYS, action.key)) throw new Error('Unsupported keyboard key.');
        suffix = '/keyboard';
        method = 'PUT';
        body = { key: KEYS[action.key] };
        break;
      case 'global':
        if (!Object.hasOwn(GLOBAL_ACTIONS, action.name))
          throw new Error('Unsupported global action.');
        suffix = '/global';
        body = { action: GLOBAL_ACTIONS[action.name] };
        break;
      default:
        throw new Error('Unsupported action type.');
    }
    await this.api(this.path(suffix), method, body);
    return inputVerification ? { inputVerification } : undefined;
  }
}
