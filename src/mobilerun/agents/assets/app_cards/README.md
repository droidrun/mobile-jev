# App Cards

App cards provide app-specific guidance to Mobilerun agents. They help agents understand how to operate specific apps more effectively.

## How It Works

1. **One folder per platform**: `android/` and `ios/`, each with its own `app_cards.json`
2. **Mapping File**: each `app_cards.json` maps an Android package name or iOS bundle id to a markdown file in that folder
3. **App Card Files**: Markdown files containing app-specific guidance
4. **Automatic Loading**: Mobilerun loads the card for the current app from the folder of the device's platform
5. **Prompt Injection**: App cards are injected into agent prompts when available

## File Structure

```
assets/app_cards/
├── android/
│   ├── app_cards.json   # Android package name → file mapping
│   └── gmail.md
└── ios/
    └── app_cards.json   # iOS bundle id → file mapping
```

Add separate entries and cards for each platform you support. The tree above shows the bundled files; other cards can be added in your own directory.

## Creating App Cards

### 1. Add an entry to the platform's app_cards.json

`android/app_cards.json`:

```json
{
  "com.google.android.gm": "gmail.md",
  "com.whatsapp": "social/whatsapp.md"
}
```

### 2. Create the markdown file

Create a `.md` file with guidance about the app in the same folder:

```markdown
# App Name Guide

## Navigation
- How to navigate the app
- Key screens and menus

## Common Actions
- List of common tasks
- How to perform them

## Tips
- App-specific tips
- Known issues or quirks
```

## Path Resolution

- Card paths are relative to the platform folder (`android/` or `ios/`), or absolute.
- An `app_cards.json` directly in `app_cards_dir` is not read; Mobilerun logs a warning and ignores it.
- A relative `app_cards_dir` is used from the working directory if it contains app cards there, otherwise from the package directory.

## Finding Package Names

To find an app's package name:

1. **Using ADB**:
   ```bash
   adb shell pm list packages | grep keyword
   ```

2. **From device**:
   - Open the app
   - Run Mobilerun with debug mode to see the current package name in logs

3. **Common apps**:
   - Gmail: `com.google.android.gm`
   - Chrome: `com.android.chrome`
   - WhatsApp: `com.whatsapp`
   - Instagram: `com.instagram.android`
   - Facebook: `com.facebook.katana`

## Configuration

Enable/disable app cards in `config.yaml`:

```yaml
agent:
  app_cards:
    enabled: true
    app_cards_dir: assets/app_cards
```

## Best Practices

1. **Be Concise**: Keep app cards focused and actionable
2. **Use Examples**: Show concrete examples of common tasks
3. **Update Regularly**: Keep app cards current with app updates
4. **Test**: Verify that guidance actually helps agents
5. **Organize**: Use subdirectories for related apps (e.g., social/, banking/)

## Programmatic Usage

```python
from mobilerun.agents import AppCardContextProvider, StepContext

provider = AppCardContextProvider(mode="local")
step = StepContext(
    package_name="com.google.android.gm", activity_name="", instruction="Summarize email",
    platform="android", step=1, ui_elements=[], last_actions=[], screenshot=None,
)
card = await provider.get_context(step)
```

Without `server_url`, `mode="server"` and `mode="composite"` log a warning and fall back to local cards.
