"""FakeAPK: drop-in replacement for androguard APK that queries app info via ADB shell."""

import re
from typing import Optional

from task_explorer.utils.device import Device


class FakeAPK:
  """Queries app information via ADB shell commands instead of downloading
  and parsing the APK with androguard.

  Provides the same interface as androguard.core.apk.APK so it can be
  used as a direct replacement for ``apk_object`` in the exploration pipeline.
  """

  def __init__(self, package_name: str, device: Device) -> None:
    self._package_name = package_name
    self._device = device
    self._dumpsys_cache: Optional[str] = None

  def _dumpsys(self) -> str:
    if self._dumpsys_cache is None:
      output, _ = self._device.run_shell_command(
          f"dumpsys package {self._package_name}", timeout=30
      )
      self._dumpsys_cache = output
    return self._dumpsys_cache

  def get_package(self) -> str:
    return self._package_name

  def get_app_name(self) -> str:
    """Try to resolve the human-readable app label."""
    output, _ = self._device.run_shell_command(
        f"cmd package resolve-activity --brief {self._package_name}", timeout=15
    )
    # The brief output typically has the activity on the last non-empty line.
    # Fall back to package name if we cannot resolve.
    lines = [l.strip() for l in output.strip().splitlines() if l.strip()]
    if lines:
      # Try to get label from the app_label field
      label_match = re.search(r"label=(.*)", self._dumpsys())
      if label_match:
        return label_match.group(1).strip().strip("'\"")
    # Fallback: use the last component of the package name
    return self._package_name.split(".")[-1].capitalize()

  def get_activities(self) -> list[str]:
    """Return all declared activities for the package."""
    dumpsys = self._dumpsys()
    activities = []
    # Match lines like "      f2a81a7 com.google.android.contacts/.activities.PeopleActivity"
    pattern = re.compile(
        r"^\s+[0-9a-f]+\s+(" + re.escape(self._package_name) + r"/\S+)",
        re.MULTILINE,
    )
    for match in pattern.finditer(dumpsys):
      full = match.group(1)
      # Expand shorthand: "pkg/.Foo" -> "pkg.Foo"
      if "/" in full:
        pkg, cls = full.split("/", 1)
        if cls.startswith("."):
          cls = pkg + cls
        activities.append(cls)
      else:
        activities.append(full)
    # De-dup while preserving order
    seen = set()
    unique = []
    for a in activities:
      if a not in seen:
        seen.add(a)
        unique.append(a)
    return unique

  def get_main_activity(self) -> str:
    """Resolve the main/launcher activity."""
    output, _ = self._device.run_shell_command(
        f"cmd package resolve-activity --brief -c android.intent.category.LAUNCHER {self._package_name}",
        timeout=15,
    )
    lines = [l.strip() for l in output.strip().splitlines() if l.strip()]
    if len(lines) >= 2:
      # Second line is usually the fully-qualified activity
      return lines[-1]
    # Fallback: first activity in the list
    activities = self.get_activities()
    return activities[0] if activities else ""

  def get_androidversion_code(self) -> str:
    match = re.search(r"versionCode=(\d+)", self._dumpsys())
    return match.group(1) if match else "0"

  def get_androidversion_name(self) -> str:
    match = re.search(r"versionName=([^\s]+)", self._dumpsys())
    return match.group(1) if match else "0"
