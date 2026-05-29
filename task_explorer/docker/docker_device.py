"""DockerDevice: Device subclass that connects via ADB-over-TCP to a Docker container."""

import logging
import subprocess
import time

from task_explorer.utils.device import Device


class DockerDevice(Device):
  """Device that connects to an Android emulator inside a Docker container
  via ADB-over-TCP (port 5555 by default)."""

  def __init__(
      self,
      docker_host: str = "localhost",
      adb_port: int = 5555,
      connect_timeout: float = 30.0,
  ) -> None:
    self.docker_host = docker_host
    self.adb_port = adb_port
    self.device_serial = f"{docker_host}:{adb_port}"
    self.logger = logging.getLogger(self.__class__.__name__)
    self.u2d = None
    self.ensure_adb_connected(timeout=connect_timeout)
    self.connect()

  def ensure_adb_connected(self, timeout: float = 30.0) -> None:
    """Run `adb connect host:port` and wait until the device appears."""
    target = self.device_serial
    self.logger.info(f"Ensuring ADB connection to {target} ...")
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
      try:
        result = subprocess.run(
            ["adb", "connect", target],
            capture_output=True,
            timeout=10,
        )
        output = result.stdout.decode("utf-8", errors="replace").strip()
        self.logger.info(f"adb connect output: {output}")
        if "connected" in output.lower():
          # Verify the device is listed
          devices_result = subprocess.run(
              ["adb", "devices"],
              capture_output=True,
              timeout=10,
          )
          devices_out = devices_result.stdout.decode(
              "utf-8", errors="replace"
          )
          if target in devices_out:
            self.logger.info(f"ADB connected to {target}")
            return
      except Exception as e:
        last_err = e
        self.logger.warning(f"adb connect attempt failed: {e}")
      time.sleep(2)
    raise ConnectionError(
        f"Failed to connect to {target} within {timeout}s. Last error: {last_err}"
    )
