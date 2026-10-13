"""Portable DeskBuddy bootstrap; importing this package starts no services."""

from deskbuddy.settings import Settings, SettingsError, load_settings

__all__ = ["Settings", "SettingsError", "load_settings"]
