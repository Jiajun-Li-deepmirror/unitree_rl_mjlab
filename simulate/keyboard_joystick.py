#!/usr/bin/env python3
"""Keyboard-driven virtual Xbox 360 gamepad for unitree_rl_mjlab's FSM.

simulate/ and deploy/robots/g1/ read a real joystick device (/dev/input/jsN)
via physics_joystick.h's XBoxJoystick, which expects the standard Linux
joydev button/axis ordering for an Xbox 360 pad:
    button_: 0=A 1=B 2=X 3=Y 4=LB 5=RB 6=back 7=start
    axis_:   0=lx 1=ly 2=LT(trigger) 3=rx 4=ry 5=RT(trigger) 6=dpad-x 7=dpad-y

This script creates a uinput device advertising exactly that capability
set (in ascending evdev-code order, which is what joydev uses to assign
indices), so it enumerates as /dev/input/js0 with no code changes needed
anywhere else. A raw keyboard combo like "hold LT, tap up" can't be typed
naturally on a terminal (no true "held key" state over stdin), so each
number key here fires a complete synthetic press-hold-release of the
whole combo instead of requiring you to hold two real keys together.

Setup (run yourself, needs root for /dev/uinput):
    sudo apt install python3-evdev
    sudo python3 simulate/keyboard_joystick.py

Start this BEFORE unitree_mujoco / g1_ctrl, and leave it running in its
own terminal. Then in that terminal:
    1  Passive -> FixStand              (LT + up)
    2  FixStand or Mimic_* -> Velocity  (RT + A)
    3  Velocity -> Mimic_Dance2_v3      (RB + Y)
    4  Velocity -> Mimic_Dance1_subject2 (RB + A)
    0  ANY -> Passive  [EMERGENCY STOP]  (LT + B)
    q  quit this script (removes the virtual joystick)
"""
import sys
import termios
import time
import tty

from evdev import AbsInfo, UInput
from evdev import ecodes as e

CAPABILITIES = {
    e.EV_KEY: [
        e.BTN_A, e.BTN_B, e.BTN_X, e.BTN_Y,
        e.BTN_TL, e.BTN_TR, e.BTN_SELECT, e.BTN_START,
    ],
    e.EV_ABS: [
        (e.ABS_X, AbsInfo(0, -32768, 32767, 0, 0, 0)),
        (e.ABS_Y, AbsInfo(0, -32768, 32767, 0, 0, 0)),
        (e.ABS_Z, AbsInfo(0, 0, 255, 0, 0, 0)),
        (e.ABS_RX, AbsInfo(0, -32768, 32767, 0, 0, 0)),
        (e.ABS_RY, AbsInfo(0, -32768, 32767, 0, 0, 0)),
        (e.ABS_RZ, AbsInfo(0, 0, 255, 0, 0, 0)),
        (e.ABS_HAT0X, AbsInfo(0, -1, 1, 0, 0, 0)),
        (e.ABS_HAT0Y, AbsInfo(0, -1, 1, 0, 0, 0)),
    ],
}

HOLD = 0.20
GAP = 0.08


def make_device():
    ui = UInput(
        CAPABILITIES,
        name="Virtual Xbox 360 Controller",
        bustype=e.BUS_USB,
        vendor=0x045E,
        product=0x028E,
        version=1,
    )
    print(f"Virtual joystick created at: {ui.device.path}")
    time.sleep(1.0)  # let joydev enumerate /dev/input/jsN before use
    return ui


def hold_axis_tap_button(ui, axis, axis_val, btn):
    ui.write(e.EV_ABS, axis, axis_val); ui.syn()
    time.sleep(GAP)
    ui.write(e.EV_KEY, btn, 1); ui.syn()
    time.sleep(HOLD)
    ui.write(e.EV_KEY, btn, 0); ui.syn()
    ui.write(e.EV_ABS, axis, 0); ui.syn()


def hold_axis_tap_hat(ui, axis, axis_val, hat, hat_val):
    ui.write(e.EV_ABS, axis, axis_val); ui.syn()
    time.sleep(GAP)
    ui.write(e.EV_ABS, hat, hat_val); ui.syn()
    time.sleep(HOLD)
    ui.write(e.EV_ABS, hat, 0); ui.syn()
    ui.write(e.EV_ABS, axis, 0); ui.syn()


def hold_button_tap_button(ui, hold_btn, tap_btn):
    ui.write(e.EV_KEY, hold_btn, 1); ui.syn()
    time.sleep(GAP)
    ui.write(e.EV_KEY, tap_btn, 1); ui.syn()
    time.sleep(HOLD)
    ui.write(e.EV_KEY, tap_btn, 0); ui.syn()
    ui.write(e.EV_KEY, hold_btn, 0); ui.syn()


def main():
    ui = make_device()

    macros = {
        "1": (lambda: hold_axis_tap_hat(ui, e.ABS_Z, 255, e.ABS_HAT0Y, -1), "LT+up -> FixStand"),
        "2": (lambda: hold_axis_tap_button(ui, e.ABS_RZ, 255, e.BTN_A), "RT+A -> Velocity"),
        "3": (lambda: hold_button_tap_button(ui, e.BTN_TR, e.BTN_Y), "RB+Y -> Mimic_Dance2_v3"),
        "4": (lambda: hold_button_tap_button(ui, e.BTN_TR, e.BTN_A), "RB+A -> Mimic_Dance1_subject2"),
        "0": (lambda: hold_axis_tap_button(ui, e.ABS_Z, 255, e.BTN_B), "LT+B -> Passive [E-STOP]"),
    }

    print("\nReady. Keys:")
    for k, (_, desc) in macros.items():
        print(f"  {k}  {desc}")
    print("  q  quit\n")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == "q":
                break
            if ch in macros:
                fn, desc = macros[ch]
                print(f"-> {desc}")
                fn()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        ui.close()
        print("Virtual joystick removed.")


if __name__ == "__main__":
    main()
