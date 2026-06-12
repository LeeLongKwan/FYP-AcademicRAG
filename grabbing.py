#!/usr/bin/env python3
# so101_arm.py — 2-link arm + gripper controller with SO‑101 tuned grasp
# Backends:
#   - UART (default for SO‑101 via Waveshare/MotorBus over USB)
#   - GPIO (pigpio PWM, for quick bring-up without the MotorBus)
#   - SIM (set SIM=1 env var for dry runs; prints commands)
#
# Examples:
#   python3 so101_arm.py --backend uart --port /dev/ttyUSB0 PICK
#   python3 so101_arm.py --backend gpio PICK
#   SIM=1 python3 so101_arm.py PICK
#
# SO‑101 references (LeRobot docs):
# - Install LeRobot + Feetech extra:  pip install -e ".[feetech]"
# - Find your USB port:               lerobot-find-port
# - Setup motor IDs/baud:             lerobot-setup-motors --robot.type=so101_follower --robot.port=...
# - Calibrate ranges:                 lerobot-calibrate --robot.type=so101_follower --robot.port=...


import os, sys, math, time, signal, argparse
from typing import Optional, Tuple

#config

#Geometry (mm). Measure pivot-to-pivot.
L1 = 135.0   # shoulder -> elbow
L2 = 120.0   # elbow -> fingertips/contact line

#Table/cube setup (mm)
Z_TABLE   = -40.0      # table height relative to shoulder pivot z=0
CUBE_SIZE = 30.0       # 3 cm cubes
CUBE_TOP  = Z_TABLE + CUBE_SIZE

#Waypoints (mm)
X_PREP   = 160.0
X_PICK   = 175.0
X_PLACE  = 120.0

#SO‑101 tuned grasp parameters (mm, µs)
SO101 = {
    "approach_above":    20.0,   # approach height over cube top
    "pre_grasp_gap":      2.0,   # stop above top before close
    "settle_forward":     5.0,   # small forward nudge
    "lift_check":        10.0,   # short verification lift
    "post_lift":         50.0,   # travel lift
    "micro_open_us":     20,     # tiny open before full release
    "preclose_step_us":  12,     # gentle pre-close step
    "preclose_dt":      0.06,    # slower pre-close cadence
    "squeeze_extra_us":  25,     # modest extra squeeze
}

#Logical channels (0-based). Map these to your backend.
CH_SHOULDER = 0
CH_ELBOW    = 1
CH_WRIST    = 2  
CH_GRIP     = 3

MAP = {
    "shoulder": {"min_deg": -90, "max_deg":  90, "min_us": 1100, "max_us": 1900},
    "elbow":    {"min_deg":   0, "max_deg": 160, "min_us": 1000, "max_us": 2000},
    "wrist":    {"min_deg": -90, "max_deg":  90, "min_us": 1100, "max_us": 1900},
}

#Safety limits (deg) inside mapping envelopes
LIM_SHOULDER = (-80, 80)
LIM_ELBOW    = ( 10,150)
LIM_WRIST    = (-70, 70)

#Gripper pulses (µs). 
GRIP_OPEN_US    = 1100
GRIP_CLOSE_US   = 2000
GRIP_STEP_US    = 15
GRIP_STEP_DT    = 0.05
GRIP_SQUEEZE_US = 30
GRIP_SETTLE_S   = 0.15

# Motion shaping
CART_STEPS_DEFAULT = 30
CART_DWELL_S       = 0.01

# GPIO mapping (only used for --backend gpio)
GPIO_MAP = {CH_SHOULDER:12, CH_ELBOW:13, CH_WRIST:19, CH_GRIP:18}
GPIO_FREQ_HZ = 50
 
def clamp(v, lo, hi): return max(lo, min(hi, v))
SIM = os.getenv("SIM","0") == "1"

#backend
class ServoBackend:
    def set_us(self, ch: int, pulse_us: int):
        raise NotImplementedError
    def close(self): pass

class SimBackend(ServoBackend):
    def set_us(self, ch: int, pulse_us: int):
        print(f"[SIM] CH{ch} -> {int(pulse_us)} us")
    def close(self): print("[SIM] close")

class PigpioPWM(ServoBackend):
    def __init__(self, ch_to_gpio: dict, freq_hz: int = 50):
        try:
            import pigpio
        except ImportError:
            print("Install pigpio: sudo apt-get install -y pigpio && sudo systemctl enable --now pigpiod")
            sys.exit(1)
        self.pi = pigpio.pi()
        if not self.pi.connected:
            print("pigpio daemon not running: sudo systemctl start pigpiod")
            sys.exit(1)
        self.map = ch_to_gpio
        self.freq = freq_hz
        for gpio in self.map.values():
            self.pi.set_mode(gpio, pigpio.ALT5)
            self.pi.set_PWM_frequency(gpio, self.freq)
    def set_us(self, ch: int, pulse_us: int):
        if ch not in self.map: return
        self.pi.set_servo_pulsewidth(self.map[ch], int(pulse_us))
    def close(self):
        for gpio in self.map.values():
            self.pi.set_servo_pulsewidth(gpio, 0)
        self.pi.stop()

class WaveshareUartPWM(ServoBackend):
    """
    Generic UART→PWM backend (USB serial). Many boards accept ASCII like:
    #CH01 P1500\r\n
    Some use different frames; if your board differs, edit write_frame().
    """
    def __init__(self, port: str, baud: int = 115200, channels: int = 16, ascii_proto: bool = True):
        try:
            import serial  # pyserial
        except ImportError:
            print("Install pyserial: pip install pyserial")
            sys.exit(1)
        self.ser = serial.Serial(port=port, baudrate=baud, timeout=0.05)
        self.channels = channels
        self.ascii_proto = ascii_proto
    def write_frame(self, ch: int, us: int):
        us = int(clamp(us, 500, 2500))
        if self.ascii_proto:
            line = f"#CH{ch:02d} P{us:04d}\r\n".encode()
            self.ser.write(line)
        else:
            #Example binary frame (placeholder — update if your board uses binary):
            #<0xFF 0xFF CH US_L US_H SUM>
            us_l = us & 0xFF; us_h = (us>>8) & 0xFF
            s = (ch + us_l + us_h) & 0xFF
            self.ser.write(bytes([0xFF,0xFF, ch & 0xFF, us_l, us_h, s]))
    def set_us(self, ch: int, pulse_us: int):
        if 0 <= ch < self.channels:
            self.write_frame(ch, int(pulse_us))
    def close(self):
        try: self.ser.close()
        except Exception: pass

#arm core 
class Arm2R:
    def __init__(self, io: ServoBackend, has_wrist=True):
        self.io = io
        self.has_wrist = has_wrist
        self.s_deg, self.e_deg, self.w_deg = 0.0, 90.0, 0.0
        self._apply_all()

    def _ang2us(self, name, deg):
        m = MAP[name]
        deg = clamp(deg, m["min_deg"], m["max_deg"])
        t = (deg - m["min_deg"]) / (m["max_deg"] - m["min_deg"])
        return int(m["min_us"] + t*(m["max_us"] - m["min_us"])), deg

    def _apply_all(self):
        # Shoulder
        d = clamp(self.s_deg, *LIM_SHOULDER); us, self.s_deg = self._ang2us("shoulder", d); self.io.set_us(CH_SHOULDER, us)
        # Elbow
        d = clamp(self.e_deg, *LIM_ELBOW);    us, self.e_deg = self._ang2us("elbow",    d); self.io.set_us(CH_ELBOW,    us)
        # Wrist (optional level-keeping)
        if self.has_wrist:
            d = clamp(self.w_deg, *LIM_WRIST);   us, self.w_deg = self._ang2us("wrist", d); self.io.set_us(CH_WRIST,   us)

    # IK: base at (0,0); x forward, z up. Elbow‑up solution.
    def ik(self, x, z):
        r2 = x*x + z*z; r = math.sqrt(r2)
        if r > L1+L2 or r < abs(L1-L2): return None
        cos_e = clamp((r2 - L1*L1 - L2*L2) / (2*L1*L2), -1.0, 1.0)
        e = math.degrees(math.acos(cos_e))
        k1 = L1 + L2*cos_e; k2 = L2*math.sin(math.radians(e))
        s = math.degrees(math.atan2(z, x) - math.atan2(k2, k1))
        w = -(s + e - 90.0)  # keep tool roughly level
        return s, e, w

    def move_to(self, x, z, steps=CART_STEPS_DEFAULT, dwell=CART_DWELL_S):
        sol = self.ik(x, z)
        if sol is None: raise ValueError(f"Out of reach: ({x:.1f},{z:.1f})")
        s2, e2, w2 = sol
        s1, e1, w1 = self.s_deg, self.e_deg, self.w_deg
        steps = max(1, int(steps))
        for i in range(1, steps+1):
            a = i/steps
            self.s_deg = s1 + a*(s2 - s1)
            self.e_deg = e1 + a*(e2 - e1)
            self.w_deg = w1 + a*(w2 - w1)
            self._apply_all(); time.sleep(dwell)

    def move_line(self, x1, z1, x2, z2, steps=40, dwell=0.01):
        for i in range(1, steps+1):
            a = i/steps
            self.move_to(x1 + a*(x2-x1), z1 + a*(z2-z1), steps=1, dwell=dwell)

    def move_forward(self, dx, x0, z0, steps=20):
        self.move_line(x0, z0, x0+dx, z0, steps=max(2, steps), dwell=0.01)
        return x0+dx, z0

    #Gripper helpers
    def grip_write(self, us: int):
        self.io.set_us(CH_GRIP, int(clamp(us, 500, 2500)))

    def open_gripper(self):
        self.grip_write(GRIP_OPEN_US); time.sleep(0.2)

    def close_guarded(self):
        us = GRIP_OPEN_US
        while us < GRIP_CLOSE_US:
            us = min(us + GRIP_STEP_US, GRIP_CLOSE_US)
            self.grip_write(us); time.sleep(GRIP_STEP_DT)
        time.sleep(GRIP_SETTLE_S)

    def home(self):
        self.s_deg, self.e_deg, self.w_deg = 0.0, 90.0, 0.0
        self._apply_all(); time.sleep(0.3)

#Grab and place
def close_guarded_so101(arm: Arm2R):
    us = GRIP_OPEN_US
    stop_at = max(GRIP_OPEN_US, GRIP_CLOSE_US - SO101["squeeze_extra_us"])
    while us < stop_at:
        us = min(us + SO101["preclose_step_us"], stop_at)
        arm.grip_write(us); time.sleep(SO101["preclose_dt"])
    time.sleep(0.12)
    arm.grip_write(min(GRIP_CLOSE_US, us + SO101["squeeze_extra_us"]))
    time.sleep(GRIP_SETTLE_S)

def verify_hold_so101(arm: Arm2R, x: float, z: float):
    arm.move_to(x, z + SO101["lift_check"])
    #Tiny corrective squeeze for pad compliance
    arm.grip_write(min(GRIP_CLOSE_US, GRIP_CLOSE_US - 5 + SO101["squeeze_extra_us"]))
    time.sleep(0.1)

def pick_sequence_so101(arm: Arm2R):
    #Wrist neutral for parallel contact
    if arm.has_wrist:
        arm.w_deg = 0.0; arm._apply_all(); time.sleep(0.1)

    #Open and approach
    arm.open_gripper()
    arm.move_to(X_PREP, CUBE_TOP + SO101["approach_above"])
    arm.move_to(X_PICK, CUBE_TOP + SO101["pre_grasp_gap"])

    #Seat the cube
    arm.move_forward(SO101["settle_forward"], X_PICK, CUBE_TOP + SO101["pre_grasp_gap"], steps=10)

    #Gentle two‑stage close, verify, lift
    close_guarded_so101(arm)
    verify_hold_so101(arm, X_PICK, CUBE_TOP + SO101["pre_grasp_gap"])
    arm.move_to(X_PICK, CUBE_TOP + SO101["post_lift"])

    #Transfer and place
    arm.move_to(X_PLACE, CUBE_TOP + SO101["post_lift"])
    arm.move_to(X_PLACE, CUBE_TOP + 5.0)

    #Gentle release
    arm.grip_write(max(GRIP_OPEN_US, GRIP_CLOSE_US - SO101["micro_open_us"])); time.sleep(0.08)
    arm.open_gripper()

    #Retreat and home
    arm.move_to(X_PREP, CUBE_TOP + SO101["post_lift"])
    arm.home()

#CLI 
def print_usage():
    print("Usage:")
    print("  PICK                         # SO‑101 tuned pick/place for 3 cm cube")
    print("  OPEN                         # open gripper")
    print("  CLOSE                        # close gripper (simple)")
    print("  HOME                         # neutral pose")
    print("  GOTO  x  z                   # Cartesian move (mm)")
    print("  LINE  x1 z1  x2 z2 [steps]   # Cartesian straight line")
    print("  FORWARD dx x0 z0             # forward dx at z0 starting from (x0,z0)")
    print("Examples:")
    print("  python3 so101_arm.py --backend uart --port /dev/ttyUSB0 PICK")
    print("  python3 so101_arm.py --backend gpio GOTO 160 10")
    print("  SIM=1 python3 so101_arm.py PICK")

def build_backend(args) -> ServoBackend:
    if SIM:
        return SimBackend()
    if args.backend == "gpio":
        return PigpioPWM(GPIO_MAP, freq_hz=GPIO_FREQ_HZ)
    if args.backend == "uart":
        if not args.port:
            print("Please provide --port for UART backend (e.g., /dev/ttyUSB0).")
            sys.exit(1)
        return WaveshareUartPWM(args.port, baud=args.baud, channels=args.channels, ascii_proto=not args.binary)
    print("Unknown backend. Use --backend uart|gpio, or set SIM=1.")
    sys.exit(1)

def main():
    ap = argparse.ArgumentParser(description="SO‑101 2‑link arm + gripper controller (UART/GPIO/SIM).")
    ap.add_argument("--backend", choices=["uart","gpio"], default="uart", help="Control backend (default: uart).")
    ap.add_argument("--port", help="Serial device for UART backend, e.g., /dev/ttyUSB0 or /dev/ttyACM0.")
    ap.add_argument("--baud", type=int, default=115200, help="UART baudrate (default 115200).")
    ap.add_argument("--channels", type=int, default=16, help="Num channels (UART backend).")
    ap.add_argument("--binary", action="store_true", help="Use example binary frame instead of ASCII in UART backend.")
    ap.add_argument("cmd", nargs="+", help="Command and args. See usage.")
    args = ap.parse_args()

    io = build_backend(args)
    arm = Arm2R(io, has_wrist=True)

    def shutdown(*_):
        io.close()
        print("\n[SHUTDOWN]")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown); signal.signal(signal.SIGTERM, shutdown)

    try:
        c = args.cmd[0].upper()
        if c == "PICK":
            pick_sequence_so101(arm); shutdown()
        elif c == "OPEN":
            arm.open_gripper(); shutdown()
        elif c == "CLOSE":
            arm.close_guarded(); shutdown()
        elif c == "HOME":
            arm.home(); shutdown()
        elif c == "GOTO":
            if len(args.cmd) < 3: print_usage(); shutdown()
            x = float(args.cmd[1]); z = float(args.cmd[2]); arm.move_to(x, z); shutdown()
        elif c == "LINE":
            if len(args.cmd) < 5: print_usage(); shutdown()
            x1=float(args.cmd[1]); z1=float(args.cmd[2]); x2=float(args.cmd[3]); z2=float(args.cmd[4])
            steps=int(args.cmd[5]) if len(args.cmd)>5 else 40
            arm.move_line(x1,z1,x2,z2,steps=steps,dwell=0.01); shutdown()
        elif c == "FORWARD":
            if len(args.cmd) < 4: print_usage(); shutdown()
            dx=float(args.cmd[1]); x0=float(args.cmd[2]); z0=float(args.cmd[3])
            arm.move_to(x0,z0); arm.move_forward(dx,x0,z0); shutdown()
        else:
            print_usage(); shutdown()
    except Exception as e:
        print("[ERROR]", e); shutdown()

if __name__ == "__main__":
    main()