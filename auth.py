"""
Basic username/password login for Streamlit.

* Users live in Streamlit secrets (never in the repo):  .streamlit/secrets.toml
      [users]
      alice = "pbkdf2$200000$<salt>$<hash>"        # create with: python make_password.py
* Passwords are stored as salted PBKDF2-SHA256 hashes, compared in constant time.
* 5 wrong attempts for a username -> locked for 15 minutes (shared across all sessions).
* Local development without secrets:  set environment variable BACKTEST_NO_AUTH=1
"""
import hashlib
import hmac
import os
import time

import streamlit as st

ITERATIONS = 200_000
MAX_FAILS = 5
LOCK_SECONDS = 15 * 60
_DUMMY = "pbkdf2$%d$%s$%s" % (ITERATIONS, "00" * 16, "00" * 32)  # equalises timing for unknown users


def hash_password(password: str, salt: bytes | None = None, iterations: int = ITERATIONS) -> str:
    salt = salt or os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2${iterations}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, it, salt_hex, hash_hex = stored.split("$")
        if scheme != "pbkdf2":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(it))
        return hmac.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False


@st.cache_resource
def _attempts() -> dict:
    """username -> [fail_count, first_fail_or_lock_time]; shared by every session of this app process."""
    return {}


def _locked_for(user: str) -> int:
    rec = _attempts().get(user)
    if not rec:
        return 0
    fails, t0 = rec
    if fails >= MAX_FAILS:
        left = int(t0 + LOCK_SECONDS - time.time())
        if left > 0:
            return left
        _attempts().pop(user, None)  # lock expired
    elif time.time() - t0 > LOCK_SECONDS:
        _attempts().pop(user, None)
    return 0


def _record_fail(user: str):
    rec = _attempts().get(user)
    if rec and rec[0] < MAX_FAILS:
        rec[0] += 1
        if rec[0] >= MAX_FAILS:
            rec[1] = time.time()  # start lock window now
    elif not rec:
        _attempts()[user] = [1, time.time()]


def _users() -> dict:
    try:
        return {k.lower(): v for k, v in dict(st.secrets["users"]).items()}
    except Exception:
        return {}


def logout():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()


def require_login() -> str:
    """Call at the top of app.py. Returns the username, or stops the script."""
    if os.environ.get("BACKTEST_NO_AUTH") == "1":
        return "local"
    if st.session_state.get("auth_user"):
        return st.session_state["auth_user"]

    users = _users()
    st.title("🔒 เข้าสู่ระบบ")
    if not users:
        st.error("ยังไม่ได้ตั้งค่าผู้ใช้ (secrets) แอปจึงถูกล็อกไว้เพื่อความปลอดภัย "
                 "ดูวิธีตั้งค่าในไฟล์ README_SETUP.md")
        st.stop()

    with st.form("login"):
        u = st.text_input("ชื่อผู้ใช้").strip().lower()
        p = st.text_input("รหัสผ่าน", type="password")
        ok = st.form_submit_button("เข้าสู่ระบบ", type="primary")
    if ok:
        left = _locked_for(u)
        if left:
            st.error(f"ลองผิดหลายครั้งเกินไป ถูกล็อกชั่วคราว รออีกประมาณ {left // 60 + 1} นาที")
        else:
            stored = users.get(u, _DUMMY)
            if verify_password(p, stored) and u in users:
                _attempts().pop(u, None)
                st.session_state["auth_user"] = u
                st.rerun()
            _record_fail(u)
            time.sleep(1)  # slow down guessing
            st.error("ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง")
    st.stop()
