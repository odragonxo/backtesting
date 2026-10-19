"""Create a password hash for .streamlit/secrets.toml:   python make_password.py"""
import getpass

from auth import hash_password

user = input("username: ").strip().lower()
pw = getpass.getpass("password (ซ่อนขณะพิมพ์): ")
if len(pw) < 10:
    raise SystemExit("ใช้รหัสผ่านอย่างน้อย 10 ตัวอักษร")
if pw != getpass.getpass("พิมพ์รหัสผ่านอีกครั้ง: "):
    raise SystemExit("รหัสผ่านไม่ตรงกัน")
print("\nคัดลอกบรรทัดนี้ไปไว้ใน .streamlit/secrets.toml ใต้หัวข้อ [users]:\n")
print(f'{user} = "{hash_password(pw)}"')
