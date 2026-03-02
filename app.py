import streamlit as st
import hashlib

st.set_page_config(page_title="Hospital Login", page_icon="🏥")

# ==========================
# Initialize Session State
# ==========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ==========================
# Hospital Staff Accounts
# ==========================
USERS = {
    "doctor1": {
        "password": hashlib.sha256("doc123".encode()).hexdigest(),
        "role": "Doctor"
    },
    "staff1": {
        "password": hashlib.sha256("staff123".encode()).hexdigest(),
        "role": "Staff"
    }
}

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ==========================
# If Already Logged In → Redirect
# ==========================
if st.session_state.logged_in:
    st.success("Already logged in. Redirecting...")
    st.switch_page("pages/dashboard.py")

# ==========================
# Login UI
# ==========================
st.title("🏥 Hospital Lung Disease AI System")
st.subheader("Authorized Staff Login Only")

with st.form("login_form"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.form_submit_button("Login")

if login_button:
    if username in USERS:
        if USERS[username]["password"] == hash_password(password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = USERS[username]["role"]

            st.success("Login Successful")
            st.switch_page("pages/dashboard.py")
        else:
            st.error("Incorrect Password")
    else:
        st.error("User not found")