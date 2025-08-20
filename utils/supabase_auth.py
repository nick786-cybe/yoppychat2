import os
from dotenv import load_dotenv
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
import logging

logging.basicConfig(level=logging.INFO) # Set logging level

load_dotenv() # Load your .env file

SUPABASE_URL = "https://glmtdjegibqaojifyxzf.supabase.co" #os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdsbXRkamVnaWJxYW9qaWZ5eHpmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTA2NjA2MDUsImV4cCI6MjA2NjIzNjYwNX0.AFqnq49ZBp-jiJ1GEHr4QDNoL0QGw3dPYFRu_2YvNVA" #os.environ.get("SUPABASE_ANON_KEY")

print(f"Test URL: '{SUPABASE_URL}'")
print(f"Test ANON Key (first 10 chars): '{SUPABASE_ANON_KEY[:10]}...' Length: {len(SUPABASE_ANON_KEY)}")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    print("Error: Supabase URL or Anon Key not loaded. Check .env file and script location.")
    exit()

try:
    # Try initializing the client
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
    }
    options = ClientOptions(headers=headers)
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY, options=options)
    print("Supabase client initialized successfully.")

    # Attempt to sign in (use a known user from your Supabase Auth table)
    test_email = "nikhilrathore127@gmail.com" # Use your actual registered email
    test_password = "your_user_password" # Use the correct password for that email

    print(f"Attempting to sign in with email: {test_email}")
    auth_response = supabase.auth.sign_in_with_password({"email": test_email, "password": test_password})
    print("Sign-in attempt completed.")

    if auth_response.user:
        print(f"SUCCESS: Logged in as user: {auth_response.user.email}")
        print(f"User ID: {auth_response.user.id}")
    else:
        print("WARNING: Sign-in response did not contain user data.")
        print(auth_response) # Print the full response for more details

except Exception as e:
    print(f"AN ERROR OCCURRED: {type(e).__name__}: {e}")
    if hasattr(e, 'response') and hasattr(e.response, 'text'):
        print(f"Full response text: {e.response.text}")
    if hasattr(e, 'message'):
        print(f"Error message from exception: {e.message}")