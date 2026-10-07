def create_user(payload):
    """Create a user record after validating and normalizing its fields."""
    if not isinstance(payload, dict):
        raise ValueError("User payload must be a dictionary.")

    email = payload.get("email")
    name = payload.get("name")
    if not isinstance(email, str) or "@" not in email.strip():
        raise ValueError("A valid email is required.")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("A name is required.")

    return {"email": email.strip().lower(), "name": name.strip()}
