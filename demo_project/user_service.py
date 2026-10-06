def create_user(payload):
    """Create a user record from an API payload."""
    return {"email": payload.get("email"), "name": payload.get("name")}