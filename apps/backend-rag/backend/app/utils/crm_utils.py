"""
CRM Utilities
Helper functions for CRM module.
"""

def is_crm_admin(user: dict) -> bool:
    """
    Check if the user has CRM admin privileges.
    
    Args:
        user: User dictionary (from get_current_user)
        
    Returns:
        True if admin, False otherwise.
    """
    if not user:
        return False
        
    # Check role
    role = user.get("role", "").lower()
    if role in ["admin", "superadmin"]:
        return True
        
    # Check specific email (legacy/fallback)
    email = user.get("email", "").lower()
    if email in ["zero@balizero.com", "admin@zantara.io"]:
        return True
        
    return False
