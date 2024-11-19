import chainlit as cl
from typing import Optional

@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: dict,
    default_user: cl.User
) -> Optional[cl.User]:
    if provider_id == "google":
        return cl.User(
            identifier=raw_user_data["email"],
            metadata={
                "name": raw_user_data.get("name"),
                "avatar": raw_user_data.get("picture")
            }
        )
    return None

