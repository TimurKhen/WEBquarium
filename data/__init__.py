from . import db_session
from . import fish
from . import user

from .fish import Fish
from .user import User

__all_models = [fish, user]