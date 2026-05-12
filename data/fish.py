import sqlalchemy as sa

from data.db_session import SqlAlchemyBase


class Fish(SqlAlchemyBase):
    __tablename__ = 'fish'

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    name = sa.Column(sa.String, nullable=False)
    image_filename = sa.Column(sa.String, nullable=False)

    x = sa.Column(sa.Float, nullable=False, default=50.0)
    y = sa.Column(sa.Float, nullable=False, default=50.0)
    vx = sa.Column(sa.Float, nullable=False, default=0.5)
    vy = sa.Column(sa.Float, nullable=False, default=0.5)

    fish_type = sa.Column(sa.String, nullable=False, default='peaceful')
    health = sa.Column(sa.Float, nullable=False, default=100.0)
    ticks_since_kill = sa.Column(sa.Integer, nullable=False, default=0)
