import uuid

from geoh5io.objects import Object


class Label(Object):
    __class_id = uuid.UUID("{E79F449D-74E3-4598-9C9C-351A28B8B69E}")

    def __init__(self):
        super().__init__()
        self.target_position = None
        self.label_position = None

    @classmethod
    def static_class_id(cls) -> uuid.UUID:
        return cls.__class_id
