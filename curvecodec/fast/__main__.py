import json

from . import status

print(json.dumps(status(), indent=1))
