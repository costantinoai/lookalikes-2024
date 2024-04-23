# ./modules/__init__.py
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(name)s - %(levelname)s - %(message)s')

print('Modules package initialized')
