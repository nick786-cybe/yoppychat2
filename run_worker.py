# In run_worker.py

import sys
from huey.bin.huey_consumer import consumer_main

# This script acts as the entry point for the Huey worker.
# When we run "python run_worker.py", Python automatically adds the
# current directory to its path. This allows the consumer_main() function
# to successfully find and import your "tasks.huey" object.

if __name__ == '__main__':
    sys.argv.extend(['--workers', '4', '--delay', '0.1', '--quiet'])
    consumer_main()